#!/usr/bin/env python
"""CLI: V5, the return-noise level of the estimator (`docs/implementation_plan.md` §13).

    uv run python scripts/probe_noise.py --runs-dir runs [--grid e1] [--seeds 0,1]
        [--checkpoint-step 300000]

For each selected run (grid membership and training seed), the deterministic
policy of a saved checkpoint is rolled out on the run's fixed evaluation seeds,
up to the 1 500-step probe horizon. At three points of each rollout
(`n // 4`, `n // 2`, `3 * n // 4`, with `n = min(T, 1000)`) the episode is
replayed from `reset(seed)` to $s_t$ — the replay must land exactly on the stored
$s_t$ (V1) — then the environment's dispersion RNG is re-seeded and the episode
continues to termination or step 1 500. The standard deviation of $G_t$ over the
original sample plus the eight continuations estimates the return noise there.

Outputs, in `runs/_validation/`: `noise_<grid>.csv` (one row per point) and
`noise_<grid>.json` (median and 90th percentile of the per-point standard
deviation, the number of points and the runs used). `analysis/` reads the JSON.
This script only reads the runs; it never trains and never writes into a run
directory.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lunar_bias import grids
from lunar_bias.agents import make_agent
from lunar_bias.config import RunConfig
from lunar_bias.envs import make_env
from lunar_bias.evaluation import returns_to_go
from lunar_bias.io import atomic_write_json, atomic_write_text, read_json

#: Seeds of the re-seeded continuations (V2's mechanism); fixed, so V5 reproduces.
CONTINUATION_SEEDS: tuple[int, ...] = tuple(10_000_000 + k for k in range(8))
#: One row per probed point; `std` is the sample standard deviation (ddof = 1).
NOISE_COLUMNS = ("run_id", "condition", "seed", "episode", "ep_seed", "t", "n",
                 "std", "G_original", "G_mean", "n_samples")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--grid", default="e1", help="grid whose runs are probed")
    p.add_argument("--seeds", default="0,1", help="training seeds, comma-separated")
    p.add_argument("--checkpoint-step", type=int, default=None,
                   help="checkpoint step to load (default: each run's max_steps)")
    return p.parse_args(argv)


def parse_seeds(spec: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(part) for part in spec.split(",") if part.strip())
    except ValueError as exc:
        raise SystemExit(f"--seeds must be comma-separated integers: {exc}") from exc
    if not seeds:
        raise SystemExit("--seeds must not be empty")
    return seeds


def point_positions(n: int) -> tuple[int, ...]:
    """The three probe points of an episode with `n` stored points: n/4, n/2, 3n/4."""
    return tuple(sorted({n // 4, n // 2, 3 * n // 4}))


def rollout(agent: Any, env: Any, seed: int) -> tuple[list[np.ndarray], np.ndarray]:
    """One deterministic-policy episode from `reset(seed)`: (observations, rewards)."""
    obs, _ = env.reset(seed=int(seed))
    observations: list[np.ndarray] = []
    rewards: list[float] = []
    terminated = truncated = False
    while not (terminated or truncated):
        observations.append(obs)
        obs, reward, terminated, truncated, _ = env.step(agent.act(obs))
        rewards.append(float(reward))
    return observations, np.asarray(rewards, dtype=np.float64)


def fork_return(agent: Any, env: Any, seed: int, t: int, rng_seed: int,
                expected_obs: np.ndarray, gamma: float) -> float:
    """Replay to $s_t$, re-seed the dispersion RNG, and return the replayed $G_t$.

    The replay must land bit-identically on the stored $s_t$ (V1): otherwise the
    continuation would not start from the probed state. Re-seeding only the
    environment RNG is exactly V2's negative control.
    """
    obs, _ = env.reset(seed=int(seed))
    for _ in range(t):
        obs, _, terminated, truncated, _ = env.step(agent.act(obs))
        if terminated or truncated:
            raise RuntimeError(f"replay of seed {seed} ended before t={t}")
    np.testing.assert_array_equal(obs, expected_obs)  # V1: bit-identical replay
    env.unwrapped.np_random = np.random.default_rng(rng_seed)  # V2's mechanism
    rewards: list[float] = []
    terminated = truncated = False
    while not (terminated or truncated):
        obs, reward, terminated, truncated, _ = env.step(agent.act(obs))
        rewards.append(float(reward))
    return float(returns_to_go(np.asarray(rewards, dtype=np.float64), gamma)[0])


def point_row(agent: Any, env: Any, seed: int, episode: int, n: int,
              observations: list[np.ndarray], rewards: np.ndarray, t: int,
              gamma: float) -> dict[str, Any]:
    """One probe point: the original $G_t$ plus the eight re-seeded continuations."""
    g = returns_to_go(rewards, gamma)
    samples = [float(g[t])]
    samples += [fork_return(agent, env, seed, t, rng_seed, observations[t], gamma)
                for rng_seed in CONTINUATION_SEEDS]
    values = np.asarray(samples, dtype=np.float64)
    return {"episode": episode, "ep_seed": int(seed), "t": t, "n": n,
            "std": float(values.std(ddof=1)), "G_original": float(values[0]),
            "G_mean": float(values.mean()), "n_samples": len(values)}


def load_agent(run_dir: Path, cfg: RunConfig, checkpoint_step: int) -> Any:
    """A fresh agent carrying the actor and critics of one saved checkpoint."""
    path = run_dir / "checkpoints" / f"step_{checkpoint_step:07d}.pt"
    if not path.exists():
        files = (run_dir / "checkpoints").glob("*.pt")
        available = sorted(p.stem.split("_")[1] for p in files)
        raise RuntimeError(
            f"{run_dir.name}: no checkpoint {path.name}; have {available or 'none'}")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    agent = make_agent(cfg)
    agent.load_state_dict({"actor": saved["actor"], "critics": saved["critics"]})
    return agent


def process_run(run_dir: Path, cfg: RunConfig,
                checkpoint_step: int) -> list[dict[str, Any]]:
    """Every probe point of one run: 3 points per evaluation episode."""
    agent = load_agent(run_dir, cfg, checkpoint_step)
    env = make_env(cfg.evaluation.probe_horizon)
    rows: list[dict[str, Any]] = []
    try:
        for episode, seed in enumerate(cfg.evaluation.seeds):
            observations, rewards = rollout(agent, env, int(seed))
            n = min(len(rewards), cfg.evaluation.train_horizon)
            for t in point_positions(n):
                row = point_row(agent, env, int(seed), episode, n, observations, rewards, t,
                                cfg.algo.gamma)
                rows.append({"run_id": cfg.run_id, "condition": grids.describe(cfg),
                             "seed": cfg.seed, **row})
    finally:
        env.close()
    return rows


def write_outputs(out_dir: Path, grid: str, rows: list[dict[str, Any]],
                  summary: dict[str, Any]) -> tuple[Path, Path]:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(NOISE_COLUMNS), extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    csv_path = out_dir / f"noise_{grid}.csv"
    json_path = out_dir / f"noise_{grid}.json"
    atomic_write_text(csv_path, buf.getvalue())
    atomic_write_json(json_path, summary)
    return csv_path, json_path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    torch.set_num_threads(1)  # I10: one torch thread per process
    runs_dir = Path(args.runs_dir)
    if args.grid not in grids.GRIDS:
        raise SystemExit(f"unknown grid {args.grid!r}; known: {sorted(grids.GRIDS)}")
    seeds = parse_seeds(args.seeds)
    configs = [cfg for cfg in grids.unique(grids.GRIDS[args.grid]()) if cfg.seed in seeds]
    configs.sort(key=lambda c: c.seed)
    if not configs:
        raise SystemExit(f"no run of grid {args.grid!r} has a seed in {list(seeds)}")

    step_note = "max_steps" if args.checkpoint_step is None else str(args.checkpoint_step)
    print(f"# V5 return noise: grid={args.grid}  seeds={list(seeds)}  "
          f"checkpoint-step={step_note}  runs-dir={runs_dir}", flush=True)

    rows: list[dict[str, Any]] = []
    used: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for grid_cfg in configs:
        run_dir = runs_dir / grid_cfg.run_id
        state = (read_json(run_dir / "status.json") or {}).get("state", "missing")
        if state != "done":
            skipped.append({"run_id": grid_cfg.run_id, "state": state})
            print(f"skip {grid_cfg.run_id}  {grids.describe(grid_cfg)}  "
                  f"seed={grid_cfg.seed}  state={state} "
                  f"(only `done` runs have a final checkpoint)", flush=True)
            continue
        cfg = RunConfig.from_dict(json.loads((run_dir / "config.json").read_text()))
        if cfg.run_id != run_dir.name:
            raise RuntimeError(f"{run_dir}: config.json hashes to {cfg.run_id}")
        checkpoint_step = args.checkpoint_step if args.checkpoint_step is not None \
            else cfg.max_steps
        run_rows = process_run(run_dir, cfg, checkpoint_step)
        rows.extend(run_rows)
        stds = np.asarray([row["std"] for row in run_rows], dtype=np.float64)
        used.append({"run_id": cfg.run_id, "condition": grids.describe(cfg),
                     "seed": cfg.seed, "checkpoint_step": checkpoint_step})
        print(f"{cfg.run_id}  {grids.describe(cfg)}  seed={cfg.seed}  "
              f"checkpoint={checkpoint_step:7d}  points={len(run_rows):4d}  "
              f"std median={np.median(stds):6.3f}  p90={np.percentile(stds, 90):6.3f}",
              flush=True)

    stds = np.asarray([row["std"] for row in rows], dtype=np.float64)
    summary = {
        "grid": args.grid,
        "seeds": list(seeds),
        "checkpoint_step": args.checkpoint_step,
        "std_ddof": 1,
        "continuation_seeds": list(CONTINUATION_SEEDS),
        "n_runs": len(used),
        "n_points": len(rows),
        "median": float(np.median(stds)) if len(stds) else None,
        "p90": float(np.percentile(stds, 90)) if len(stds) else None,
        "runs_used": [run["run_id"] for run in used],
        "runs": used,
        "skipped": skipped,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    out_dir = runs_dir / "_validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path, json_path = write_outputs(out_dir, args.grid, rows, summary)

    print(f"\nV5: {len(rows)} point(s) over {len(used)} run(s); "
          f"per-point std of G (9 samples, ddof=1): "
          f"median={summary['median']}  p90={summary['p90']}")
    print(f"    runs used: {', '.join(summary['runs_used']) or '(none)'}")
    if skipped:
        print(f"    skipped: {len(skipped)} run(s) not done")
    print(f"    wrote {csv_path} and {json_path}")
    return 0 if rows else 1


if __name__ == "__main__":
    sys.exit(main())
