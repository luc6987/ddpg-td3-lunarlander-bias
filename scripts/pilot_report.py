#!/usr/bin/env python
"""CLI: the Gate G1 table, read from `runs/` only (`docs/implementation_plan.md` §13).

    uv run python scripts/pilot_report.py --runs-dir runs [--grid pilot]

Per run it prints the state, the wall time, P (the pre-registered performance
summary: mean return over the last five checkpoints), the last evaluation and
its landing rate, and the Q1 bias at the checkpoints nearest to 0, max_steps/2
and max_steps. It then states V4 (the random-critic sanity check at step 0, for
one TD3 run) and applies the G1 rule **mechanically**:

    PASS if the mean P of the TD3 runs >= 100 and every run is done.

This script does not interpret anything and writes nothing: the gate decision
belongs to the author (docs/protocol.md §7).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from lunar_bias import grids
from lunar_bias.config import RunConfig
from lunar_bias.io import read_json

#: G1 threshold on the mean P of the TD3 runs (docs/protocol.md §7).
G1_THRESHOLD = 100.0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--grid", default="pilot", help="grid whose runs are reported")
    return p.parse_args(argv)


def read_eval_rows(run_dir: Path) -> list[dict[str, str]]:
    path = run_dir / "eval_episodes.csv"
    if not path.exists():
        return []
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def evaluation_steps(rows: list[dict[str, str]]) -> list[int]:
    return sorted({int(row["step"]) for row in rows})


def performance(rows: list[dict[str, str]], n_last: int = 5) -> float:
    """P: mean `return_1000` over the last `n_last` evaluation steps (protocol §1.4)."""
    steps = evaluation_steps(rows)[-n_last:]
    values = [float(row["return_1000"]) for row in rows if int(row["step"]) in steps]
    return float(np.mean(values)) if values else float("nan")


def nearest(steps: list[int], target: float) -> int | None:
    """The available evaluation step closest to `target` (smoke grids have few)."""
    return min(steps, key=lambda s: abs(s - target)) if steps else None


def probe_bias(path: Path) -> tuple[float, float, float]:
    """(mean Q1, mean G, mean Q1 - G) of one probe file. Row 0 of `q` is Q1 (I8)."""
    with np.load(path) as data:
        q1, g = data["q"][0], data["G"]
    return float(q1.mean()), float(g.mean()), float((q1 - g).mean())


def load_run(run_dir: Path) -> dict[str, Any]:
    cfg = RunConfig.from_dict(json.loads((run_dir / "config.json").read_text()))
    status = read_json(run_dir / "status.json") or {}
    rows = read_eval_rows(run_dir)
    return {
        "run_id": cfg.run_id, "dir": run_dir, "cfg": cfg, "status": status, "rows": rows,
        "state": status.get("state", "missing"),
        "wall_s": status.get("wall_s"),
        "P": performance(rows),
        "steps": evaluation_steps(rows),
        "probes": {int(p.stem.split("_")[1]): p for p in (run_dir / "probe").glob("*.npz")},
    }


def last_evaluation(rows: list[dict[str, str]]) -> tuple[float, int, int]:
    """(mean return, landed, n) of the last evaluation step."""
    steps = evaluation_steps(rows)
    if not steps:
        return float("nan"), 0, 0
    last = [row for row in rows if int(row["step"]) == steps[-1]]
    returns = [float(row["return_1000"]) for row in last]
    landed = sum(1 for row in last if row["outcome"] == "landed")
    return float(np.mean(returns)), landed, len(last)


def print_run(run: dict[str, Any]) -> None:
    cfg: RunConfig = run["cfg"]
    steps = sorted(run["probes"])
    targets = {"b0": 0.0, "bmid": cfg.max_steps / 2, "bend": float(cfg.max_steps)}
    biases: dict[str, float] = {}
    for name, target in targets.items():
        step = nearest(steps, target)
        if step is not None:
            biases[name] = probe_bias(run["probes"][step])[2]
    ret, landed, n = last_evaluation(run["rows"])
    wall = "     n/a" if run["wall_s"] is None else f"{run['wall_s'] / 60:7.1f}m"
    print(f"{run['run_id']}  {cfg.algo.name:4s}  norm={cfg.critic.norm:5s}  "
          f"seed={cfg.seed}  {run['state']:8s}  wall={wall}  P={run['P']:8.2f}  "
          f"last={ret:8.2f}  landed={landed:2d}/{n:<2d}  "
          f"bias@0={biases.get('b0', float('nan')):8.2f}  "
          f"bias@mid={biases.get('bmid', float('nan')):8.2f}  "
          f"bias@end={biases.get('bend', float('nan')):8.2f}")


def print_v4(run: dict[str, Any]) -> None:
    """V4: at step 0 the critic is random, so the bias must be large and equal
    mean(Q1) - mean(G) — and one explicit point is shown so it can be checked by hand."""
    path = run["probes"].get(0)
    if path is None:
        print(f"\nV4 (step 0): no step-0 probe for run {run['run_id']}")
        return
    mean_q1, mean_g, bias = probe_bias(path)
    with np.load(path) as data:
        q1, g, t = data["q"][0], data["G"], data["t"]
        episode = data["episode"]
    first = np.flatnonzero((episode == 0) & (t == 0))
    i = int(first[0]) if len(first) else 0
    print(f"\nV4 (step 0, run {run['run_id']}, {run['cfg'].algo.name}/"
          f"{run['cfg'].critic.norm}): mean Q1 = {mean_q1:.3f}, mean G = {mean_g:.3f}, "
          f"bias = {bias:.3f}, n_points = {len(g)}")
    diff = float(q1[i] - g[i])
    print(f"    example probe point: episode={int(episode[i])} t={int(t[i])} "
          f"Q1 = {float(q1[i]):.4f} G = {float(g[i]):.4f} Q1 - G = {diff:.4f}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    runs_dir = Path(args.runs_dir)
    if args.grid not in grids.GRIDS:
        raise SystemExit(f"unknown grid {args.grid!r}; known: {sorted(grids.GRIDS)}")
    configs = grids.unique(grids.GRIDS[args.grid]())
    configs.sort(key=lambda c: c.seed)

    print(f"# grid={args.grid}  runs-dir={runs_dir}  {len(configs)} run(s)")
    runs = []
    for cfg in configs:
        run_dir = runs_dir / cfg.run_id
        if not run_dir.is_dir():
            print(f"{cfg.run_id}  {cfg.algo.name:4s}  norm={cfg.critic.norm:5s}  "
                  f"seed={cfg.seed}  {'missing':8s}")
            continue
        runs.append(load_run(run_dir))
        print_run(runs[-1])

    td3 = [run for run in runs if run["cfg"].algo.name == "td3"]
    all_done = bool(runs) and all(run["state"] == "done" for run in runs) \
        and len(runs) == len(configs)
    mean_p = float(np.mean([run["P"] for run in td3])) if td3 else float("nan")
    passed = bool(all_done and np.isfinite(mean_p) and mean_p >= G1_THRESHOLD)

    if td3:
        print_v4(td3[0])

    print("\nG1 rule (docs/protocol.md §7): PASS if the mean P of the TD3 runs >= "
          f"{G1_THRESHOLD:g} and every run is done.")
    print(f"    TD3 runs: {len(td3)}, mean P = {mean_p:.2f}")
    print(f"    every run done: {'yes' if all_done else 'no'} "
          f"({sum(1 for r in runs if r['state'] == 'done')}/{len(configs)} done, "
          f"{len(configs) - len(runs)} missing)")
    print(f"    G1 = {'PASS' if passed else 'FAIL'}")

    if not configs:
        print("no runs in this grid", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
