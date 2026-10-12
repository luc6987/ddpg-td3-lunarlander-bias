#!/usr/bin/env python
"""CLI: run the union of some grids, resumably and in parallel (§11).

    uv run python scripts/sweep.py --grid e1,e2 --jobs 8 [--runs-dir runs]
    uv run python scripts/sweep.py --grid e1,e2 --dry-run
    uv run python scripts/sweep.py --status --grid e1,e2

The union is de-duplicated by `run_id`, so a shared condition (E2's `min` arm,
E3's UTD-1 arms, E4's depth-2 arms) is run once. Runs are ordered seed-major, so
an interrupted sweep leaves every condition balanced and the same command
resumes it: `done` and `diverged` runs are skipped, `crashed` and stale
`running` ones start again from a wiped directory.
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from lunar_bias import grids, runner
from lunar_bias.config import RunConfig
from lunar_bias.io import read_json

#: `./run.sh status` without an argument shows the whole core study.
DEFAULT_STATUS_GRIDS = "pilot,e1,e2"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--grid", default=None,
                   help="comma-separated grid names: smoke, pilot, e1, e2, e3, e4")
    p.add_argument("--jobs", type=int, default=1, help="worker processes (default 1)")
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--dry-run", action="store_true",
                   help="list the runs and their current state, run nothing")
    p.add_argument("--status", action="store_true",
                   help="per grid: done / diverged / crashed / running / missing")
    return p.parse_args(argv)


def parse_grids(spec: str) -> list[str]:
    names = [name.strip() for name in spec.split(",") if name.strip()]
    unknown = [name for name in names if name not in grids.GRIDS]
    if unknown:
        raise SystemExit(f"unknown grid(s) {unknown}; known: {sorted(grids.GRIDS)}")
    return names


def select(spec: str) -> list[RunConfig]:
    """Union of the named grids, de-duplicated, in seed-major order."""
    configs = grids.unique([cfg for name in parse_grids(spec)
                            for cfg in grids.GRIDS[name]()])
    configs.sort(key=lambda c: c.seed)  # stable: seed-major, grid order inside a seed
    return configs


def state_of(runs_dir: Path, run_id: str) -> str:
    run_dir = Path(runs_dir) / run_id
    if not run_dir.is_dir():
        return "missing"
    status = read_json(run_dir / "status.json") or {}
    return str(status.get("state") or "missing")


def run_one(payload: dict[str, Any]) -> dict[str, Any]:
    """Run one config. Top-level so that the spawn context can pickle it."""
    import torch

    torch.set_num_threads(1)  # I10: one thread per worker process
    cfg = RunConfig.from_dict(payload["cfg"])
    started = time.time()
    state = runner.run(cfg, Path(payload["runs_dir"]))
    return {"run_id": cfg.run_id, "condition": grids.describe(cfg), "seed": cfg.seed,
            "state": state, "wall_s": time.time() - started}


def print_result(res: dict[str, Any]) -> None:
    print(f"{res['run_id']}  {res['condition']}  seed={res['seed']:2d}  "
          f"{res['state']:8s}  {res['wall_s']:7.1f}s", flush=True)


def cmd_status(spec: str, runs_dir: Path) -> int:
    for name in parse_grids(spec):
        configs = grids.unique(grids.GRIDS[name]())
        counts = Counter(state_of(runs_dir, cfg.run_id) for cfg in configs)
        cells = "  ".join(f"{state}={counts.get(state, 0)}" for state in
                          ("done", "diverged", "crashed", "running", "missing"))
        print(f"{name:6s} n={len(configs):3d}  {cells}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    runs_dir = Path(args.runs_dir)

    if args.status:
        return cmd_status(args.grid or DEFAULT_STATUS_GRIDS, runs_dir)
    if args.grid is None:
        raise SystemExit("--grid is required to run a sweep (e.g. --grid e1,e2)")

    configs = select(args.grid)
    if args.dry_run:
        for cfg in configs:
            print(f"{cfg.run_id}  {grids.describe(cfg)}  seed={cfg.seed:2d}  "
                  f"{state_of(runs_dir, cfg.run_id)}")
        print(f"{len(configs)} unique run(s) in grids {args.grid} "
              f"({', '.join(sorted({c.tag for c in configs}))})")
        return 0

    payloads = [{"cfg": cfg.to_dict(), "runs_dir": str(runs_dir)} for cfg in configs]
    print(f"{len(payloads)} run(s) in grids {args.grid}, jobs={args.jobs}, "
          f"runs-dir={runs_dir}", flush=True)

    results: list[dict[str, Any]] = []
    if args.jobs <= 1:
        for payload in payloads:
            res = run_one(payload)
            print_result(res)
            results.append(res)
    else:
        # One BLAS/torch thread per worker; set before the pool so the workers inherit it.
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.jobs, mp_context=ctx) as pool:
            futures = {pool.submit(run_one, payload): payload for payload in payloads}
            for future in as_completed(futures):
                payload = futures[future]
                try:
                    res = future.result()
                except Exception as exc:  # a worker died: report it, do not lose the run
                    cfg = RunConfig.from_dict(payload["cfg"])
                    res = {"run_id": cfg.run_id, "condition": grids.describe(cfg),
                           "seed": cfg.seed, "state": "crashed", "wall_s": 0.0,
                           "error": repr(exc)}
                print_result(res)
                results.append(res)

    counts = Counter(res["state"] for res in results)
    print("--- " + "  ".join(f"{state}={counts[state]}" for state in sorted(counts)),
          flush=True)
    return 1 if counts.get("crashed") else 0


if __name__ == "__main__":
    sys.exit(main())
