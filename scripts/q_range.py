#!/usr/bin/env python
"""CLI: critic values against the range of observed returns (exploratory X4).

    uv run python scripts/q_range.py --runs-dir runs [--grid e1,e2]

Reads `runs/` only and writes nothing. It prints

- the largest discounted return-to-go G recorded at any probe point of any run
  of the grids, over every checkpoint: an empirical stand-in for the top of the
  realisable value range (LunarLander has no tight closed-form bound);
- per condition, each run's mean online Q1 over the probe points of the
  checkpoints of the second half of training (the window of B), summarised over
  runs, and how many runs exceed that largest G.

van Hasselt et al. (2018, arXiv:1812.02648) call values beyond the realisable
range "soft divergence". This script only counts; whether that reading applies
is the author's call. Post hoc, so exploratory (`docs/protocol.md` §2, X4).
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from lunar_bias.analysis.load import grid_run_ids, load_run, run_dirs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--grid", default="e1,e2", help="comma-separated grid names")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    membership = grid_run_ids()
    wanted: set[str] = set()
    for name in args.grid.split(","):
        wanted |= membership[name.strip()]

    g_max, g_where = -np.inf, ""
    late_q1: dict[str, list[float]] = defaultdict(list)
    for run_dir in run_dirs(Path(args.runs_dir)):
        if run_dir.name not in wanted:
            continue
        run = load_run(run_dir)
        half = run.cfg.max_steps // 2
        per_ckpt: list[float] = []
        for step, probe in sorted(run.probes.items()):
            g = float(probe["G"].max())
            if g > g_max:
                g_max, g_where = g, f"{run.run_id} step {step}"
            if step >= half:
                per_ckpt.append(float(probe["q"][0].mean()))
        if per_ckpt:
            late_q1[run.condition].append(float(np.mean(per_ckpt)))

    print(f"largest G at any probe point: {g_max:.1f} ({g_where})")
    print(f"{'condition':48s} {'n':>3s} {'median':>8s} {'min':>8s} {'max':>8s}  > max G")
    for cond, values in sorted(late_q1.items()):
        v = np.asarray(values)
        print(f"{cond:48s} {v.size:3d} {np.median(v):8.1f} {v.min():8.1f} {v.max():8.1f}"
              f"  {int((v > g_max).sum())}/{v.size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
