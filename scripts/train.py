#!/usr/bin/env python
"""CLI: run ONE config (docs/implementation_plan.md §16).

The config is built with `grids.condition`, so a run launched by hand has
exactly the same `run_id` as the same condition inside a grid.

    uv run python scripts/train.py --algo td3 --critic-norm layer --seed 0 \
        [--steps 300000] [--reduction min] [--utd 1] [--depth 2] [--runs-dir runs]
"""

from __future__ import annotations

import argparse
import dataclasses
import sys

from lunar_bias import grids, runner
from lunar_bias.config import RunConfig


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--algo", choices=("ddpg", "td3"), default="td3")
    p.add_argument("--critic-norm", choices=("none", "layer"), default="none")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=300_000)
    p.add_argument("--reduction", choices=("min", "mean", "max"), default="min")
    p.add_argument("--utd", type=int, default=1)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--tag", default="", help="human label; never part of run_id")
    p.add_argument("--runs-dir", default="runs")
    return p.parse_args(argv)


def build_config(args: argparse.Namespace) -> RunConfig:
    cfg = grids.condition(algo=args.algo, norm=args.critic_norm, seed=args.seed,
                          reduction=args.reduction, utd=args.utd, depth=args.depth,
                          tag=args.tag)
    return dataclasses.replace(cfg, max_steps=args.steps)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = build_config(args)
    print(f"run_id={cfg.run_id}  {grids.describe(cfg)}  seed={cfg.seed}  "
          f"steps={cfg.max_steps}  runs-dir={args.runs_dir}", flush=True)
    state = runner.run(cfg, args.runs_dir, raise_on_crash=True)
    print(f"[{cfg.run_id}] {state}", flush=True)
    return 0 if state in ("done", "skipped") else 1


if __name__ == "__main__":
    sys.exit(main())
