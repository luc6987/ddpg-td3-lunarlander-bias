"""`runs/` -> a run index and tidy tables (`docs/implementation_plan.md` §12).

The analysis reads only what training wrote (`config.json`, `status.json`,
`eval_episodes.csv`, `metrics.csv` and `probe/*.npz`) and never trains. Every
condition is identified from config fields or from grid membership — never from
`tag`, which is not part of `run_id` and only records the first grid that
created a run.

Two kinds of incompleteness are possible while a sweep is running: a run
directory whose `.npz` is mid-write, and a run whose `status.json` says
`running`. Nothing is imputed: unreadable artifacts are dropped with a notice,
and the caller decides which states enter which statistic.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lunar_bias import grids
from lunar_bias.config import RunConfig

#: One row per run directory (`docs/implementation_plan.md` §12).
INDEX_COLUMNS = ("run_id", "state", "algo", "critic_norm", "reduction", "utd",
                 "depth", "width", "seed", "max_steps", "step", "wall_s", "grids")


def grid_run_ids() -> dict[str, set[str]]:
    """The `run_id`s of every named grid, for membership tests."""
    return {name: {cfg.run_id for cfg in factory()}
            for name, factory in grids.GRIDS.items()}


def format_grids(run_id: str, membership: dict[str, set[str]] | None = None) -> str:
    """Comma-joined names of the grids containing `run_id` ("" when none)."""
    membership = grid_run_ids() if membership is None else membership
    return ",".join(name for name, ids in membership.items() if run_id in ids)


def run_dirs(runs_dir: Path) -> list[Path]:
    """Directories that look like a run: they contain a `config.json`."""
    runs_dir = Path(runs_dir)
    return sorted(p for p in runs_dir.iterdir()
                  if p.is_dir() and (p / "config.json").is_file())


def read_config(run_dir: Path) -> RunConfig:
    """The run's frozen config; raises if `config.json` does not hash to the name."""
    cfg = RunConfig.from_dict(json.loads((run_dir / "config.json").read_text()))
    if cfg.run_id != run_dir.name:
        raise RuntimeError(f"{run_dir}: config.json hashes to {cfg.run_id}, "
                           f"not {run_dir.name}")
    return cfg


def read_status(run_dir: Path) -> dict[str, Any]:
    """`status.json`, or an empty dict when it is missing or unreadable."""
    try:
        return json.loads((run_dir / "status.json").read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def load_index(runs_dir: Path) -> pd.DataFrame:
    """One row per run directory (`INDEX_COLUMNS`); `grids` lists grid membership."""
    membership = grid_run_ids()
    rows: list[dict[str, Any]] = []
    for run_dir in run_dirs(runs_dir):
        cfg = read_config(run_dir)
        status = read_status(run_dir)
        rows.append({
            "run_id": cfg.run_id, "state": status.get("state", "missing"),
            "algo": cfg.algo.name, "critic_norm": cfg.critic.norm,
            "reduction": cfg.algo.target_reduction, "utd": cfg.algo.utd,
            "depth": len(cfg.critic.hidden), "width": cfg.critic.hidden[0],
            "seed": cfg.seed, "max_steps": cfg.max_steps,
            "step": int(status.get("step", 0)),
            "wall_s": float(status.get("wall_s", float("nan"))),
            "grids": format_grids(cfg.run_id, membership),
        })
    frame = pd.DataFrame(rows, columns=list(INDEX_COLUMNS))
    return frame.sort_values(["seed", "algo", "critic_norm"]).reset_index(drop=True)


def load_eval_episodes(run_dir: Path) -> pd.DataFrame:
    """`eval_episodes.csv`; an empty frame with the right columns when absent."""
    columns = ["step", "episode_index", "seed", "return_1000", "length_1000", "outcome"]
    path = Path(run_dir) / "eval_episodes.csv"
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame(columns=columns)


def load_metrics(run_dir: Path) -> pd.DataFrame:
    """`metrics.csv`; an empty frame when absent."""
    path = Path(run_dir) / "metrics.csv"
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame()


def probe_steps(run_dir: Path) -> list[int]:
    """Sorted steps that have a probe `.npz`."""
    out: list[int] = []
    for path in (Path(run_dir) / "probe").glob("step_*.npz"):
        try:
            out.append(int(path.stem.split("_")[1]))
        except (IndexError, ValueError):
            continue
    return sorted(out)


def load_probe(run_dir: Path, step: int) -> dict[str, np.ndarray] | None:
    """The probe arrays of one checkpoint, or None if the file is unreadable.

    A sweep writing this file concurrently can leave a half-written zip; the
    caller records the drop rather than crashing the whole analysis.
    """
    path = Path(run_dir) / "probe" / f"step_{step:07d}.npz"
    try:
        with np.load(path) as data:
            return {key: data[key] for key in data.files}
    except (OSError, ValueError, EOFError) as exc:
        warnings.warn(f"{path}: unreadable probe ({exc})", stacklevel=2)
        return None


@dataclass
class RunData:
    """Everything the analysis needs from one run, loaded once."""

    run_id: str
    cfg: RunConfig
    state: str
    step: int
    wall_s: float
    eval: pd.DataFrame
    metrics: pd.DataFrame
    probes: dict[int, dict[str, np.ndarray]] = field(default_factory=dict)

    @property
    def condition(self) -> str:
        """Condition label built from config fields (never from `tag`)."""
        return grids.describe(self.cfg)

    @property
    def complete(self) -> bool:
        """True when the run stopped as `done` and probed through `max_steps`."""
        return self.state == "done" and self.step >= self.cfg.max_steps


def load_run(run_dir: Path, with_probes: bool = True) -> RunData:
    """Load one run directory; `with_probes=False` skips the (larger) probe npz."""
    run_dir = Path(run_dir)
    cfg = read_config(run_dir)
    status = read_status(run_dir)
    probes: dict[int, dict[str, np.ndarray]] = {}
    if with_probes:
        for step in probe_steps(run_dir):
            arrays = load_probe(run_dir, step)
            if arrays is not None:
                probes[step] = arrays
    return RunData(run_id=cfg.run_id, cfg=cfg, state=status.get("state", "missing"),
                   step=int(status.get("step", 0)),
                   wall_s=float(status.get("wall_s", float("nan"))),
                   eval=load_eval_episodes(run_dir), metrics=load_metrics(run_dir),
                   probes=probes)


def load_runs(runs_dir: Path, run_ids: list[str] | None = None,
              with_probes: bool = True) -> dict[str, RunData]:
    """`{run_id: RunData}` for the whole runs directory, or for `run_ids` only."""
    wanted = set(run_ids) if run_ids is not None else None
    out: dict[str, RunData] = {}
    for run_dir in run_dirs(runs_dir):
        if wanted is not None and run_dir.name not in wanted:
            continue
        out[run_dir.name] = load_run(run_dir, with_probes=with_probes)
    return out
