"""Artifacts of one run: atomic JSON, append-only CSVs, probes and checkpoints.

Training writes raw data only (`docs/implementation_plan.md` §0.4). Every
statistic is computed afterwards in `analysis/`, from these files.

Layout of `runs/<run_id>/` (`docs/implementation_plan.md` §15):
`config.json`, `meta.json`, `status.json`, `metrics.csv`, `eval_episodes.csv`,
`probe/step_XXXXXXX.npz`, `probe_summary.csv` (monitoring only) and
`checkpoints/step_XXXXXXX.pt`.
"""

from __future__ import annotations

import csv
import json
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from lunar_bias.config import RunConfig
from lunar_bias.evaluation import ProbeData, evaluate_and_probe

if TYPE_CHECKING:  # avoid a circular import: train.py imports this module
    from lunar_bias.evaluation import PolicyAgent

#: Window means of the training loop, one row per `log_every` steps (NaN when empty).
METRICS_COLUMNS = ("step", "wall_s", "n_updates", "n_actor_updates", "critic_loss",
                   "actor_loss", "q1_batch_mean", "q_abs_max", "critic_grad_norm",
                   "train_episodes", "train_return_mean")
#: One row per evaluation episode.
EVAL_COLUMNS = ("step", "episode_index", "seed", "return_1000", "length_1000", "outcome")
#: Monitoring only: the analysis recomputes everything from the probe npz files.
PROBE_SUMMARY_COLUMNS = ("step", "n_points", "mean_G", "mean_q1", "bias_q1")


def atomic_write_text(path: Path, text: str) -> None:
    """Write `path.tmp`, then os.replace: a reader never sees a half-written file."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def atomic_write_json(path: Path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2) + "\n")


def read_json(path: Path) -> dict[str, Any] | None:
    """Parse `path`, or return None if it is missing or unreadable."""
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return None


def append_csv(path: Path, columns: Sequence[str],
               rows: Sequence[Mapping[str, Any]]) -> None:
    """Append `rows` to a CSV, writing the header when the file is created."""
    path = Path(path)
    new = not path.exists()
    with path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(columns), extrasaction="ignore")
        if new:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)
        fh.flush()


class RunWriter:
    """Writes everything a run produces, and prints one line per evaluation."""

    def __init__(self, run_dir: Path, cfg: RunConfig, started_at: float | None = None):
        self.dir = Path(run_dir)
        self.cfg = cfg
        self.started_at = time.time() if started_at is None else started_at
        (self.dir / "probe").mkdir(parents=True, exist_ok=True)
        (self.dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    @property
    def wall_s(self) -> float:
        return time.time() - self.started_at

    def status(self, state: str, step: int = 0, error: str | None = None) -> None:
        atomic_write_json(self.dir / "status.json", {
            "state": state, "step": int(step), "wall_s": round(self.wall_s, 3),
            "error": error, "updated": time.time(),
        })

    def metrics(self, step: int, agent: Any, window: Mapping[str, float]) -> None:
        append_csv(self.dir / "metrics.csv", METRICS_COLUMNS,
                   [{"step": step, "wall_s": round(self.wall_s, 3),
                     "n_updates": agent.n_updates,
                     "n_actor_updates": agent.n_actor_updates, **window}])

    def evaluation(self, step: int, agent: PolicyAgent) -> None:
        ev = self.cfg.evaluation
        data = evaluate_and_probe(agent, ev.seeds, self.cfg.algo.gamma,
                                  ev.train_horizon, ev.probe_horizon)
        append_csv(self.dir / "eval_episodes.csv", EVAL_COLUMNS,
                   [{**row, "step": step} for row in data.episodes])
        np.savez_compressed(self.dir / "probe" / f"step_{step:07d}.npz", **data.arrays)
        append_csv(self.dir / "probe_summary.csv", PROBE_SUMMARY_COLUMNS,
                   [self._summary_row(step, data)])
        self.status("running", step)
        print(self._line(step, data), flush=True)

    def checkpoint(self, step: int, agent: Any) -> None:
        torch.save({"actor": agent.actor.state_dict(),
                    "critics": agent.critics.state_dict(),
                    "step": int(step)},
                   self.dir / "checkpoints" / f"step_{step:07d}.pt")

    # --- helpers -----------------------------------------------------------
    @staticmethod
    def _summary_row(step: int, data: ProbeData) -> dict[str, Any]:
        q1, g = data.arrays["q"][0], data.arrays["G"]
        return {"step": step, "n_points": len(g), "mean_G": float(g.mean()),
                "mean_q1": float(q1.mean()), "bias_q1": float((q1 - g).mean())}

    def _line(self, step: int, data: ProbeData) -> str:
        rows = data.episodes
        ret = float(np.mean([r["return_1000"] for r in rows]))
        landed = sum(1 for r in rows if r["outcome"] == "landed")
        bias = float(np.mean(data.arrays["q"][0] - data.arrays["G"]))
        return (f"[{self.cfg.run_id}] step={step:7d} return={ret:7.1f} "
                f"landed={landed:2d}/{len(rows)} bias_q1={bias:7.1f} "
                f"wall={self.wall_s:5.0f}s")
