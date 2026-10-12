"""Per-run summaries of `docs/protocol.md` §1.4, computed from raw artifacts.

| symbol | definition |
| --- | --- |
| `checkpoint_bias` | per checkpoint: `mean(q - G)`, `sum(q - G)/sum|G|`, mean q, mean G |
| `P` | mean `return_1000` over the last five evaluation steps |
| `B` | mean Q1 bias over the checkpoints with `step >= max_steps / 2` |
| `B_mid` | the same over `max_steps / 3 <= step <= max_steps / 2` |
| `horizon_artefact` | V6: what the 1 000-step horizon would have truncated |
| `stability` | V7: `B` from 10 instead of 20 probe episodes |
| `stability_by_condition` | V7 and measurement precision, per E1 condition |

$G_t$ is one unbiased sample of $Q^\\pi(s_t, a_t)$, never an exact value: the
environment's transitions are stochastic (`docs/protocol.md` §1.1).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd

from lunar_bias.analysis.load import RunData
from lunar_bias.analysis.stats import cluster_bootstrap

Critic = Literal["q1", "qmin"]

#: Steps used by `P` (protocol §1.4): the last five evaluation steps.
P_STEPS = 5
#: Probe episodes used by the V7 first half.
V7_EPISODES = 10
#: Episode-clustered bootstrap resamples of one checkpoint (protocol §6).
CLUSTER_BOOTSTRAP_N = 2000


def critic_values(npz: dict[str, np.ndarray], critic: Critic = "q1") -> np.ndarray:
    """The critic row: Q1 (row 0) or the per-point minimum over the critics."""
    q = npz["q"]
    if critic == "q1":
        return q[0]
    if critic == "qmin":
        return q.min(axis=0)
    raise ValueError(f"unknown critic {critic!r}")


def checkpoint_bias(npz: dict[str, np.ndarray],
                    critic: Critic = "q1") -> dict[str, float]:
    """Bias of one checkpoint (`docs/implementation_plan.md` §12).

    `norm_bias` divides by `sum|G|`, which cannot vanish when returns cross zero.
    """
    q = np.asarray(critic_values(npz, critic), dtype=np.float64)
    g = np.asarray(npz["G"], dtype=np.float64)
    diff = q - g
    return {"bias": float(diff.mean()),
            "norm_bias": float(diff.sum() / np.abs(g).sum()),
            "mean_q": float(q.mean()), "mean_G": float(g.mean()),
            "n_points": int(len(g))}


def checkpoint_bias_table(run: RunData, critic: Critic = "q1",
                          episodes: Sequence[int] | None = None) -> pd.DataFrame:
    """One row per checkpoint: `step, bias, mean_q, mean_G, n_points`.

    `episodes` restricts the probe points to those episode indices (V7 uses the
    first ten of the twenty).
    """
    rows = []
    for step in sorted(run.probes):
        npz = run.probes[step]
        if episodes is not None:
            keep = np.isin(npz["episode"], np.asarray(list(episodes)))
            npz = {**npz, "G": npz["G"][keep], "q": npz["q"][:, keep]}
        rows.append({"step": int(step), **checkpoint_bias(npz, critic)})
    return pd.DataFrame(rows, columns=["step", "bias", "norm_bias", "mean_q",
                                       "mean_G", "n_points"])


def _window(run: RunData, low: tuple[int, int], high: tuple[int, int] | None,
            critic: Critic = "q1") -> pd.DataFrame:
    """Checkpoint rows with `low * max_steps <= step [<= high * max_steps]`.

    The fractions are integers (`(1, 2)` is a half) so that a checkpoint exactly
    on the boundary is never lost to floating-point rounding.
    """
    table = checkpoint_bias_table(run, critic)
    if table.empty:
        return table
    step = table["step"].to_numpy()
    keep = step * low[1] >= low[0] * run.cfg.max_steps
    if high is not None:
        keep &= step * high[1] <= high[0] * run.cfg.max_steps
    return table.loc[keep]


def B(run: RunData, critic: Critic = "q1") -> float | None:
    """Mean checkpoint bias over the second half of training (`step >= max/2`)."""
    table = _window(run, (1, 2), None, critic)
    if table.empty:
        return None
    return float(table["bias"].mean())


def B_mid(run: RunData, critic: Critic = "q1") -> float | None:
    """Mean checkpoint bias over `max/3 <= step <= max/2` (exploratory only)."""
    table = _window(run, (1, 3), (1, 2), critic)
    if table.empty:
        return None
    return float(table["bias"].mean())


def P(run: RunData) -> float | None:
    """Mean `return_1000` over the last five evaluation steps present.

    For a run that finished at `max_steps` these are exactly 260 k–300 k steps
    (`docs/protocol.md` §1.4); a run that has not finished has no defined $P$
    and the caller excludes it.
    """
    if run.eval.empty:
        return None
    steps = sorted(run.eval["step"].unique())[-P_STEPS:]
    rows = run.eval[run.eval["step"].isin(steps)]
    return float(rows["return_1000"].mean())


def landing_rate(run: RunData) -> float | None:
    """Fraction of landings over the last five evaluation steps (X4)."""
    if run.eval.empty:
        return None
    steps = sorted(run.eval["step"].unique())[-P_STEPS:]
    rows = run.eval[run.eval["step"].isin(steps)]
    return float((rows["outcome"] == "landed").mean())


def final_return(run: RunData) -> float | None:
    """Mean `return_1000` of the last evaluation step alone (monitoring)."""
    if run.eval.empty:
        return None
    last = run.eval["step"].max()
    return float(run.eval.loc[run.eval["step"] == last, "return_1000"].mean())


def horizon_artefact(npz: dict[str, np.ndarray]) -> dict[str, float]:
    """V6: the tail a 1 000-step horizon would have dropped, in return units.

    For an episode with `ep_len > train_horizon` and a point at `t`,
    `G_t - G_t^1000 = gamma**(train_horizon - t) * G_{train_horizon}`. Episodes
    that ended before the horizon contribute nothing: nothing was truncated.
    """
    lengths = np.asarray(npz["ep_len"])
    episode = np.asarray(npz["episode"])
    t = np.asarray(npz["t"], dtype=np.float64)
    gamma = float(npz["gamma"])
    train_horizon = float(npz["train_horizon"])
    g_at_h = np.asarray(npz["ep_G_at_train_horizon"], dtype=np.float64)
    running = np.isin(episode, np.flatnonzero(lengths > train_horizon))
    if not running.any():
        return {"mean": 0.0, "max": 0.0, "n_points": 0,
                "n_episodes": int((lengths > train_horizon).sum())}
    artefact = gamma ** (train_horizon - t[running]) * g_at_h[episode[running]]
    return {"mean": float(artefact.mean()), "max": float(artefact.max()),
            "n_points": int(running.sum()),
            "n_episodes": int((lengths > train_horizon).sum())}


def stability(run: RunData, critic: Critic = "q1") -> dict[str, float | None]:
    """V7: `B` from the first 10 probe episodes minus `B` from all 20."""
    full = checkpoint_bias_table(run, critic)
    half = checkpoint_bias_table(run, critic, episodes=range(V7_EPISODES))
    b_all = _mean_second_half(run, full)
    b_half = _mean_second_half(run, half)
    diff = None if (b_all is None or b_half is None) else float(b_half - b_all)
    return {"B_all": b_all, "B_first10": b_half, "diff": diff,
            "n_checkpoints": int(len(full))}


def _mean_second_half(run: RunData, table: pd.DataFrame) -> float | None:
    if table.empty:
        return None
    keep = table["step"].to_numpy() * 2 >= run.cfg.max_steps
    if not keep.any():
        return None
    return float(table.loc[keep, "bias"].mean())


def _checkpoint_half_widths(run: RunData, critic: Critic,
                            n_boot: int) -> list[float]:
    """95 % episode-clustered bootstrap half-widths of the run's checkpoints."""
    out: list[float] = []
    for step in sorted(run.probes):
        npz = run.probes[step]
        values = (np.asarray(critic_values(npz, critic), dtype=np.float64)
                  - np.asarray(npz["G"], dtype=np.float64))
        _, lo, hi = cluster_bootstrap(values, npz["episode"], n_boot=n_boot,
                                      rng=np.random.default_rng(0))
        out.append((hi - lo) / 2.0)
    return out


def stability_by_condition(conditions: Mapping[str, Sequence[RunData]],
                           critic: Critic = "q1",
                           n_boot: int = CLUSTER_BOOTSTRAP_N
                           ) -> dict[str, dict[str, Any]]:
    """V7 and the measurement-precision statement, one row per E1 condition.

    Every quantity is **within-condition**: the between-run sd of `B`, the
    median and p90 of `|B(10 episodes) - B(20 episodes)|`, the fraction of runs
    whose `|dB|` is below that sd, and the median per-checkpoint 95 %
    episode-clustered bootstrap half-width. Pooling the four E1 conditions would
    mix DDPG+LN (`B` around 1000) with TD3 (`B` around -10), inflate the sd and
    let V7 pass trivially; the reference sd of V7 is therefore the condition's
    own.

    V7 passes for a condition when its median `|dB|` is below that sd. A
    condition with fewer than two runs that have a defined `B` has no sd to
    compare with: `passes` is then `None`, never a silent True.
    """
    out: dict[str, dict[str, Any]] = {}
    for name, runs in conditions.items():
        bs: list[float] = []
        diffs: list[float] = []
        half_widths: list[float] = []
        for run in runs:
            b = B(run, critic)
            if b is not None:
                bs.append(float(b))
            stat = stability(run, critic)
            if stat["diff"] is not None:
                diffs.append(float(stat["diff"]))
            half_widths.extend(_checkpoint_half_widths(run, critic, n_boot))
        row: dict[str, Any] = {
            "n_runs": len(bs),
            "n_checkpoints": len(half_widths),
            "median_half_width": (float(np.median(half_widths))
                                  if half_widths else None),
            "p90_half_width": (float(np.percentile(half_widths, 90))
                               if half_widths else None),
        }
        if len(bs) < 2 or not diffs:
            row |= {"between_run_sd_B": None, "median_abs_diff": None,
                    "p90_abs_diff": None, "fraction_runs_below_sd": None,
                    "passes": None}
            out[name] = row
            continue
        sd = float(np.std(bs, ddof=1))
        abs_diffs = np.abs(np.asarray(diffs, dtype=np.float64))
        row |= {
            "between_run_sd_B": sd,
            "median_abs_diff": float(np.median(abs_diffs)),
            "p90_abs_diff": float(np.percentile(abs_diffs, 90)),
            "fraction_runs_below_sd": float(np.mean(abs_diffs < sd)),
            "passes": bool(np.median(abs_diffs) < sd),
        }
        out[name] = row
    return out
