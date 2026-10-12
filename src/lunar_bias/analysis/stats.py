"""Statistical primitives of the analysis (`docs/implementation_plan.md` §12).

The plan fixes the tests before any result is seen (`docs/protocol.md` §6):
Welch's two-sided t-test with Holm within a family, Hedges' g, the exact
Mann–Whitney U p-value as a robustness check, Page's trend test for E2, and an
episode-clustered bootstrap used only for a measurement-precision sentence.

Nothing here touches the runs; these are pure functions of arrays, so they can
be checked against hand-computed values (`tests/test_stats.py`).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import stats


def t_interval(x: Sequence[float] | np.ndarray,
               level: float = 0.95) -> tuple[float, float, float]:
    """Mean and two-sided Student t-interval of `x` (across runs, ddof = 1)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    m = float(x.mean())
    h = float(stats.t.ppf(0.5 + level / 2, n - 1) * x.std(ddof=1) / np.sqrt(n))
    return m, m - h, m + h


def describe(x: Sequence[float] | np.ndarray,
             level: float = 0.95) -> dict[str, float | int | list[float]]:
    """Per-condition descriptives: mean, 95 % t-interval, median, sd and n."""
    x = np.asarray(x, dtype=float)
    mean, lo, hi = t_interval(x, level)
    return {"n": int(len(x)), "mean": mean, "ci_lo": lo, "ci_hi": hi,
            "median": float(np.median(x)) if len(x) else float("nan"),
            "sd": float(x.std(ddof=1)) if len(x) > 1 else float("nan"),
            "values": [float(v) for v in x]}


def hedges_g(x: Sequence[float] | np.ndarray, y: Sequence[float] | np.ndarray) -> float:
    """Hedges' g for the contrast `y - x`, with the small-sample correction."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    nx, ny = len(x), len(y)
    sp = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1))
                 / (nx + ny - 2))
    return float((np.mean(y) - np.mean(x)) / sp * (1 - 3 / (4 * (nx + ny) - 9)))


def welch(x: Sequence[float] | np.ndarray, y: Sequence[float] | np.ndarray,
          level: float = 0.95) -> dict[str, float]:
    """Contrast `y - x`: Welch t-test (primary) + Mann-Whitney U (robustness)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    r = stats.ttest_ind(y, x, equal_var=False)
    ci = r.confidence_interval(level)
    return {"delta": float(np.mean(y) - np.mean(x)), "ci_lo": float(ci.low),
            "ci_hi": float(ci.high), "t": float(r.statistic), "df": float(r.df),
            "p": float(r.pvalue), "hedges_g": hedges_g(x, y),
            "mw_p": float(stats.mannwhitneyu(y, x, alternative="two-sided").pvalue),
            "n_x": int(len(x)), "n_y": int(len(y))}


def holm(pvals: Sequence[float] | np.ndarray) -> np.ndarray:
    """Holm–Bonferroni adjusted p-values, monotone in the sorted order."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(np.argsort(p)):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj


def holm_family(results: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    """Add `p_holm` to each Welch result of one family, in place, and return it.

    Family A (bias) and Family B (performance) each contain three contrasts and
    are corrected separately (`docs/protocol.md` §6). A contrast that could not
    be computed — fewer than two runs in an arm — carries an `error` instead of
    a p-value and is left untouched: it is reported, never imputed.
    """
    names = [name for name, result in results.items() if "p" in result]
    if not names:
        return results
    adj = holm([results[name]["p"] for name in names])
    for name, p_adj in zip(names, adj, strict=True):
        results[name]["p_holm"] = float(p_adj)
    return results


def page(blocks: np.ndarray,
         predicted_ranks: Sequence[int]) -> dict[str, float | bool | str]:
    """Page's trend test; `blocks` is [n_seeds, 3], columns ordered min, mean, max.

    Rank 1 is the smallest value. Bias (C0) predicts [1, 2, 3] and performance
    (C1) predicts [3, 2, 1]. A diverged run is encoded as +inf for $B$ and -inf
    for $P$ before calling, so it takes the worst predicted rank.
    """
    blocks = np.asarray(blocks, dtype=float)
    # scipy's exact p-value casts L to int, which silently truncates a
    # fractional L; L is fractional only when a block contains tied values
    # (e.g. two diverged runs at the same rank in one seed). Fall back to the
    # asymptotic p-value there and say so in the result.
    tied = bool(any(len(np.unique(row)) < len(row) for row in blocks))
    r = stats.page_trend_test(blocks, predicted_ranks=list(predicted_ranks),
                              method="asymptotic" if tied else "auto")
    return {"L": float(r.statistic), "p": float(r.pvalue),
            "n_blocks": int(blocks.shape[0]), "method": str(r.method), "ties": tied}


def cluster_bootstrap(values: Sequence[float] | np.ndarray,
                      clusters: Sequence[int] | np.ndarray | None = None,
                      n_boot: int = 2000, level: float = 0.95,
                      rng: np.random.Generator | None = None
                      ) -> tuple[float, float, float]:
    """Percentile interval for the pooled mean, resampling whole clusters.

    The statistic is the pooled mean (every probe point weighs the same), not a
    mean of per-cluster means. With `clusters=None` each observation is its own
    cluster, i.e. the ordinary i.i.d. bootstrap; the clustered one is wider when
    the observations are strongly clustered. Used only for the secondary
    measurement-precision sentence (`docs/protocol.md` §6).
    """
    values = np.asarray(values, dtype=float)
    if clusters is None:
        idx = np.arange(len(values))
    else:
        _, idx = np.unique(np.asarray(clusters), return_inverse=True)
    n_clusters = int(idx.max()) + 1 if len(idx) else 0
    sums = np.bincount(idx, weights=values, minlength=n_clusters)
    counts = np.bincount(idx, minlength=n_clusters)
    rng = np.random.default_rng(0) if rng is None else rng
    picks = rng.integers(0, n_clusters, size=(n_boot, n_clusters))
    totals = counts[picks].sum(axis=1)
    means = np.divide(sums[picks].sum(axis=1), totals, out=np.full(n_boot, np.nan),
                      where=totals > 0)
    lo, hi = np.percentile(means, [100 * (1 - level) / 2, 100 * (1 + level) / 2])
    return float(values.mean()), float(lo), float(hi)
