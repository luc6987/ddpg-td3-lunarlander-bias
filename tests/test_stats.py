"""The analysis primitives against hand-computed values (`implementation_plan` §14).

Holm on a hand example, the Welch confidence interval against the manual
Welch–Satterthwaite formula, Hedges' g by hand, `page` against scipy, and the
episode-clustered bootstrap against the i.i.d. one on strongly clustered data.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from lunar_bias.analysis.stats import (
    cluster_bootstrap,
    describe,
    hedges_g,
    holm,
    holm_family,
    page,
    t_interval,
    welch,
)


def test_t_interval_matches_manual() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0])
    mean, lo, hi = t_interval(x)
    half = stats.t.ppf(0.975, 3) * x.std(ddof=1) / 2.0
    assert mean == pytest.approx(2.5)
    assert (lo, hi) == pytest.approx((2.5 - half, 2.5 + half))


def test_holm_hand_example() -> None:
    # sorted p = [0.01, 0.03, 0.04]; adjusted = [3*0.01, 2*0.03, 1*0.04] with
    # the monotonicity fixup -> [0.03, 0.06, 0.06] in the sorted order.
    adjusted = holm([0.01, 0.04, 0.03])
    assert adjusted == pytest.approx([0.03, 0.06, 0.06])


def test_holm_family_adds_adjusted_p() -> None:
    family = {"a": {"p": 0.02}, "b": {"p": 0.01}, "c": {"p": 0.5}}
    out = holm_family(family)
    assert out["b"]["p_holm"] == pytest.approx(0.03)
    assert out["a"]["p_holm"] == pytest.approx(0.04)
    assert out["c"]["p_holm"] == pytest.approx(0.5)


def test_holm_family_skips_contrasts_that_could_not_be_computed() -> None:
    # A condition with fewer than two runs gets an `error`, not a p-value; Holm
    # then runs over the remaining contrasts only and never imputes the missing one.
    family = {"a": {"p": 0.02}, "b": {"p": 0.01}, "c": {"error": "not computed: n < 2"}}
    out = holm_family(family)
    assert out["b"]["p_holm"] == pytest.approx(0.02)
    assert out["a"]["p_holm"] == pytest.approx(0.02)
    assert out["c"] == {"error": "not computed: n < 2"}
    assert holm_family({"a": {"error": "n < 2"}}) == {"a": {"error": "n < 2"}}


def test_welch_ci_equals_manual_welch_satterthwaite() -> None:
    x = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    y = np.array([2.0, 3.0, 5.0, 7.0, 11.0, 13.0])
    r = welch(x, y)
    vx, vy = x.var(ddof=1) / len(x), y.var(ddof=1) / len(y)
    df = (vx + vy) ** 2 / (vx**2 / (len(x) - 1) + vy**2 / (len(y) - 1))
    half = stats.t.ppf(0.975, df) * np.sqrt(vx + vy)
    delta = y.mean() - x.mean()
    assert r["delta"] == pytest.approx(delta)
    assert r["df"] == pytest.approx(df)
    assert (r["ci_lo"], r["ci_hi"]) == pytest.approx((delta - half, delta + half))
    assert r["p"] == pytest.approx(stats.ttest_ind(y, x, equal_var=False).pvalue)
    assert r["mw_p"] == pytest.approx(
        stats.mannwhitneyu(y, x, alternative="two-sided").pvalue)
    assert (r["n_x"], r["n_y"]) == (len(x), len(y))


def test_hedges_g_hand_example() -> None:
    # x = 1,2,3 and y = 2,4,6: pooled sd = sqrt(2.5), d = 2 / sqrt(2.5),
    # correction J = 1 - 3 / (4 * 6 - 9) = 0.8.
    g = hedges_g([1.0, 2.0, 3.0], [2.0, 4.0, 6.0])
    expected = (2.0 / np.sqrt(2.5)) * 0.8
    assert g == pytest.approx(expected)


def test_hedges_g_sign_follows_y_minus_x() -> None:
    assert hedges_g([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]) > 0
    assert hedges_g([2.0, 4.0, 6.0], [1.0, 2.0, 3.0]) < 0


def test_describe_shape() -> None:
    out = describe([1.0, 2.0, 3.0])
    assert out["n"] == 3
    assert out["mean"] == pytest.approx(2.0)
    assert out["median"] == pytest.approx(2.0)
    assert (out["ci_lo"], out["ci_hi"]) == pytest.approx(t_interval([1.0, 2.0, 3.0])[1:])


def _trend_blocks() -> np.ndarray:
    """20 blocks with a clear increasing trend in the columns."""
    rng = np.random.default_rng(7)
    base = np.arange(20, dtype=float)[:, None]
    return base * np.array([0.0, 1.0, 2.0]) + rng.normal(0.0, 0.3, size=(20, 3))


def test_page_equals_scipy() -> None:
    blocks = _trend_blocks()
    out = page(blocks, [1, 2, 3])
    ref = stats.page_trend_test(blocks, predicted_ranks=[1, 2, 3])
    assert out["L"] == pytest.approx(float(ref.statistic))
    assert out["p"] == pytest.approx(float(ref.pvalue))
    assert out["n_blocks"] == 20
    assert out["method"] == ref.method
    assert out["ties"] is False
    # the predicted order is real: the increasing order is significant...
    assert out["p"] < 0.01
    # ...and the reversed prediction is not.
    assert page(blocks, [3, 2, 1])["p"] > 0.5


def test_page_ranks_diverged_runs_worst() -> None:
    # C0 encodes a diverged run as +inf: it must take the top rank.
    blocks = _trend_blocks()
    blocks[0, 0] = np.inf
    out = page(blocks, [1, 2, 3])
    assert np.isfinite(out["L"]) and 0.0 <= out["p"] <= 1.0
    # Two diverged runs in the same block tie, so scipy's exact L (which casts a
    # fractional L to int) must not be used; the method is reported, not hidden.
    blocks[0, 1] = np.inf
    tied = page(blocks, [1, 2, 3])
    assert tied["ties"] is True
    assert tied["method"] == "asymptotic"


def test_page_rejects_too_few_blocks() -> None:
    with pytest.raises(ValueError):
        page(np.zeros((1, 3)), [1, 2, 3])  # fewer than two blocks


def test_cluster_bootstrap_wider_than_iid_on_clustered_data() -> None:
    rng = np.random.default_rng(0)
    n_clusters, per_cluster = 8, 50
    centres = rng.normal(0.0, 10.0, size=n_clusters)  # strong between-cluster spread
    values = np.repeat(centres, per_cluster) + rng.normal(
        0.0, 0.1, size=n_clusters * per_cluster)
    clusters = np.repeat(np.arange(n_clusters), per_cluster)
    mean_c, lo_c, hi_c = cluster_bootstrap(values, clusters, n_boot=2000,
                                           rng=np.random.default_rng(1))
    mean_i, lo_i, hi_i = cluster_bootstrap(values, None, n_boot=2000,
                                           rng=np.random.default_rng(1))
    assert mean_c == pytest.approx(values.mean())
    assert mean_i == pytest.approx(values.mean())
    assert (hi_c - lo_c) > 3 * (hi_i - lo_i)
    assert lo_c < mean_c < hi_c


def test_cluster_bootstrap_head_width_shrinks_with_clusters() -> None:
    # 40 clusters of 1 give the i.i.d. interval back (within Monte-Carlo error).
    values = np.arange(40, dtype=float)
    _, lo, hi = cluster_bootstrap(values, np.arange(40), n_boot=2000,
                                  rng=np.random.default_rng(3))
    _, lo_i, hi_i = cluster_bootstrap(values, None, n_boot=2000,
                                      rng=np.random.default_rng(3))
    assert (hi - lo) == pytest.approx(hi_i - lo_i, rel=0.2)
