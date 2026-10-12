"""The window definitions of `docs/protocol.md` §1.4 and the figure checklist.

`P`, `B` and `B_mid` are checked on a synthetic run whose bias and return are
known exactly, so a boundary error (e.g. losing the 150 k checkpoint) fails the
test. V7 and the measurement precision are checked per E1 condition: their
reference spread must be the condition's own between-run sd of `B`, never a
pooled one that mixes conditions whose `B` differ by orders of magnitude. The
figure self-check is checked both ways: it passes on a complete figure and
raises when a required element is missing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection, PathCollection, PolyCollection
from matplotlib.lines import Line2D

from lunar_bias.analysis import figures, summaries
from lunar_bias.analysis.load import RunData
from lunar_bias.config import RunConfig

STEPS = tuple(range(0, 300_000 + 1, 10_000))


@pytest.fixture(autouse=True)
def _report_style() -> None:
    """Render like the report: the figure geometry depends on the rcParams."""
    figures.use_style()


def fake_npz(bias: float, n_points: int = 3) -> dict[str, np.ndarray]:
    """A probe file whose Q1 is exactly `bias` above G at every point."""
    g = np.arange(1.0, n_points + 1.0, dtype=np.float64)
    q = np.zeros((1, n_points), dtype=np.float32)
    q[0] = g + bias
    return {"q": q, "G": g, "episode": np.zeros(n_points, dtype=np.int16),
            "t": np.arange(n_points, dtype=np.int16)}


def fake_run(state: str = "done") -> RunData:
    """A run whose Q1 bias at step `s` is `s / 10 000` and return likewise."""
    probes = {step: fake_npz(step / 10_000.0) for step in STEPS}
    rows = [{"step": step, "episode_index": i, "seed": 1_000_000 + i,
             "return_1000": step / 10_000.0, "length_1000": 100,
             "outcome": "landed"} for step in STEPS for i in range(2)]
    cfg = RunConfig()
    return RunData(run_id=cfg.run_id, cfg=cfg, state=state, step=cfg.max_steps,
                   wall_s=600.0, eval=pd.DataFrame(rows),
                   metrics=pd.DataFrame(), probes=probes)


def test_b_uses_the_second_half_only() -> None:
    # bias = step / 10 000, so steps 150 k … 300 k are 15 … 30 -> mean 22.5.
    assert summaries.B(fake_run()) == pytest.approx(22.5)


def test_b_mid_uses_the_third_to_half_window() -> None:
    # steps 100 k … 150 k are 10 … 15 -> mean 12.5 (the 150 k point is shared).
    assert summaries.B_mid(fake_run()) == pytest.approx(12.5)


def test_p_uses_the_last_five_evaluation_steps() -> None:
    # steps 260 k … 300 k are 26 … 30 -> mean 28.
    assert summaries.P(fake_run()) == pytest.approx(28.0)


def test_p_and_b_are_none_without_data() -> None:
    run = fake_run()
    run.eval = run.eval.iloc[0:0]
    run.probes = {}
    assert summaries.P(run) is None
    assert summaries.B(run) is None
    assert summaries.B_mid(run) is None


def test_checkpoint_bias_norm_bias_and_qmin() -> None:
    npz = {"q": np.asarray([[2.0, 4.0], [1.0, 3.0]], dtype=np.float32),
           "G": np.asarray([1.0, 2.0])}
    q1 = summaries.checkpoint_bias(npz, "q1")
    assert q1["bias"] == pytest.approx(1.5)
    assert q1["mean_q"] == pytest.approx(3.0)
    assert q1["norm_bias"] == pytest.approx(3.0 / 3.0)
    # qmin = elementwise min([2, 4], [1, 3]) = [1, 3]; diff = [0, 1].
    assert summaries.checkpoint_bias(npz, "qmin")["bias"] == pytest.approx(0.5)
    assert summaries.checkpoint_bias(npz, "qmin")["mean_q"] == pytest.approx(2.0)


def test_horizon_artefact_counts_only_running_episodes() -> None:
    npz = {"episode": np.asarray([0, 0, 1], dtype=np.int16),
           "t": np.asarray([0, 1, 0], dtype=np.int16),
           "ep_len": np.asarray([1500, 5], dtype=np.int32),
           "ep_G_at_train_horizon": np.asarray([10.0, 0.0]),
           "gamma": np.float64(0.99), "train_horizon": np.int32(1000)}
    out = summaries.horizon_artefact(npz)
    # Episode 0 runs past the horizon: at t = 0 and t = 1 the artefact is
    # gamma**1000 * 10 and gamma**999 * 10. Episode 1 ended early: zero.
    expected = float((0.99**1000 * 10.0 + 0.99**999 * 10.0) / 2)
    assert out["mean"] == pytest.approx(expected)
    assert out["max"] == pytest.approx(0.99**999 * 10.0)
    assert (out["n_points"], out["n_episodes"]) == (2, 1)


def _split_bias_npz(bias_all: float, bias_first10: float,
                    per_episode: float = 0.0) -> dict[str, np.ndarray]:
    """One checkpoint of 20 episodes: the first 10 carry `bias_first10`, all 20
    together carry `bias_all`. `per_episode` adds a deterministic spread whose
    mean is zero over the first ten and over all twenty episodes, so it changes
    the bootstrap width without moving either bias."""
    n_episodes, per = 20, 3
    episode = np.repeat(np.arange(n_episodes), per).astype(np.int16)
    spread = per_episode * (np.arange(n_episodes) % 5 - 2.0)
    # The last ten episodes must carry 2 * bias_all - bias_first10 so that the
    # mean over all twenty is exactly bias_all.
    bias = np.where(episode < 10, bias_first10,
                    2.0 * bias_all - bias_first10) + spread[episode]
    g = np.ones(n_episodes * per, dtype=np.float64)
    q = np.zeros((1, n_episodes * per), dtype=np.float32)
    q[0] = g + bias
    return {"q": q, "G": g, "episode": episode,
            "t": np.tile(np.arange(per), n_episodes).astype(np.int16)}


def controlled_run(run_id: str, *, bias_all: float, bias_first10: float | None = None,
                   per_episode: float = 0.0, ret: float | None = None) -> RunData:
    """A run with a known `B`, a known 10-episode `B` and, optionally, a known
    evaluation return (`ret`)."""
    bias_first10 = bias_all if bias_first10 is None else bias_first10
    probes = {step: _split_bias_npz(bias_all, bias_first10, per_episode)
              for step in STEPS}
    rows = [{"step": step, "episode_index": i, "seed": 1_000_000 + i,
             "return_1000": ret, "length_1000": 100, "outcome": "landed"}
            for step in STEPS for i in range(2)]
    cfg = RunConfig()
    return RunData(run_id=run_id, cfg=cfg, state="done", step=cfg.max_steps,
                   wall_s=600.0,
                   eval=pd.DataFrame(rows) if ret is not None else pd.DataFrame(),
                   metrics=pd.DataFrame(), probes=probes)


def test_v7_and_precision_use_the_within_condition_sd() -> None:
    # Two conditions whose own between-run spreads differ by four orders of
    # magnitude. The pooled sd over the eight runs is dominated by the wide
    # condition, so a pooled V7 would pass the narrow condition trivially.
    narrow = [(0.0, 0.05), (0.02, 0.02), (-0.02, 0.01), (0.01, 0.03)]
    wide = [(1000.0, 1.0), (1500.0, 1.0), (800.0, 1.0), (1200.0, 1.0)]
    conditions = {"narrow": [controlled_run(f"n{i}", bias_all=b, bias_first10=b + d)
                             for i, (b, d) in enumerate(narrow)],
                  "wide": [controlled_run(f"w{i}", bias_all=b, bias_first10=b + d)
                           for i, (b, d) in enumerate(wide)]}
    table = summaries.stability_by_condition(conditions)

    sd_narrow = float(np.std([b for b, _ in narrow], ddof=1))
    assert float(np.std([b for b, _ in narrow + wide], ddof=1)) > 20 * sd_narrow
    assert table["narrow"]["between_run_sd_B"] == pytest.approx(sd_narrow)
    assert table["wide"]["between_run_sd_B"] == pytest.approx(
        float(np.std([b for b, _ in wide], ddof=1)))
    diffs = [abs(d) for _, d in narrow]
    assert table["narrow"]["median_abs_diff"] == pytest.approx(float(np.median(diffs)))
    assert table["narrow"]["p90_abs_diff"] == pytest.approx(
        float(np.percentile(diffs, 90)))
    # |dB| = 0.01 is below the narrow sd (0.0171), the other three are not.
    assert table["narrow"]["fraction_runs_below_sd"] == pytest.approx(0.25)
    assert table["narrow"]["passes"] is False    # median 0.025 > sd 0.017
    assert table["wide"]["passes"] is True       # median 1.0 < sd ~299
    # Both conditions report their own checkpoint count and half-widths.
    assert table["narrow"]["n_checkpoints"] == 4 * len(STEPS)
    assert table["wide"]["median_half_width"] is not None


def test_precision_is_reported_per_condition() -> None:
    # Same biases, different return noise: the bootstrap half-width must follow
    # the noise of its own condition.
    quiet = [controlled_run(f"q{i}", bias_all=10.0 + 0.2 * i, per_episode=0.2)
             for i in range(2)]
    noisy = [controlled_run(f"s{i}", bias_all=10.0 + 0.2 * i, per_episode=50.0)
             for i in range(2)]
    table = summaries.stability_by_condition({"quiet": quiet, "noisy": noisy})
    assert table["quiet"]["n_checkpoints"] == 2 * len(STEPS)
    assert table["noisy"]["n_checkpoints"] == 2 * len(STEPS)
    assert table["noisy"]["median_half_width"] > table["quiet"]["median_half_width"]
    assert table["noisy"]["p90_half_width"] > table["quiet"]["p90_half_width"]


#: Per-checkpoint bias ranges of the real E1 runs (measured from `runs/`), for
#: the curves of panel (a).
E1_CURVE_RANGE = {
    "ddpg/none": (48.0, 420.0), "ddpg/layer": (37.0, 3500.0),
    "td3/none": (-43.0, 157.0), "td3/layer": (-26.0, 157.0),
}
#: Per-run `B` ranges of the same runs (`numbers.json`), for the dots of panel
#: (b): a run's `B` is far tighter than its individual checkpoints.
E1_B_RANGE = {
    "ddpg/none": (132.0, 243.0), "ddpg/layer": (431.0, 2671.0),
    "td3/none": (-22.0, -5.0), "td3/layer": (-9.0, 5.0),
}
#: Runs per condition of the synthetic figures: the report's n = 10.
FAKE_N = 10


def _values(condition: str, ranges: dict[str, tuple[float, float]]) -> list[float]:
    """Ten per-run values spanning the measured range of the condition."""
    lo, hi = ranges[condition]
    return [float(v) for v in np.linspace(lo, hi, FAKE_N)]


def _fake_e1_figures() -> tuple[tuple[Any, dict[str, str]], ...]:
    """The three report figures built from synthetic runs (same shapes as real)."""
    return_curves = {c: figures.return_curves(
        [controlled_run(f"{c}-{i}", bias_all=10.0 + i, ret=100.0 + i) for i in range(2)])
        for c in figures.E1_ORDER}
    bias_by_condition = {c: figures.bias_curves(
        [controlled_run(f"{c}-b{i}", bias_all=v)
         for i, v in enumerate(_values(c, E1_CURVE_RANGE))])
        for c in figures.E1_ORDER}
    b_values = {c: [(i, v) for i, v in enumerate(_values(c, E1_B_RANGE))]
                for c in figures.E1_ORDER}
    arm_curves = {a: figures.bias_curves(
        [controlled_run(f"{a}-{i}", bias_all=v)
         for i, v in enumerate(np.linspace(5.0, 500.0, FAKE_N))])
        for a in figures.E2_ORDER}
    p_values = {a: [(i, 100.0 + i) for i in range(FAKE_N)] for a in figures.E2_ORDER}
    return (figures.fig1_learning_curves(return_curves, None),
            figures.fig2_bias(bias_by_condition, b_values, None),
            figures.fig3_e2(arm_curves, p_values, None, None))


def test_titles_carry_no_figure_number() -> None:
    """LaTeX numbers the figures; a hard-coded 'Fig. N' would duplicate it."""
    for fig, caption in _fake_e1_figures():
        titles = [fig.get_suptitle(), *[ax.get_title() for ax in fig.axes]]
        assert any(titles)
        assert all(not title.startswith("Fig.") for title in titles if title)
        # The caption heading becomes LaTeX's \caption{}, which is numbered too.
        assert not caption["title"].startswith("Fig.")
    plt.close("all")


def _artist_points(artist: Any) -> np.ndarray | None:
    """The data coordinates an artist draws: lines, bands, dots or error bars."""
    if isinstance(artist, Line2D):
        return artist.get_xydata()
    if isinstance(artist, LineCollection):
        return np.concatenate(artist.get_segments()) if artist.get_segments() else None
    if isinstance(artist, PathCollection):  # scatter: the dot offsets
        return np.asarray(artist.get_offsets()) if len(artist.get_offsets()) else None
    if isinstance(artist, PolyCollection):  # fill_between: the band outline
        verts = [path.vertices for path in artist.get_paths()]
        return np.concatenate(verts) if verts else None
    return None


def _legends_covering_data(fig: Any) -> list[str]:
    """Titles of the axes whose legend is drawn on top of data.

    Every artist is sampled densely along its path, so a line crossing the
    legend counts even when its bounding box is much larger.
    """
    fig.canvas.draw()
    covered: list[str] = []
    for ax in fig.axes:
        box = ax.get_legend().get_window_extent()
        for artist in ax.get_children():
            points = _artist_points(artist)
            if points is None or len(points) < 2:
                continue
            dense = np.concatenate([np.linspace(a, b, 24)
                                    for a, b in zip(points[:-1], points[1:],
                                                    strict=True)])
            x, y = ax.transData.transform(dense).T
            if ((x > box.x0) & (x < box.x1) & (y > box.y0) & (y < box.y1)).any():
                covered.append(ax.get_title())
                break
    return covered


def test_fig2_is_symlog_with_legends_clear_of_the_data() -> None:
    fig2, caption2 = _fake_e1_figures()[1]
    # Fig. 2 must show the TD3 conditions next to DDPG+LN: symlog, linear
    # within +-20 return units.
    for ax in fig2.axes:
        assert ax.get_yscale() == "symlog"
        assert ax.yaxis.get_transform().linthresh == figures.Y_SYMLOG_LINTHRESH
    assert "symlog" in caption2["shows"]
    # The upper-left legend used to sit on the DDPG curves; the lower-left one
    # must not touch any line, band, dot or error bar.
    assert _legends_covering_data(fig2) == []
    plt.close("all")


def test_e2_colours_avoid_the_ddpg_orange() -> None:
    ddpg_colours = {figures.STYLE_E1["ddpg/none"][0], figures.STYLE_E1["ddpg/layer"][0]}
    assert figures.STYLE_E2["min"][0] == "blue"  # min is TD3
    for arm in ("mean", "max"):
        assert figures.STYLE_E2[arm][0] not in ddpg_colours
    assert len({figures.STYLE_E2[arm][0] for arm in figures.E2_ORDER}) == 3


def _good_caption() -> dict[str, str]:
    """A caption that satisfies every checklist item of `figures.self_check`."""
    return {"title": "Fig. X", "shows": "something, n = 10 runs",
            "variability": "95 % t-interval across runs",
            "tests": "Welch p = 0.01", "takeaway": figures.TAKEAWAY}


def test_self_check_passes_on_a_complete_figure() -> None:
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], label="a")
    ax.set_xlabel(figures.X_STEPS)
    ax.set_ylabel(figures.Y_BIAS)
    ax.legend()
    fig.suptitle("Fig. X — test")
    checks = figures.self_check(fig, _good_caption())
    assert any(check.startswith("title") for check in checks)
    plt.close(fig)


def test_self_check_raises_without_legend_or_labels() -> None:
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    fig.suptitle("Fig. X — test")
    with pytest.raises(RuntimeError, match="labels|legend"):
        figures.self_check(fig, _good_caption())
    plt.close(fig)


def test_self_check_raises_when_caption_lacks_tests_or_n() -> None:
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], label="a")
    ax.set_xlabel(figures.X_STEPS)
    ax.set_ylabel(figures.Y_BIAS)
    ax.legend()
    fig.suptitle("Fig. X — test")
    caption = _good_caption()
    caption["tests"] = "nothing here"
    with pytest.raises(RuntimeError, match="test results"):
        figures.self_check(fig, caption)
    caption = _good_caption()
    caption["shows"], caption["variability"] = "something", "some spread"
    with pytest.raises(RuntimeError, match="n not stated"):
        figures.self_check(fig, caption)
    caption = _good_caption()
    caption["takeaway"] = "bias is bad"
    with pytest.raises(RuntimeError, match="takeaway"):
        figures.self_check(fig, caption)
    plt.close(fig)


def test_save_writes_pdf_png_and_caption(tmp_path: Path) -> None:
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], label="a")
    ax.set_xlabel(figures.X_STEPS)
    ax.set_ylabel(figures.Y_BIAS)
    ax.legend()
    fig.suptitle("Fig. X — test")
    paths = figures.save(fig, _good_caption(), tmp_path / "figx")
    assert [p.name for p in paths] == ["figx.pdf", "figx.png", "figx.caption.md"]
    assert all(p.exists() for p in paths)
    text = (tmp_path / "figx.caption.md").read_text()
    assert "TODO(author)" in text and "95 % t-interval" in text
    plt.close(fig)
