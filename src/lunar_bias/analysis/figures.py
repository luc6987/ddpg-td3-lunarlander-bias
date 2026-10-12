"""Every report figure, as `(fig, caption)` (`docs/implementation_plan.md` §12).

The caption is a dict with the keys `title`, `shows`, `variability`, `tests`
and `takeaway`. It holds every factual element the graded figure checklist asks
for (n, the interval type, the test results) and nothing interpretive: the
takeaway is always the literal `TODO(author)`, which the author replaces.

Style: Okabe–Ito colours, solid for `norm = none` and dashed for `norm = layer`,
a 95 % t-interval band across runs, thin per-run lines at alpha 0.15, no
smoothing, font >= 7 pt, full text width and <= 2.4 in high.

`self_check(fig, caption)` verifies the checklist and raises; `scripts/analyze.py`
calls it for every figure it writes.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from scipy import stats

from lunar_bias.analysis import summaries
from lunar_bias.analysis.load import RunData
from lunar_bias.analysis.stats import t_interval

#: Okabe–Ito palette (colour-blind safe). Colour encodes the algorithm, the
#: line style the critic norm, so a grayscale print still separates the two.
OKABE_ITO = {
    "black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
    "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7",
}
#: Colour and style per E1 condition; a condition is (algo, critic_norm).
STYLE_E1 = {
    "ddpg/none": ("orange", "-"), "ddpg/layer": ("orange", "--"),
    "td3/none": ("blue", "-"), "td3/layer": ("blue", "--"),
}
#: Colour and style per E2 arm (`target_reduction`). `min` keeps TD3's blue;
#: `mean` and `max` deliberately avoid the DDPG orange of Figs 1–2, so that
#: E2's arms cannot be confused with E1's algorithms.
STYLE_E2 = {"min": ("blue", "-"), "mean": ("green", "-"), "max": ("purple", "-")}
#: Order in which conditions appear everywhere.
E1_ORDER = ("ddpg/none", "ddpg/layer", "td3/none", "td3/layer")
E2_ORDER = ("min", "mean", "max")

#: Axis labels with their units (the checklist requires labels and scales). They
#: are kept short, and split over two lines where needed, so that a rotated
#: label never grows taller than its ~1.5 in axes; the caption spells them out.
X_STEPS = "Environment steps (×10³)"
Y_RETURN = "Return, first 1 000 steps\n(return units)"
Y_BIAS = "Bias  Q₁ − G\n(return units)"
Y_BIAS_SYMLOG = "Bias  Q₁ − G\n(return units, symlog)"
Y_B = "B, mean Q₁ bias ≥ 150 k\n(return units, symlog)"
Y_P = "P, mean return last 5 ckpts\n(return units)"

#: Fig. 2 uses a symmetric-log y axis: linear within ±20 return units, log
#: outside, so that TD3's near-zero bias and DDPG+LN's large one share an axis.
Y_SYMLOG_LINTHRESH = 20.0
#: Symlog headroom, so that no legend sits on a curve: the top clears DDPG+LN's
#: checkpoints (max ≈ 3.5 · 10³), the bottom gives panel (a) a free band under
#: the TD3 curves (min ≈ −43) for its lower-left legend.
Y_SYMLOG_TOP = 1e4
Y_SYMLOG_BOTTOM = -1e4

TAKEAWAY = "TODO(author)"


def use_style() -> None:
    """Set the report-wide matplotlib style (called once by `scripts/analyze.py`)."""
    plt.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42,   # embed TrueType, not Type 3
        "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
        "legend.fontsize": 6.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 150, "savefig.bbox": "tight",
    })


def _figure(height: float = 2.2) -> tuple[Figure, np.ndarray]:
    # Full text width; constrained layout so the figure title never collides
    # with the panel titles.
    return plt.subplots(1, 2, figsize=(6.5, height), layout="constrained")


def _family_label(condition: str, n: int) -> str:
    if "/" not in condition:
        return f"{condition} (n = {n})"
    algo, norm = condition.split("/")
    name = algo.upper()
    norm_text = "no LN" if norm == "none" else "LN"
    return f"{name}, {norm_text} (n = {n})"


def _short_label(condition: str, n: int) -> str:
    """Two-line tick label, so four conditions fit under a half-width panel.

    n is carried by the legend and the caption, not by the tick label, which
    would otherwise collide with its neighbour.
    """
    if "/" not in condition:
        return f"{condition}\n(n = {n})"
    algo, norm = condition.split("/")
    norm_text = "no LN" if norm == "none" else "LN"
    return f"{algo.upper()}\n{norm_text}"


def _n_text(sizes: dict[str, int]) -> str:
    """The n (and df) of a figure's conditions, for the caption."""
    sizes = {name: n for name, n in sizes.items() if n}
    if not sizes:
        return "n = 0"
    if len(set(sizes.values())) == 1:
        n = next(iter(sizes.values()))
        return f"n = {n} runs per condition, df = {max(n - 1, 0)}"
    return "n = " + ", ".join(f"{name}: {n}" for name, n in sizes.items())


def _summary(curves: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean, 95 % t-interval half-width and n at every step of a run×step frame."""
    x = curves.index.to_numpy(dtype=float) / 1000.0
    n = curves.notna().sum(axis=1).to_numpy()
    mean = curves.mean(axis=1).to_numpy()
    sd = curves.std(axis=1, ddof=1).to_numpy()
    half = np.where(n >= 2, stats.t.ppf(0.975, np.maximum(n - 1, 1)) * sd
                    / np.sqrt(np.maximum(n, 1)), np.nan)
    return x, mean, half


def _style(condition: str) -> tuple[str, str]:
    """Colour and line style of a condition, E1 (`algo/norm`) or E2 (`min`…)."""
    if condition in STYLE_E1:
        return STYLE_E1[condition]
    return STYLE_E2[condition]


def _plot_curves(ax: Any, curves: pd.DataFrame, condition: str, *, thin: bool = True,
                 label: str | None = None) -> None:
    """One condition: mean line, 95 % t-interval band, optional thin per-run lines."""
    colour, linestyle = _style(condition)
    x, mean, half = _summary(curves)
    if thin:
        for column in curves.columns:
            ax.plot(x, curves[column].to_numpy(), color=OKABE_ITO[colour],
                    lw=0.5, alpha=0.15, ls=linestyle, zorder=1)
    ax.fill_between(x, mean - half, mean + half, color=OKABE_ITO[colour],
                    alpha=0.18, lw=0, zorder=2)
    ax.plot(x, mean, color=OKABE_ITO[colour], ls=linestyle, lw=1.4, zorder=3,
            label=label or _family_label(condition, curves.shape[1]))


def return_curves(runs: list[RunData]) -> pd.DataFrame:
    """run_id × step matrix of the mean `return_1000` over the fixed episodes."""
    return pd.DataFrame({run.run_id: run.eval.groupby("step")["return_1000"].mean()
                         for run in runs})


def bias_curves(runs: list[RunData], critic: str = "q1") -> pd.DataFrame:
    """run_id × step matrix of the per-checkpoint Q1 bias."""
    return pd.DataFrame({
        run.run_id: summaries.checkpoint_bias_table(run, critic)  # type: ignore[arg-type]
        .set_index("step")["bias"] for run in runs})


def _family_tests_text(results: dict[str, dict[str, Any]] | None) -> str:
    """Family A/B results as one sentence, or a plain statement when not run."""
    if not results:
        return "No test: the analysis needs at least two runs per arm (not computed)."
    parts = []
    for name, r in results.items():
        if "delta" not in r:
            parts.append(f"{name}: {r.get('error', 'not computed')}")
            continue
        parts.append(f"{name}: Δ = {r['delta']:.2f} [{r['ci_lo']:.2f}, {r['ci_hi']:.2f}], "
                     f"Welch t = {r['t']:.2f}, df = {r['df']:.1f}, p = {_p(r['p'])}, "
                     f"Holm p = {_p(r['p_holm'])}, Hedges g = {r['hedges_g']:.2f}, "
                     f"Mann–Whitney p = {_p(r['mw_p'])} (n = {r['n_x']} vs {r['n_y']})")
    return "Welch two-sided tests, Holm within the family; Δ = mean(y) − mean(x). " \
           + " | ".join(parts)


# --- Fig. 1 ---------------------------------------------------------------------

def fig1_learning_curves(curves_by_condition: dict[str, pd.DataFrame],
                         family_b: dict[str, dict[str, float]] | None = None
                         ) -> tuple[Figure, dict[str, str]]:
    """E1 learning curves: return vs environment steps, four conditions."""
    fig, ax = plt.subplots(figsize=(6.5, 2.4))
    n_note, sizes = [], {}
    for condition in E1_ORDER:
        curves = curves_by_condition.get(condition)
        if curves is None or curves.empty:
            continue
        _plot_curves(ax, curves, condition)
        n_note.append(_family_label(condition, curves.shape[1]))
        sizes[condition] = int(curves.shape[1])
    ax.set_xlabel(X_STEPS)
    ax.set_ylabel(Y_RETURN)
    ax.set_title("E1: learning curves on LunarLander (continuous actions)")
    ax.set_xlim(left=0)
    ax.legend(loc="lower right", frameon=False, ncols=2)
    caption = {
        "title": "E1 learning curves",
        "shows": ("Mean return over the first 1 000 steps of the 20 fixed evaluation "
                  "episodes against environment steps, for the four E1 conditions "
                  "(DDPG and TD3, critic norm 'none' or 'layer'; F1 nets 256×256). "
                  + "; ".join(n_note) + "."),
        "variability": ("Line: mean across runs; band: 95 % t-interval across runs ("
                        + _n_text(sizes) + "); thin lines: individual runs at "
                        "alpha 0.15. The 20 evaluation episodes are identical for "
                        "every run and checkpoint. Curves are not smoothed."),
        "tests": "Family B (performance), " + _family_tests_text(family_b),
        "takeaway": TAKEAWAY,
    }
    return fig, caption


# --- Fig. 2 ---------------------------------------------------------------------

def _per_run_axis(ax: Any, values: dict[str, list[tuple[int, float]]],
                  order: tuple[str, ...], ylabel: str, title: str,
                  xlabel: str) -> None:
    """Dots per run and a mean with its 95 % t-interval, one column per condition."""
    for position, condition in enumerate(order):
        points = values.get(condition, [])
        if not points:
            continue
        seeds = np.asarray([p[0] for p in points], dtype=float)
        vals = np.asarray([p[1] for p in points], dtype=float)
        colour, _ = _style(condition)
        # Seeded jitter: the runs of one condition would otherwise overplot exactly.
        jitter = 0.05 * (seeds - seeds.mean()) / max(np.ptp(seeds), 1.0) * len(seeds) * 2
        ax.scatter(position + jitter, vals, s=12, color=OKABE_ITO[colour], marker="o",
                   alpha=0.85, zorder=3)
        if len(vals) >= 2:
            mean, lo, hi = t_interval(vals)
            ax.errorbar(position, mean, yerr=[[mean - lo], [hi - mean]], color="black",
                        capsize=3, lw=1.2, marker="D", ms=3, zorder=4)
    ax.axhline(0.0, color="black", lw=0.6, alpha=0.5, zorder=0)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([_short_label(c, len(values.get(c, []))) for c in order],
                       fontsize=6.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    # Panel (a) carries the four condition labels. Repeating them here would
    # reach either the TD3 points or the DDPG+LN dots, so this legend explains
    # only the two mark types, in the free band between those two clusters.
    ax.legend(handles=[
        plt.Line2D([], [], color="black", marker="o", ls="", ms=4),
        plt.Line2D([], [], color="black", marker="D", ls="", ms=3),
    ], labels=["one run (seeds 0–9)", "mean ± 95 % CI"],
        loc="center left", frameon=False, fontsize=6.5, ncols=2, borderpad=0.25)


def fig2_bias(curves_by_condition: dict[str, pd.DataFrame],
              b_values: dict[str, list[tuple[int, float]]],
              family_a: dict[str, dict[str, float]] | None = None
              ) -> tuple[Figure, dict[str, str]]:
    """E1 bias: (a) curves with a zero line; (b) per-run B with mean ± 95 % CI."""
    fig, axes = _figure()
    ax = axes[0]
    n_note = []
    for condition in E1_ORDER:
        curves = curves_by_condition.get(condition)
        if curves is None or curves.empty:
            continue
        _plot_curves(ax, curves, condition)
        n_note.append(_family_label(condition, curves.shape[1]))
    ax.axhline(0.0, color="black", lw=0.7, ls=":", zorder=0)
    # Symlog on both panels: TD3's bias (a few return units) and DDPG+LN's
    # (hundreds to thousands) would otherwise share an axis unreadably.
    ax.set_yscale("symlog", linthresh=Y_SYMLOG_LINTHRESH)
    ax.set_ylim(bottom=Y_SYMLOG_BOTTOM, top=Y_SYMLOG_TOP)
    ax.set_xlabel(X_STEPS)
    ax.set_ylabel(Y_BIAS_SYMLOG)
    ax.set_title("(a) bias of Q₁, per checkpoint")
    # Inside, in the empty band under the TD3 curves: at the top the legend
    # would sit on the DDPG curves, and below the axes it would break the
    # alignment of the two panels.
    ax.legend(loc="lower left", frameon=False, ncols=2, fontsize=6,
              labelspacing=0.25, borderpad=0.25)

    _per_run_axis(axes[1], b_values, E1_ORDER, Y_B,
                  "(b) per-run B, mean ± 95 % t-interval",
                  "Condition (algorithm × critic norm)")
    axes[1].set_yscale("symlog", linthresh=Y_SYMLOG_LINTHRESH)
    axes[1].set_ylim(top=Y_SYMLOG_TOP)
    fig.suptitle("E1: overestimation bias Q₁ − G over training")
    caption = {
        "title": "E1 bias: curves and per-run B",
        "shows": ("(a) Mean Q₁ bias against the Monte-Carlo return-to-go G, per "
                  "checkpoint, for the four E1 conditions; the dotted line is zero "
                  "(no bias). (b) B (mean Q₁ bias over the 16 checkpoints 150 k–300 k) "
                  "per run as a dot, with the condition mean and its 95 % t-interval. "
                  "Both panels use a symmetric-log y axis (symlog: linear within "
                  "±20 return units, logarithmic outside), so that TD3's small bias "
                  "stays readable next to DDPG+LN's large one. "
                  + "; ".join(n_note) + "."),
        "variability": ("Band in (a) and error bars in (b): 95 % t-interval across runs ("
                        + _n_text({c: len(b_values.get(c, [])) for c in E1_ORDER})
                        + "). Thin lines in (a): individual runs at alpha 0.15; dots in "
                          "(b): one dot per run (seeds 0–9). G is one unbiased "
                          "Monte-Carlo sample per point, not an exact value."),
        "tests": "Family A (bias), " + _family_tests_text(family_a),
        "takeaway": TAKEAWAY,
    }
    return fig, caption


# --- Fig. 3 ---------------------------------------------------------------------

def fig3_e2(curves_by_arm: dict[str, pd.DataFrame],
            p_values: dict[str, list[tuple[int, float]]],
            page_c0: dict[str, Any] | None = None,
            page_c1: dict[str, Any] | None = None) -> tuple[Figure, dict[str, str]]:
    """E2: (a) bias curves for min/mean/max; (b) per-seed P with seed lines."""
    fig, axes = _figure()
    ax = axes[0]
    sizes = {}
    for arm in E2_ORDER:
        curves = curves_by_arm.get(arm)
        if curves is None or curves.empty:
            continue
        _plot_curves(ax, curves, arm, label=f"{arm} (n = {curves.shape[1]})")
        sizes[arm] = int(curves.shape[1])
    ax.axhline(0.0, color="black", lw=0.7, ls=":", zorder=0)
    ax.set_xlabel(X_STEPS)
    ax.set_ylabel(Y_BIAS)
    ax.set_title("(a) bias of Q₁ by target reduction")
    ax.legend(loc="upper left", frameon=False)

    ax = axes[1]
    per_seed: dict[int, list[float | None]] = {}
    means: list[tuple[int, float, float, float]] = []
    for position, arm in enumerate(E2_ORDER):
        points = p_values.get(arm, [])
        for seed, value in points:
            per_seed.setdefault(seed, [None] * len(E2_ORDER))[position] = value
        present = np.asarray([v for _, v in points], dtype=float)
        if len(present) >= 2:
            means.append((position, *t_interval(present)))
        ax.scatter([position] * len(present), present, s=12,
                   color=OKABE_ITO[STYLE_E2[arm][0]], alpha=0.85, zorder=3)
    for vals in per_seed.values():
        xs = [i for i, v in enumerate(vals) if v is not None]
        ys = [v for v in vals if v is not None]
        ax.plot(xs, ys, color="black", lw=0.5, alpha=0.25, zorder=1)
    for position, mean, lo, hi in means:
        ax.errorbar(position, mean, yerr=[[mean - lo], [hi - mean]], color="black",
                    capsize=3, lw=1.2, marker="D", ms=3, zorder=4)
    ax.set_xticks(range(len(E2_ORDER)))
    ax.set_xticklabels([f"{arm} (n = {len(p_values.get(arm, []))})" for arm in E2_ORDER])
    ax.set_xlabel("Target reduction (min / mean / max)")
    ax.set_ylabel(Y_P)
    ax.set_title("(b) per-seed P by target reduction")
    ax.legend(handles=[plt.Line2D([], [], color=OKABE_ITO[STYLE_E2[a][0]], marker="o",
                                  ls="", label=f"{a} (n = {len(p_values.get(a, []))})")
                       for a in E2_ORDER]
                       + [plt.Line2D([], [], color="black", marker="D", ls="",
                                     label="mean ± 95 % CI")],
              loc="best", frameon=False)
    fig.suptitle("E2: the target reduction moves bias and performance")
    caption = {
        "title": "E2: bias and performance by target reduction",
        "shows": ("(a) Mean Q₁ bias against G over training for TD3 without LN with "
                  "target reduction min, mean or max. (b) P (mean return over the last "
                  "five checkpoints, i.e. 100 episodes) per run; thin lines join the "
                  "same training seed across the three arms. "
                  + _n_text(sizes) + "."),
        "variability": ("Band in (a) and error bars in (b): 95 % t-interval across runs "
                        "(df = n − 1). Dots in (b): one run each, "
                        "seeds shown; the min arm is the same run as E1's TD3/none "
                        "condition, not a new one."),
        "tests": ("C0 (manipulation check, B, predicted min < mean < max): "
                  + _page_text(page_c0) + ". C1 (outcome, P, predicted min > mean > max): "
                  + _page_text(page_c1) + ". Page's trend test with seeds as blocks; "
                  "diverged runs, if any, are ranked worst."),
        "takeaway": TAKEAWAY,
    }
    return fig, caption


def _p(value: float) -> str:
    """A p-value: four decimals, or scientific notation when very small."""
    return f"{value:.4f}" if value >= 1e-4 else f"{value:.2e}"


def _page_text(result: dict[str, Any] | None) -> str:
    if not result:
        return "not run (no data)"
    return (f"L = {result['L']:.1f}, p = {_p(result['p'])} ({result['method']}), "
            f"n = {result['n_blocks']} blocks")


# --- the graded-checklist self-check ---------------------------------------------

_UNIT = re.compile(r"\(")


def self_check(fig: Figure, caption: dict[str, str]) -> list[str]:
    """Verify one figure against the report checklist; raise on the first failure.

    Returns the list of successful checks so that `scripts/analyze.py` can print
    them. The checks are deliberately strict: a figure that fails here is not
    allowed into `report/`.
    """
    problems: list[str] = []
    ok: list[str] = []

    panels = [ax for ax in fig.axes if ax.get_label() != "<colorbar>"]
    # A single-panel figure carries its title on the axes; a multi-panel one on
    # the figure. Either way the title must be there and the caption must have one.
    title = caption.get("title", "")
    shown = fig.get_suptitle() or next((ax.get_title() for ax in panels
                                        if ax.get_title()), "")
    if title and shown:
        ok.append(f"title: {shown!r}")
    else:
        problems.append("title missing (caption['title'] and/or a visible title)")
    for i, ax in enumerate(panels):
        xlabel, ylabel = ax.get_xlabel(), ax.get_ylabel()
        if xlabel and ylabel and _UNIT.search(xlabel) and _UNIT.search(ylabel):
            ok.append(f"axes {i}: labels with units ({xlabel!r} / {ylabel!r})")
        else:
            problems.append(f"axes {i}: x and y labels must be non-empty and carry "
                            f"units in parentheses (got {xlabel!r} / {ylabel!r})")
        if ax.get_legend() is not None:
            ok.append(f"axes {i}: legend present")
        else:
            problems.append(f"axes {i}: no legend")
        if not ax.get_xticks().size or not ax.get_yticks().size:
            problems.append(f"axes {i}: empty tick scale on an axis")

    text = " ".join([caption.get("shows", ""), caption.get("variability", "")])
    if re.search(r"\bn\s*=\s*\d+", text):
        ok.append("caption: n stated")
    else:
        problems.append("caption: n not stated (expected 'n = <int>')")
    if "95 %" in text and "interval" in text:
        ok.append("caption: interval type stated")
    else:
        problems.append("caption: interval type not stated (expected '95 % ... interval')")

    tests = caption.get("tests", "")
    # A p-value is required; "not computed"/"not run" is accepted only as the
    # explicit statement that the test could not be run (fewer than 2 runs/arm).
    if tests and (re.search(r"\bp\s*=", tests)
                  or re.search(r"not (computed|run)", tests)):
        ok.append("caption: test results stated")
    else:
        problems.append("caption: no statistical test results (expected 'p = ...')")

    if caption.get("takeaway") == TAKEAWAY:
        ok.append(f"caption: takeaway is the literal {TAKEAWAY!r}")
    else:
        problems.append(f"caption: takeaway must be the literal {TAKEAWAY!r}")

    if problems:
        raise RuntimeError("figure self-check failed:\n  - " + "\n  - ".join(problems))
    return ok


def save(fig: Figure, caption: dict[str, str], stem: Any) -> list[Path]:
    """Write `<stem>.pdf`, `<stem>.png` and `<stem>.caption.md`; return the paths."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    pdf, png = stem.with_suffix(".pdf"), stem.with_suffix(".png")
    fig.savefig(pdf)
    fig.savefig(png, dpi=150)
    caption_path = stem.with_name(stem.name + ".caption.md")
    lines = [f"# {caption['title']}", "",
             f"**Shows.** {caption['shows']}", "",
             f"**Variability.** {caption['variability']}", "",
             f"**Tests.** {caption['tests']}", "",
             f"**TAKEAWAY: {caption['takeaway']}**", ""]
    caption_path.write_text("\n".join(lines))
    return [pdf, png, caption_path]
