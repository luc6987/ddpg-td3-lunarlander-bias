#!/usr/bin/env python
"""CLI: `runs/` -> `report/` — figures, tables and every number the text cites.

    uv run python scripts/analyze.py --runs runs --out report

The analysis never trains and never writes outside `--out`. It works on partial
data: conditions with fewer runs than planned are rendered with their n shown,
never imputed, and empty blocks are skipped with a printed notice.

Everything interpretive is left to the author: captions carry the factual
elements (what is shown, n, the interval type, the test results) and the literal
`TODO(author)` takeaway.

`docs/protocol.md` §6 fixes what is computed here: $P$ and $B$ per run, Welch +
Holm in Families A (bias) and B (performance), Page's trend tests for E2,
descriptives, and the validation numbers V4–V7.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from lunar_bias import grids
from lunar_bias.analysis import figures, load, summaries, tables
from lunar_bias.analysis.load import RunData
from lunar_bias.analysis.stats import describe, holm_family, page, welch

#: E1 contrasts, as (name, x-condition, y-condition): Δ = mean(y) − mean(x).
FAMILY_A_B = (
    ("H1: TD3 − DDPG (no LN)", "ddpg/none", "td3/none"),
    ("H2-DDPG: LN − none", "ddpg/none", "ddpg/layer"),
    ("H2-TD3: LN − none", "td3/none", "td3/layer"),
)
#: E2 Page trend tests (`docs/protocol.md` §6): (name, metric, predicted ranks).
FAMILY_C = (
    ("C0", "B", (1, 2, 3), "manipulation check: min < mean < max"),
    ("C1", "P", (3, 2, 1), "outcome: min > mean > max"),
)
#: A condition with fewer runs than this cannot enter a Welch test or an interval.
MIN_RUNS = 2


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", default="runs", help="directory of run directories")
    p.add_argument("--out", default="report", help="output directory")
    return p.parse_args(argv)


def notice(message: str) -> None:
    print(f"notice: {message}", flush=True)


# --- selecting runs ---------------------------------------------------------------

def grid_ids() -> dict[str, set[str]]:
    return {name: {cfg.run_id for cfg in factory()}
            for name, factory in grids.GRIDS.items()}


def e1_condition(run: RunData) -> str:
    return f"{run.cfg.algo.name}/{run.cfg.critic.norm}"


def e2_arm(run: RunData) -> str:
    return run.cfg.algo.target_reduction


def group_runs(runs: dict[str, RunData], ids: set[str], key_fn: Any,
               order: tuple[str, ...]) -> dict[str, list[RunData]]:
    """Group the runs of one grid by condition, in a fixed order, seeds sorted."""
    out: dict[str, list[RunData]] = {name: [] for name in order}
    for run_id in sorted(ids):
        run = runs.get(run_id)
        if run is None:
            continue
        out[key_fn(run)].append(run)
    for name in out:
        out[name].sort(key=lambda r: r.cfg.seed)
    return out


def only_done(groups: dict[str, list[RunData]]) -> dict[str, list[RunData]]:
    """The `done` runs of each condition: diverged/crashed/running never enter means."""
    return {name: [run for run in rs if run.state == "done"] for name, rs in groups.items()}


def report_coverage(label: str, expected: set[str],
                    runs: dict[str, RunData]) -> dict[str, Any]:
    """Count done / diverged / crashed / running / missing for one grid."""
    counts: dict[str, Any] = {"expected": len(expected)}
    for state in ("done", "diverged", "crashed", "running", "missing"):
        counts[state] = 0
    for run_id in expected:
        counts[runs[run_id].state if run_id in runs else "missing"] += 1
    counts["absent_dirs"] = sorted(expected - set(runs))
    notice(f"{label}: {counts['done']} done, {counts['diverged']} diverged, "
           f"{counts['crashed']} crashed, {counts['running']} running, "
           f"{counts['missing']} missing (of {counts['expected']} planned)")
    return counts


def condition_sizes(label: str, conditions: dict[str, list[RunData]],
                    planned: int) -> None:
    for name, runs in conditions.items():
        if len(runs) < planned:
            states = {}
            for run in runs:
                states[run.state] = states.get(run.state, 0) + 1
            notice(f"{label}: condition {name} has {len(runs)}/{planned} runs "
                   f"(never imputed){' ' + str(states) if states else ''}")


# --- statistics -------------------------------------------------------------------

def e2_stats(runs: list[RunData]) -> dict[str, Any]:
    """Per-arm descriptives of one E2 reduction (P and B over its runs)."""
    p = [v for _, v in per_seed(runs, summaries.P)]
    b = [v for _, v in per_seed(runs, summaries.B)]
    return {"n_runs": len(p),
            "P": describe(p) if p else None,
            "B": describe(b) if b else None}


def per_seed(values: list[RunData], fn: Any) -> list[tuple[int, float]]:
    """`[(seed, value)]` for the runs whose value is defined, sorted by seed."""
    out = []
    for run in values:
        value = fn(run)
        if value is not None:
            out.append((run.cfg.seed, float(value)))
    return sorted(out)


def contrasts(condition_values: dict[str, list[tuple[int, float]]],
              pairs: tuple[tuple[str, str, str], ...]) -> dict[str, dict[str, Any]]:
    """Welch + Holm over one family; an arm with n < 2 is reported, not imputed."""
    results: dict[str, dict[str, Any]] = {}
    for name, x_cond, y_cond in pairs:
        x = [v for _, v in condition_values.get(x_cond, [])]
        y = [v for _, v in condition_values.get(y_cond, [])]
        if len(x) < MIN_RUNS or len(y) < MIN_RUNS:
            results[name] = {"error": f"not computed: n = {len(x)} vs {len(y)} (needs "
                                      f"≥ {MIN_RUNS} runs per arm)"}
            notice(f"{name}: {results[name]['error']}")
            continue
        results[name] = welch(x, y)
    return holm_family(results) if any("p" in r for r in results.values()) else results


def page_blocks(arm_values: dict[str, dict[int, float | None]],
                diverged: dict[str, dict[int, bool]],
                sign: float) -> tuple[np.ndarray, int]:
    """Blocks = seeds for E2; a diverged run is encoded as `sign * inf`.

    Returns the `[n_blocks, 3]` matrix for (min, mean, max) and the number of
    seeds dropped because an arm has no value at all.
    """
    arms = ("min", "mean", "max")
    seeds = sorted(set.intersection(*(set(arm_values.get(a, {})) for a in arms)))
    rows, dropped = [], 0
    for seed in seeds:
        row = []
        missing = False
        for arm in arms:
            value = arm_values.get(arm, {}).get(seed)
            if value is None:
                if diverged.get(arm, {}).get(seed):
                    value = sign * np.inf
                else:
                    missing = True
                    break
            row.append(value)
        if missing:
            dropped += 1
        else:
            rows.append(row)
    return np.asarray(rows, dtype=float), dropped


def measurement_precision(v7: dict[str, Any] | None) -> dict[str, Any]:
    """Secondary: per-condition bootstrap half-width vs the within-condition sd(B)."""
    by_condition = {
        name: {"n_checkpoints": row["n_checkpoints"],
               "median_half_width": row["median_half_width"],
               "p90_half_width": row["p90_half_width"],
               "between_run_sd_B": row["between_run_sd_B"]}
        for name, row in (v7 or {}).get("by_condition", {}).items()}
    return {
        "definition": ("per checkpoint, a 95 % episode-clustered bootstrap interval "
                       "(2 000 resamples of the 20 probe episodes); the median and p90 "
                       "are over the checkpoints of the condition's runs. "
                       "between_run_sd_B is the WITHIN-condition between-run sd of B "
                       "(the same value V7 uses)"),
        "by_condition": by_condition,
    }


def validation_v4(runs: list[RunData]) -> dict[str, Any] | None:
    """V4: the step-0 probe of the pilot runs, with one explicit point."""
    for run in runs:
        npz = run.probes.get(0)
        if npz is None:
            continue
        q1 = np.asarray(npz["q"][0], dtype=np.float64)
        g = np.asarray(npz["G"], dtype=np.float64)
        return {"run_id": run.run_id, "condition": run.condition,
                "mean_q1": float(q1.mean()), "mean_G": float(g.mean()),
                "bias": float((q1 - g).mean()),
                "point": {"episode": int(npz["episode"][0]), "t": int(npz["t"][0]),
                          "Q1": float(q1[0]), "G": float(g[0]),
                          "Q1_minus_G": float(q1[0] - g[0])},
                "n_points": int(len(g))}
    return None


def validation_v6(runs: list[RunData]) -> dict[str, Any] | None:
    """V6: what the 1 000-step horizon would have truncated, in return units."""
    per_run, worst = [], 0.0
    for run in runs:
        means = []
        for step in sorted(run.probes):
            artefact = summaries.horizon_artefact(run.probes[step])
            means.append(artefact["mean"])
            worst = max(worst, artefact["max"])
        if means:
            per_run.append(float(np.mean(means)))
    if not per_run:
        return None
    return {"runs": len(per_run), "mean_over_runs": float(np.mean(per_run)),
            "median_over_runs": float(np.median(per_run)),
            "max_over_runs": float(np.max(per_run)), "max_point": worst}


def validation_v7(conditions: dict[str, list[RunData]]) -> dict[str, Any] | None:
    """V7 per E1 condition: the 10-episode B vs the within-condition sd of B."""
    table = summaries.stability_by_condition(conditions)
    if not table:
        return None
    row_passes = [row["passes"] for row in table.values()]
    return {"by_condition": table,
            "passes_all": bool(row_passes) and all(p is True for p in row_passes)}


def across_condition_correlation(values: dict[str, list[tuple[int, float]]],
                                 metric: str) -> dict[str, Any]:
    """Per-seed scores are correlated across conditions only as a diagnostic."""
    from scipy import stats as sstats

    names = [name for name, v in values.items() if len(v) >= 2]
    out: dict[str, Any] = {}
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            a = dict(values[first])
            b = dict(values[second])
            seeds = sorted(set(a) & set(b))
            if len(seeds) < 3:
                continue
            x = [a[s] for s in seeds]
            y = [b[s] for s in seeds]
            out[f"{first} vs {second}"] = {
                "metric": metric, "n": len(seeds),
                "pearson": float(sstats.pearsonr(x, y).statistic),
                "spearman": float(sstats.spearmanr(x, y).statistic),
            }
    return out


def x1_within_condition(conditions: dict[str, list[RunData]]) -> dict[str, Any]:
    """X1 (exploratory): Spearman between B_mid and P within each condition."""
    from scipy import stats as sstats

    out = {}
    for name, runs in conditions.items():
        pairs = [(summaries.B_mid(run), summaries.P(run)) for run in runs]
        pairs = [(b, p) for b, p in pairs if b is not None and p is not None]
        if len(pairs) < 3:
            continue
        b, p = zip(*pairs, strict=True)
        out[name] = {"n": len(pairs), "spearman": float(sstats.spearmanr(b, p).statistic),
                     "p": float(sstats.spearmanr(b, p).pvalue)}
    return out


# --- the run ----------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    t0 = time.time()
    figures.use_style()
    runs_dir, out_dir = Path(args.runs), Path(args.out)
    fig_dir, table_dir = out_dir / "figures", out_dir / "tables"

    index = load.load_index(runs_dir)
    ids = grid_ids()
    wanted = ids["e1"] | ids["e2"] | ids["pilot"] | ids["e3"] | ids["e4"]
    runs = load.load_runs(runs_dir, run_ids=sorted(wanted))
    print(f"# analysis: {len(index)} run directories, {len(runs)} in a grid, "
          f"{time.time() - t0:.1f}s to load", flush=True)

    extra = set(index["run_id"]) - wanted
    if extra:
        notice(f"{len(extra)} run director{'y' if len(extra) == 1 else 'ies'} not in any "
               f"grid (e.g. the pre-F1 pilot): listed in run_table.csv only")

    coverage = {
        "e1": report_coverage("E1", ids["e1"], runs),
        "e2": report_coverage("E2", ids["e2"], runs),
    }
    e1_all = group_runs(runs, ids["e1"], e1_condition, figures.E1_ORDER)
    e2_all = group_runs(runs, ids["e2"], e2_arm, figures.E2_ORDER)
    e1, e2 = only_done(e1_all), only_done(e2_all)
    condition_sizes("E1", e1, len(grids.S10))
    condition_sizes("E2", e2, len(grids.S10))

    # --- per-run summaries ---------------------------------------------------------
    e1_p = {name: per_seed(rs, summaries.P) for name, rs in e1.items()}
    e1_b = {name: per_seed(rs, summaries.B) for name, rs in e1.items()}
    e1_b_mid = {name: per_seed(rs, summaries.B_mid) for name, rs in e1.items()}
    e1_curves = {name: figures.bias_curves(rs) for name, rs in e1.items() if rs}
    return_curves = {name: figures.return_curves(rs) for name, rs in e1.items() if rs}
    e2_curves = {name: figures.bias_curves(rs) for name, rs in e2.items() if rs}

    descriptives: dict[str, dict[str, Any]] = {}
    for name in e1:
        values = {"P": [v for _, v in e1_p[name]], "B": [v for _, v in e1_b[name]],
                  "B_mid": [v for _, v in e1_b_mid[name]]}
        if not values["P"]:
            continue
        descriptives[name] = {metric: describe(v) for metric, v in values.items()
                              if len(v) >= 1}
    family_a = contrasts(e1_b, FAMILY_A_B)
    family_b = contrasts(e1_p, FAMILY_A_B)

    # --- E2: Page trend tests ------------------------------------------------------
    def arm_metric(fn: Any) -> tuple[dict[str, dict[int, float | None]],
                                     dict[str, dict[int, bool]]]:
        """Values per arm, and which (arm, seed) cells are a diverged run."""
        values: dict[str, dict[int, float | None]] = {}
        broken: dict[str, dict[int, bool]] = {}
        for arm, rs in e2.items():
            values[arm] = {run.cfg.seed: fn(run) for run in rs}
            broken[arm] = {run.cfg.seed: True
                           for run in e2_all[arm] if run.state == "diverged"}
        return values, broken

    e2_b_values, e2_b_diverged = arm_metric(summaries.B)
    e2_p_values, e2_p_diverged = arm_metric(summaries.P)
    page_results: dict[str, dict[str, Any]] = {}
    for name, metric, ranks, description in FAMILY_C:
        values, broken = (e2_b_values, e2_b_diverged) if metric == "B" \
            else (e2_p_values, e2_p_diverged)
        sign = 1.0 if metric == "B" else -1.0
        blocks, dropped = page_blocks(values, broken, sign)
        if dropped:
            notice(f"{name}: {dropped} seed block(s) dropped (an arm has no run)")
        if blocks.shape[0] < 2:
            notice(f"{name}: skipped, only {blocks.shape[0]} complete seed block(s)")
            continue
        page_results[name] = {**page(blocks, ranks), "description": description,
                              "metric": metric}
        print(f"{name} ({metric}, {description}): L = {page_results[name]['L']:.1f}, "
              f"p = {_p_text(page_results[name]['p'])}, n_blocks = {blocks.shape[0]}, "
              f"method = {page_results[name]['method']}")

    # --- validation and diagnostics ------------------------------------------------
    pilot = [runs[rid] for rid in sorted(ids["pilot"]) if rid in runs]
    e1_all = [run for rs in e1.values() for run in rs]
    v4 = validation_v4(pilot)
    v6 = validation_v6(e1_all)
    v7 = validation_v7(e1)
    precision = measurement_precision(v7)
    noise_path = runs_dir / "_validation" / "noise_e1.json"
    v5 = json.loads(noise_path.read_text()) if noise_path.exists() else None
    if v5 is None:
        notice(f"V5: {noise_path} is absent — numbers.json carries null "
               "(run ./run.sh noise first)")

    diagnostics = {
        "across_condition_per_seed_correlation_P":
            across_condition_correlation(e1_p, "P"),
        "across_condition_per_seed_correlation_B":
            across_condition_correlation(e1_b, "B"),
        "X1_within_condition_Bmid_vs_P": x1_within_condition(e1),
    }

    landing = {}
    for name, rs in e1.items():
        rates = [r for run in rs if (r := summaries.landing_rate(run)) is not None]
        landing[name] = float(np.mean(rates)) if rates else None

    e1_e2_ids = ids["e1"] | ids["e2"]
    present_e1_e2 = sorted(e1_e2_ids & set(runs))  # planned 60; fewer mid-sweep
    compute = {
        "e1_e2_runs_used": len(present_e1_e2),
        "e1_e2_runs_planned": len(e1_e2_ids),
        "e1_e2_total_hours": float(sum(runs[rid].wall_s for rid in present_e1_e2) / 3600.0),
        "all_grids_runs_used": len(wanted & set(runs)),
        "all_grids_total_hours": float(sum(run.wall_s for rid, run in runs.items()
                                           if rid in wanted) / 3600.0),
    }
    divergence = {name: sum(1 for run in runs.values() if run.state == name)
                  for name in ("done", "diverged", "crashed", "running")}

    # --- figures -------------------------------------------------------------------
    written: list[Path] = []
    figures_to_write = []
    if any(not curves.empty for curves in return_curves.values()):
        figures_to_write.append(("fig1_learning_curves",
                                 *figures.fig1_learning_curves(return_curves, family_b)))
        figures_to_write.append(("fig2_bias",
                                 *figures.fig2_bias(e1_curves, e1_b, family_a)))
    else:
        notice("fig1/fig2 skipped: no E1 run has finished")
    if any(not curves.empty for curves in e2_curves.values()):
        figures_to_write.append(("fig3_e2", *figures.fig3_e2(
            e2_curves, {arm: per_seed(e2[arm], summaries.P) for arm in figures.E2_ORDER},
            page_results.get("C0"), page_results.get("C1"))))
    else:
        notice("fig3 skipped: no E2 runs")

    for stem, fig, caption in figures_to_write:
        checks = figures.self_check(fig, caption)  # raises if the checklist fails
        paths = figures.save(fig, caption, fig_dir / stem)
        written += paths
        print(f"# self-check {stem}: OK")
        for line in checks:
            print(f"#   {line}")
    import matplotlib.pyplot as plt

    plt.close("all")

    # --- tables --------------------------------------------------------------------
    e1_runs = [run for rs in e1.values() for run in rs]
    if e1_runs:
        tex, csv = tables.table1_hparams(e1_runs)
        written += tables.write(table_dir, "table1_hparams", tex, csv)
    else:
        notice("table1 skipped: no E1 runs")
    contrast_rows = {f"Family A: {k}": {**v, "metric": "B"} for k, v in family_a.items()
                     if "delta" in v}
    # Family B tests no hypothesis of its own: its labels drop the H1/H2 tags.
    contrast_rows |= {f"Family B: {k.removeprefix('H1: ').removeprefix('H2-')}":
                      {**v, "metric": "P"} for k, v in family_b.items() if "delta" in v}
    e2_descriptives = {}
    for arm in figures.E2_ORDER:
        values = [v for _, v in per_seed(e2[arm], summaries.P)]
        b_values = [v for _, v in per_seed(e2[arm], summaries.B)]
        if values and b_values:
            e2_descriptives[f"TD3/none/{arm}"] = {"P": describe(values),
                                                  "B": describe(b_values)}
    if descriptives or e2_descriptives:
        tex, csv = tables.table2_results({**descriptives, **e2_descriptives},
                                         contrast_rows, page_results)
        written += tables.write(table_dir, "table2_results", tex, csv)
    else:
        notice("table2 skipped: no complete conditions")
    run_table = tables.run_table(index)
    run_table_path = table_dir / "run_table.csv"
    table_dir.mkdir(parents=True, exist_ok=True)
    run_table.to_csv(run_table_path, index=False)
    written.append(run_table_path)

    # --- numbers.json --------------------------------------------------------------
    numbers = {
        "meta": {"runs_dir": str(runs_dir), "n_run_directories": int(len(index)),
                 "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
                 "analysis_seconds": round(time.time() - t0, 1)},
        "coverage": coverage,
        "divergence_counts": divergence,
        "compute": compute,
        "definitions": {
            "P": "mean return_1000 over the last five evaluation steps (260k-300k)",
            "B": "mean checkpoint bias of Q1 over steps >= max_steps/2 (150k-300k)",
            "B_mid": "the same over max_steps/3 <= step <= max_steps/2 (100k-150k)",
            "bias": "mean(q1 - G) over the probe points of one checkpoint",
            "family_A": "Welch two-sided + Holm over the three E1 contrasts on B",
            "family_B": "the same contrasts on P",
            "family_C": "Page trend test, seeds as blocks, columns (min, mean, max)",
            "power": "80 % power only for d ~ 1.3 at n = 10 (protocol §6)",
        },
        "conditions": {name: {"n_runs": int(len(e1_p[name])),
                              "seeds": [s for s, _ in e1_p[name]],
                              "P": descriptives.get(name, {}).get("P"),
                              "B": descriptives.get(name, {}).get("B"),
                              "B_mid": descriptives.get(name, {}).get("B_mid"),
                              "P_values": [v for _, v in e1_p[name]],
                              "B_values": [v for _, v in e1_b[name]],
                              "B_mid_values": [v for _, v in e1_b_mid[name]],
                              "landing_rate": landing.get(name)}
                      for name in e1},
        "family_A_bias_E1": family_a,
        "family_B_performance_E1": family_b,
        "family_C_E2": page_results,
        "e2_conditions": {arm: e2_stats(e2[arm]) for arm in figures.E2_ORDER},
        "measurement_precision": precision,
        "validation": {"V4_step0_pilot": v4, "V5_return_noise": v5,
                       "V6_horizon_artefact": v6, "V7_stability": v7},
        "diagnostics": diagnostics,
    }
    numbers_path = out_dir / "numbers.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    numbers_path.write_text(json.dumps(_jsonable(numbers), indent=2) + "\n")
    written.append(numbers_path)

    # --- report ---------------------------------------------------------------------
    print("\n# files written:")
    for path in written:
        print(f"#   {path}")
    print("\n# numbers (no interpretation):")
    print_numbers(numbers)
    print(f"\n# done in {time.time() - t0:.1f}s")
    return 0


def _p_text(value: float) -> str:
    """A p-value: four decimals, or scientific notation when very small."""
    return f"{value:.4f}" if value >= 1e-4 else f"{value:.2e}"


def _num(value: float | None, digits: int = 2) -> str:
    """A number for the stdout summary; `None` (undefined) prints as `n/a`."""
    return "n/a" if value is None else f"{value:.{digits}f}"


def _jsonable(obj: Any) -> Any:
    """json.dumps with NaN -> null, numpy scalars -> Python scalars."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    return obj


def print_numbers(numbers: dict[str, Any]) -> None:
    """Every computed number, without interpretation (the author reads these)."""
    print("# E1 conditions: n, P [95 % CI], median, B [95 % CI], median")
    for name, stats in numbers["conditions"].items():
        p, b = stats["P"], stats["B"]
        if not p or not b:
            print(f"#   {name}: no data")
            continue
        print(f"#   {name}: n = {p['n']}, P = {p['mean']:.1f} "
              f"[{p['ci_lo']:.1f}, {p['ci_hi']:.1f}], median {p['median']:.1f}; "
              f"B = {b['mean']:.1f} [{b['ci_lo']:.1f}, {b['ci_hi']:.1f}], "
              f"median {b['median']:.1f}")
    print("# Family A (B) and Family B (P): Welch two-sided + Holm")
    for family in ("family_A_bias_E1", "family_B_performance_E1"):
        for name, r in numbers[family].items():
            if "delta" not in r:
                print(f"#   {family} {name}: {r.get('error')}")
                continue
            print(f"#   {family} {name}: Δ = {r['delta']:.2f} "
                  f"[{r['ci_lo']:.2f}, {r['ci_hi']:.2f}], t = {r['t']:.2f}, "
                  f"df = {r['df']:.1f}, p = {_p_text(r['p'])}, "
                  f"p_holm = {_p_text(r['p_holm'])}, g = {r['hedges_g']:.2f}, "
                  f"mw_p = {_p_text(r['mw_p'])}")
    for name, r in numbers["family_C_E2"].items():
        print(f"# Family C {name}: L = {r['L']:.1f}, p = {_p_text(r['p'])}, "
              f"n_blocks = {r['n_blocks']}, method = {r['method']}, "
              f"{r['description']}")
    for arm, stats in numbers["e2_conditions"].items():
        if stats["P"]:
            p, b = stats["P"], stats["B"]
            print(f"# E2 {arm}: n = {p['n']}, P = {p['mean']:.1f} "
                  f"[{p['ci_lo']:.1f}, {p['ci_hi']:.1f}], B = {b['mean']:.1f} "
                  f"[{b['ci_lo']:.1f}, {b['ci_hi']:.1f}]")
    v = numbers["validation"]
    print(f"# V4 step 0: {json.dumps(v['V4_step0_pilot'])}")
    noise = v["V5_return_noise"]
    print(f"# V5 noise: median = {None if not noise else noise['median']}"
          f", p90 = {None if not noise else noise['p90']}")
    print(f"# V6 horizon artefact: {json.dumps(v['V6_horizon_artefact'])}")
    v7 = v["V7_stability"]
    if v7:
        print(f"# V7 stability (within-condition sd of B), passes_all = "
              f"{v7['passes_all']}")
        for name, row in v7["by_condition"].items():
            print(f"#   {name}: sd_B = {_num(row['between_run_sd_B'])}, "
                  f"median|dB| = {_num(row['median_abs_diff'])}, "
                  f"p90|dB| = {_num(row['p90_abs_diff'])}, "
                  f"frac(|dB| < sd) = {_num(row['fraction_runs_below_sd'])}, "
                  f"median half-width = {_num(row['median_half_width'])}, "
                  f"passes = {row['passes']}")
    else:
        print("# V7 stability: not computed")
    for name, row in numbers["measurement_precision"]["by_condition"].items():
        print(f"# precision {name}: n_checkpoints = {row['n_checkpoints']}, "
              f"median half-width = {_num(row['median_half_width'])}, "
              f"p90 = {_num(row['p90_half_width'])}, "
              f"sd_B = {_num(row['between_run_sd_B'])}")
    print(f"# compute: {json.dumps(numbers['compute'])}")
    print(f"# divergence counts: {json.dumps(numbers['divergence_counts'])}")
    print(f"# diagnostics: {json.dumps(numbers['diagnostics'])}")


if __name__ == "__main__":
    sys.exit(main())
