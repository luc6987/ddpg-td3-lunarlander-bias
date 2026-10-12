"""The analysis pipeline: `runs/` in, figures, tables and numbers out.

Training writes raw data; every statistic lives here (`docs/implementation_plan.md`
§0.4, §12). `scripts/analyze.py` is the CLI that ties the modules together.

The submodules are imported lazily by the CLI: `figures` pulls in matplotlib, so
it is deliberately not imported here.
"""

from __future__ import annotations

from lunar_bias.analysis.load import RunData, load_eval_episodes, load_index, load_runs
from lunar_bias.analysis.stats import (
    cluster_bootstrap,
    hedges_g,
    holm,
    page,
    t_interval,
    welch,
)
from lunar_bias.analysis.summaries import (
    B,
    B_mid,
    P,
    checkpoint_bias,
    horizon_artefact,
    stability,
)

__all__ = [
    "B", "B_mid", "P", "RunData", "checkpoint_bias", "cluster_bootstrap",
    "hedges_g", "holm", "horizon_artefact", "load_eval_episodes", "load_index",
    "load_runs", "page", "stability", "t_interval", "welch",
]
