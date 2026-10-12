"""The named grids, as lists of run configs (`docs/implementation_plan.md` §11).

`run_id` excludes `tag`, so a condition shared by two grids is ONE run
(E2's `min` arm is E1's TD3/none; E3's UTD-1 and E4's depth-2 arms are E1 runs).
Consequently the analysis identifies conditions from config fields or from grid
membership, never from `tag`.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

from lunar_bias.config import (
    EVAL_SEEDS,
    AlgoConfig,
    AlgoName,
    EvalConfig,
    NetConfig,
    Norm,
    Reduction,
    RunConfig,
)

WIDTH = 256  # fallback F1 sets 256 (docs/protocol.md §5); change only through Gate G1
S10 = tuple(range(10))
S5 = tuple(range(5))


def condition(*, algo: AlgoName = "td3", norm: Norm = "none", seed: int = 0,
              reduction: Reduction = "min", utd: int = 1, depth: int = 2,
              tag: str = "") -> RunConfig:
    hidden = (WIDTH,) * depth
    return RunConfig(seed=seed, tag=tag,
                     actor=NetConfig(hidden=hidden),  # the actor never has LN
                     critic=NetConfig(hidden=hidden, norm=norm),
                     algo=AlgoConfig(name=algo, target_reduction=reduction, utd=utd))


def pilot() -> list[RunConfig]:
    return [condition(algo=a, seed=s, tag="pilot")
            for s in (0, 1, 2) for a in ("ddpg", "td3")]


def e1() -> list[RunConfig]:
    return [condition(algo=a, norm=n, seed=s, tag="e1")
            for s in S10 for a in ("ddpg", "td3") for n in ("none", "layer")]


def e2() -> list[RunConfig]:
    return [condition(reduction=r, seed=s, tag="e2")
            for s in S10 for r in ("min", "mean", "max")]


def e3() -> list[RunConfig]:
    return [condition(norm=n, utd=u, seed=s, tag="e3")
            for s in S5 for n in ("none", "layer") for u in (1, 4)]


def e4() -> list[RunConfig]:
    return [condition(norm=n, depth=d, seed=s, tag="e4")
            for s in S5 for n in ("none", "layer") for d in (2, 3)]


def smoke() -> list[RunConfig]:
    ev = EvalConfig(every=1_000, seeds=EVAL_SEEDS[:2], checkpoint_every=1_500)
    out = []
    for algo, norm in (("ddpg", "none"), ("td3", "layer")):
        c = condition(algo=algo, norm=norm, tag="smoke")
        out.append(dataclasses.replace(
            c, max_steps=3_000, log_every=500, evaluation=ev,
            algo=dataclasses.replace(c.algo, learning_starts=1_000)))
    return out


GRIDS: dict[str, Callable[[], list[RunConfig]]] = {
    "smoke": smoke, "pilot": pilot, "e1": e1, "e2": e2, "e3": e3, "e4": e4,
}


def unique(configs: list[RunConfig]) -> list[RunConfig]:
    """De-duplicate by `run_id`, keeping the first config (grid order is priority)."""
    seen: dict[str, RunConfig] = {}
    for cfg in configs:
        seen.setdefault(cfg.run_id, cfg)
    return list(seen.values())


def describe(cfg: RunConfig) -> str:
    """A one-line condition description, built from config fields (never from `tag`)."""
    return (f"{cfg.algo.name}/norm={cfg.critic.norm}/red={cfg.algo.target_reduction}"
            f"/utd={cfg.algo.utd}/depth={len(cfg.critic.hidden)}/width={cfg.critic.hidden[0]}")
