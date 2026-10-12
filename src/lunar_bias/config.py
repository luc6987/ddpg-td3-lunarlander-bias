"""Frozen configuration for one training run.

A run is a pure function of its config: the config is serialised to
`runs/<run_id>/config.json`, and `run_id` is the sha1 of that config *without*
`tag`. Two grids that contain the same condition therefore share one run
(`docs/implementation_plan.md` §3), and conditions are always identified from
config fields or grid membership, never from `tag`.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Literal

AlgoName = Literal["ddpg", "td3"]
Norm = Literal["none", "layer"]
Reduction = Literal["min", "mean", "max"]

#: Evaluation / probe seeds: fixed, shared by every run and every checkpoint.
EVAL_SEEDS: tuple[int, ...] = tuple(range(1_000_000, 1_000_020))


@dataclass(frozen=True)
class NetConfig:
    hidden: tuple[int, ...] = (64, 64)
    norm: Norm = "none"  # "layer": LayerNorm after each hidden Linear, before the ReLU


@dataclass(frozen=True)
class AlgoConfig:
    name: AlgoName = "td3"
    gamma: float = 0.99
    tau: float = 0.005
    lr_actor: float = 1e-3
    lr_critic: float = 1e-3
    batch_size: int = 256
    buffer_size: int = 1_000_000
    # uniform-random actions for this many steps; no gradient update before
    learning_starts: int = 10_000
    action_noise: float = 0.1
    utd: int = 1  # gradient steps per environment step (single environment)
    # --- TD3 only (ignored by DDPG; grids keep the defaults for DDPG) ---
    policy_delay: int = 2
    target_noise: float = 0.2
    target_noise_clip: float = 0.5
    target_reduction: Reduction = "min"


@dataclass(frozen=True)
class EvalConfig:
    every: int = 10_000
    seeds: tuple[int, ...] = EVAL_SEEDS
    train_horizon: int = 1_000  # training time limit; performance = first 1 000 steps
    # probe rollouts continue past the limit, so that G_t (t < 1000) is never truncated
    probe_horizon: int = 1_500
    checkpoint_every: int = 50_000


@dataclass(frozen=True)
class RunConfig:
    seed: int = 0
    max_steps: int = 300_000
    log_every: int = 1_000
    actor: NetConfig = NetConfig()
    critic: NetConfig = NetConfig()
    algo: AlgoConfig = AlgoConfig()
    evaluation: EvalConfig = EvalConfig()
    tag: str = ""  # human label; NOT part of run_id

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, indent=2)

    @property
    def run_id(self) -> str:
        d = self.to_dict()
        d.pop("tag")
        blob = json.dumps(d, sort_keys=True, separators=(",", ":"))
        return hashlib.sha1(blob.encode()).hexdigest()[:10]

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RunConfig:
        def net(x: dict[str, Any]) -> NetConfig:
            return NetConfig(hidden=tuple(x["hidden"]), norm=x["norm"])

        ev = dict(d["evaluation"])
        ev["seeds"] = tuple(ev["seeds"])
        return cls(
            seed=d["seed"], max_steps=d["max_steps"], log_every=d["log_every"],
            actor=net(d["actor"]), critic=net(d["critic"]),
            algo=AlgoConfig(**d["algo"]), evaluation=EvalConfig(**ev),
            tag=d.get("tag", ""),
        )


if __name__ == "__main__":
    print(RunConfig().to_json())
