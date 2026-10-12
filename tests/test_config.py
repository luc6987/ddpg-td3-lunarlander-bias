"""`config.py`: the config is the run's identity (docs/implementation_plan.md §3).

`run_id` is the sha1 of the config *without* `tag`. Everything else in the
config is scientific: changing it must produce a different run directory.
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
from typing import Any

import pytest

from lunar_bias import grids
from lunar_bias.config import EVAL_SEEDS, AlgoConfig, EvalConfig, NetConfig, RunConfig


def _with(**kwargs: Any) -> RunConfig:
    """Default config with top-level fields replaced."""
    return dataclasses.replace(RunConfig(), **kwargs)


def _algo(**kwargs: Any) -> RunConfig:
    return _with(algo=dataclasses.replace(AlgoConfig(), **kwargs))


def _eval(**kwargs: Any) -> RunConfig:
    return _with(evaluation=dataclasses.replace(EvalConfig(), **kwargs))


#: One (label, config) pair per config field that is not `tag`.
VARIANTS: list[tuple[str, RunConfig]] = [
    ("seed", _with(seed=1)),
    ("max_steps", _with(max_steps=300_001)),
    ("log_every", _with(log_every=2_000)),
    ("actor.hidden", _with(actor=NetConfig(hidden=(32, 32)))),
    ("actor.norm", _with(actor=NetConfig(norm="layer"))),
    ("critic.hidden", _with(critic=NetConfig(hidden=(32, 32)))),
    ("critic.norm", _with(critic=NetConfig(norm="layer"))),
    ("algo.name", _algo(name="ddpg")),
    ("algo.gamma", _algo(gamma=0.98)),
    ("algo.tau", _algo(tau=0.01)),
    ("algo.lr_actor", _algo(lr_actor=1e-4)),
    ("algo.lr_critic", _algo(lr_critic=1e-4)),
    ("algo.batch_size", _algo(batch_size=128)),
    ("algo.buffer_size", _algo(buffer_size=100_000)),
    ("algo.learning_starts", _algo(learning_starts=5_000)),
    ("algo.action_noise", _algo(action_noise=0.2)),
    ("algo.utd", _algo(utd=4)),
    ("algo.policy_delay", _algo(policy_delay=3)),
    ("algo.target_noise", _algo(target_noise=0.1)),
    ("algo.target_noise_clip", _algo(target_noise_clip=0.25)),
    ("algo.target_reduction", _algo(target_reduction="mean")),
    ("evaluation.every", _eval(every=5_000)),
    ("evaluation.seeds", _eval(seeds=EVAL_SEEDS[:5])),
    ("evaluation.train_horizon", _eval(train_horizon=900)),
    ("evaluation.probe_horizon", _eval(probe_horizon=1_400)),
    ("evaluation.checkpoint_every", _eval(checkpoint_every=25_000)),
]


def test_run_id_is_stable_and_short():
    cfg = RunConfig()
    assert cfg.run_id == RunConfig().run_id
    assert len(cfg.run_id) == 10
    assert all(c in "0123456789abcdef" for c in cfg.run_id)


def test_json_round_trip_keeps_run_id():
    cfg = RunConfig(seed=3, tag="e1")
    restored = RunConfig.from_dict(json.loads(cfg.to_json()))
    assert restored == cfg
    assert restored.run_id == cfg.run_id


def test_json_round_trip_keeps_tuples():
    restored = RunConfig.from_dict(json.loads(RunConfig().to_json()))
    assert isinstance(restored.evaluation.seeds, tuple)
    assert isinstance(restored.actor.hidden, tuple)
    assert isinstance(restored.critic.hidden, tuple)


def test_tag_is_not_part_of_run_id():
    a = dataclasses.replace(RunConfig(), tag="pilot")
    b = dataclasses.replace(RunConfig(), tag="e1")
    c = dataclasses.replace(RunConfig(), tag="")
    assert a.run_id == b.run_id == c.run_id


@pytest.mark.parametrize(("label", "variant"), VARIANTS, ids=[v[0] for v in VARIANTS])
def test_scientific_fields_change_run_id(label: str, variant: RunConfig):
    assert variant.run_id != RunConfig().run_id, f"changing {label} must change run_id"


def test_scientific_variants_are_pairwise_distinct():
    """No two different conditions may collide on the same run directory."""
    ids = [cfg.run_id for _, cfg in VARIANTS]
    assert len(set(ids)) == len(ids)


def test_eval_seeds_are_the_fixed_ones():
    assert tuple(range(1_000_000, 1_000_020)) == EVAL_SEEDS
    assert RunConfig().evaluation.seeds == EVAL_SEEDS


def test_grid_sizes_after_deduplication():
    """E2's `min` arm is E1's TD3/none; E3's UTD-1 and E4's depth-2 arms are E1 runs."""
    assert len(grids.unique(grids.pilot())) == 6
    assert len(grids.unique(grids.e1())) == 40
    assert len(grids.unique(grids.e1() + grids.e2())) == 60
    assert len(grids.unique(grids.e1() + grids.e2() + grids.e3() + grids.e4())) == 80
    assert len(grids.unique(grids.smoke())) == 2
    assert len(grids.e2()) == 30 and len(grids.e3()) == 20 and len(grids.e4()) == 20


def test_shared_arms_are_the_same_run():
    e1_ids = {cfg.run_id for cfg in grids.e1()}
    e2_min = {cfg.run_id for cfg in grids.e2() if cfg.algo.target_reduction == "min"}
    e3_utd1 = {cfg.run_id for cfg in grids.e3() if cfg.algo.utd == 1}
    e4_depth2 = {cfg.run_id for cfg in grids.e4() if len(cfg.critic.hidden) == 2}
    assert e2_min and e2_min <= e1_ids
    assert e3_utd1 and e3_utd1 <= e1_ids
    assert e4_depth2 and e4_depth2 <= e1_ids
    # the pilot is E1's DDPG/TD3 without LN on seeds 0-2
    assert {cfg.run_id for cfg in grids.pilot()} <= e1_ids
    assert {cfg.seed for cfg in grids.pilot()} == {0, 1, 2}


def test_grid_conditions_never_carry_layer_norm_in_the_actor():
    for name in ("smoke", "pilot", "e1", "e2", "e3", "e4"):
        for cfg in grids.GRIDS[name]():
            assert cfg.actor.norm == "none", f"{name} puts LN in the actor"


def test_module_main_prints_the_default_config():
    proc = subprocess.run(
        [sys.executable, "-m", "lunar_bias.config"],
        capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert RunConfig.from_dict(json.loads(proc.stdout)) == RunConfig()
