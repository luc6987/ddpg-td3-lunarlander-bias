"""`train.py`: the invariants that no artifact shows.

I1 (a time-limit truncation is not a termination), I2 and I3 (uniform random
actions up to `learning_starts`, then clip(pi(s) + N(0, 0.1^2), -1, 1)) and the
NaN semantics of `Window` are properties of the loop itself. They are checked
here with a recording chain environment, so the assertions are exact instead of
statistical about LunarLander.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import torch
from conftest import ChainEnv
from gymnasium.wrappers import TimeLimit

from lunar_bias import grids, runner, train
from lunar_bias.buffer import ReplayBuffer
from lunar_bias.config import EVAL_SEEDS, RunConfig
from lunar_bias.envs import ACT_DIM
from lunar_bias.io import read_json


class RecordingChain(ChainEnv):
    """Chain env that records the (pre-step observation, action) of every step."""

    def __init__(self, length: int | None = None):
        super().__init__(length)
        self.steps: list[tuple[np.ndarray, np.ndarray]] = []

    def step(self, action):
        self.steps.append((self._obs(), np.asarray(action, dtype=np.float32).copy()))
        return super().step(action)


class RecordingBuffer(ReplayBuffer):
    """Replay buffer that records the `terminated` flag it was given."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.flags: list[bool] = []

    def add(self, obs, act, rew, next_obs, terminated):
        self.flags.append(bool(terminated))
        super().add(obs, act, rew, next_obs, terminated)


class ConstAgent:
    """A stand-in agent whose deterministic policy is a known constant action."""

    n_critics = 1

    def __init__(self, value: float = 0.5):
        self.value = value
        self.n_updates = 0
        self.n_actor_updates = 0
        self.last_actor_loss = float("nan")
        # the checkpoint writer asks any agent for state_dicts
        self.actor = torch.nn.Linear(1, 1)
        self.critics = torch.nn.ModuleList([torch.nn.Linear(1, 1)])

    def act(self, obs: np.ndarray) -> np.ndarray:
        return np.full(ACT_DIM, self.value, dtype=np.float32)

    def q_values(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        return torch.zeros(self.n_critics, obs.shape[0])

    def update(self, batch) -> dict[str, float]:
        self.n_updates += 1
        return {"critic_loss": 1.0, "q1_batch_mean": 0.0, "q_abs_max": 0.0,
                "critic_grad_norm": 0.0, "actor_loss": float("nan")}


class DivergingAgent(ConstAgent):
    """An agent whose critic loss goes non-finite, as a diverged run does (I11)."""

    def update(self, batch) -> dict[str, float]:
        self.n_updates += 1
        stats = {"critic_loss": 1.0, "q1_batch_mean": 0.0, "q_abs_max": 0.0,
                 "critic_grad_norm": 0.0, "actor_loss": float("nan")}
        stats["critic_loss"] = float("nan")
        return stats


def tiny_config(**overrides) -> RunConfig:
    """A 12-step run on the chain environment: enough to exercise the loop."""
    base = grids.smoke()[0]
    ev = dataclasses.replace(base.evaluation, every=10**9, checkpoint_every=10**9,
                             train_horizon=5, probe_horizon=20, seeds=EVAL_SEEDS[:1])
    cfg = dataclasses.replace(
        base, max_steps=12, log_every=12, evaluation=ev,
        algo=dataclasses.replace(base.algo, learning_starts=4))
    return dataclasses.replace(cfg, **overrides) if overrides else cfg


def _run(monkeypatch, runs_dir: Path, cfg: RunConfig, *, length: int | None = None,
         agent: ConstAgent | None = None, expect: str = "done"):
    """Run `train` on a recording chain env; return (training env, buffer)."""
    holder: dict[str, list] = {"envs": [], "buffers": []}

    def make_env_(max_episode_steps: int):
        env = TimeLimit(RecordingChain(length), max_episode_steps=max_episode_steps)
        holder["envs"].append(env)  # [0] is the training env, [1] the probe env
        return env

    def make_buffer(*args, **kwargs):
        buffer = RecordingBuffer(*args, **kwargs)
        holder["buffers"].append(buffer)
        return buffer

    monkeypatch.setattr(train, "make_env", make_env_)
    monkeypatch.setattr(train, "ReplayBuffer", make_buffer)
    if agent is not None:
        monkeypatch.setattr(train, "make_agent", lambda cfg_: agent)
    state = runner.run(cfg, runs_dir)
    error = (read_json(runs_dir / cfg.run_id / "status.json") or {}).get("error")
    assert state == expect, f"{state}: {error}"
    return holder["envs"][0].env, holder["buffers"][0]


def test_a_truncation_is_not_stored_as_termination(monkeypatch, tmp_path: Path):
    """I1: the chain never terminates; every episode ends by the time limit."""
    cfg = tiny_config()  # train_horizon = 5: an episode every 5 steps
    env, buffer = _run(monkeypatch, tmp_path, cfg, length=None)

    assert len(env.steps) == cfg.max_steps
    assert [int(obs[0]) for obs, _ in env.steps] == [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1]
    assert buffer.flags == [False] * cfg.max_steps, "a truncation is not a termination"


def test_a_termination_is_stored_as_termination(monkeypatch, tmp_path: Path):
    """I1, other half: bootstrapping past a real termination would be a bug."""
    cfg = tiny_config(max_steps=9)
    env, buffer = _run(monkeypatch, tmp_path, cfg, length=3)  # terminates every 3rd step

    assert [int(obs[0]) for obs, _ in env.steps] == [0, 1, 2] * 3
    assert buffer.flags == [False, False, True] * 3


def test_warmup_is_uniform_random_then_the_noisy_policy(monkeypatch, tmp_path: Path):
    """I2 and I3: uniform actions up to learning_starts, then clip(pi(s) + N(0, 0.1^2))."""
    cfg = tiny_config(max_steps=108, log_every=108,
                      algo=dataclasses.replace(tiny_config().algo, learning_starts=8))
    env, _ = _run(monkeypatch, tmp_path, cfg, agent=ConstAgent(0.5))
    actions = np.asarray([action for _, action in env.steps], dtype=np.float64)
    assert actions.shape == (cfg.max_steps, ACT_DIM)

    warm_up, policy = actions[:8], actions[8:]
    assert np.all(np.abs(warm_up) <= 1.0), "warm-up actions are in [-1, 1]^2"
    assert not np.any(warm_up == 0.5), "warm-up must not use the policy"
    assert warm_up.max() - warm_up.min() > 0.5 and abs(warm_up.mean()) < 0.3, "not uniform"

    assert np.all(np.abs(policy) <= 1.0), "the action is clipped"
    assert np.any(policy != 0.5), "exploration noise must be applied"
    assert 0.06 < policy.std() < 0.16, "the exploration noise is sigma = 0.1"
    assert np.abs(policy - 0.5).max() <= 0.5 + 1e-6


def test_a_non_finite_critic_loss_stops_the_run_as_diverged(monkeypatch, tmp_path: Path):
    """I11: divergence is a result, not a bug. It is final and never retried."""
    cfg = tiny_config()  # the first gradient step is at step 4
    _run(monkeypatch, tmp_path, cfg, agent=DivergingAgent(), expect="diverged")

    status = read_json(tmp_path / cfg.run_id / "status.json")
    assert status["state"] == "diverged" and status["step"] == 4
    assert status["error"] == "non-finite critic loss at step 4"
    assert runner.run(cfg, tmp_path) == "skipped"


def test_window_flush_is_nan_aware():
    """No update (or no episode) in the window => NaN, never a misleading 0."""
    window = train.Window()
    empty = window.flush()
    for key in ("critic_loss", "actor_loss", "q1_batch_mean", "q_abs_max",
                "critic_grad_norm", "train_return_mean"):
        assert np.isnan(empty[key]), key
    assert empty["train_episodes"] == 0

    window.add_update({"critic_loss": 2.0, "actor_loss": float("nan"),
                       "q1_batch_mean": 0.5, "q_abs_max": 7.0, "critic_grad_norm": 1.0})
    assert np.isnan(window.flush()["actor_loss"]), "an all-NaN mean is NaN, not an error"

    window.add_update({"critic_loss": 2.0, "actor_loss": float("nan"),
                       "q1_batch_mean": 0.5, "q_abs_max": 7.0, "critic_grad_norm": 1.0})
    window.add_update({"critic_loss": 4.0, "actor_loss": 3.0,
                       "q1_batch_mean": 1.5, "q_abs_max": 2.0, "critic_grad_norm": 3.0})
    window.add_episode(-10.0)
    window.add_episode(-20.0)
    out = window.flush()
    assert out["critic_loss"] == 3.0
    assert out["actor_loss"] == 3.0, "NaN-aware mean"
    assert out["q1_batch_mean"] == 1.0
    assert out["q_abs_max"] == 7.0, "the max, not the mean"
    assert out["critic_grad_norm"] == 2.0
    assert out["train_episodes"] == 2
    assert out["train_return_mean"] == -15.0
    assert np.isnan(window.flush()["critic_loss"]), "flush resets the window"


def test_ddpg_updates_every_step_and_td3_delays(monkeypatch, tmp_path: Path):
    """I3/I5: UTD = 1 gradient step per env step; TD3 updates the actor every 2nd."""
    for algo, expected_actor_updates in (("ddpg", 9), ("td3", 4)):
        base = tiny_config()
        cfg = dataclasses.replace(
            base, algo=dataclasses.replace(base.algo, name=algo, learning_starts=4))
        _run(monkeypatch, tmp_path / algo, cfg, length=None)
        metrics = tmp_path / algo / cfg.run_id / "metrics.csv"
        tail = metrics.read_text().strip().splitlines()[1].split(",")
        step, n_updates, n_actor = int(tail[0]), int(tail[2]), int(tail[3])
        assert step == cfg.max_steps
        assert n_updates == cfg.max_steps - 4 + 1, "one update per step after warm-up"
        assert n_actor == expected_actor_updates, "policy_delay 2 halves the actor steps"
