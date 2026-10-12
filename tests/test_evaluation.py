"""V3 — the probe pipeline against a known truth, plus a LunarLander sanity check.

On a chain environment the return-to-go has a closed form, (1 - gamma**(L-t)) /
(1 - gamma), so a stub critic returning Q^pi + c must give a bias of exactly c,
and every piece of horizon bookkeeping is checkable. The last test runs the
real pipeline on LunarLander with gymnasium's heuristic controller.
"""

from __future__ import annotations

import types

import numpy as np
import pytest
import torch
from conftest import ChainEnv, StubAgent, chain_env_fn
from gymnasium.envs.box2d.lunar_lander import heuristic
from gymnasium.wrappers import TimeLimit
from torch import Tensor

from lunar_bias.envs import ACT_DIM, make_env
from lunar_bias.evaluation import evaluate_and_probe, returns_to_go

GAMMA = 0.99


def _closed_form(length: int, t: np.ndarray, gamma: float = GAMMA) -> np.ndarray:
    """sum_{k=0}^{L-1-t} gamma**k for a chain that terminates at `length`."""
    return (1.0 - gamma ** (length - t)) / (1.0 - gamma)


# --- known truth -----------------------------------------------------------


def test_terminating_chain_has_the_closed_form_return_to_go():
    length = 8
    data = evaluate_and_probe(
        StubAgent(), seeds=(1, 2, 3), gamma=GAMMA, train_horizon=1_000,
        probe_horizon=1_500, env_fn=chain_env_fn(length))

    t = data.arrays["t"].astype(np.int64)
    np.testing.assert_allclose(data.arrays["G"], _closed_form(length, t), rtol=0,
                               atol=1e-12)
    assert data.arrays["G"].dtype == np.float64
    assert data.arrays["q"].shape == (1, 3 * length)
    assert (data.arrays["ep_len"] == length).all()
    assert data.arrays["ep_terminated"].all()
    assert [row["outcome"] for row in data.episodes] == ["terminated"] * 3
    assert [row["seed"] for row in data.episodes] == [1, 2, 3]


def test_a_constant_critic_offset_is_returned_as_the_bias():
    """Q^pi + c over a chain => bias = c, to 1e-6 (the stored q is float32)."""
    length, c = 8, 3.5

    def q_fn(obs: np.ndarray) -> np.ndarray:
        return _closed_form(length, obs[:, 0].astype(np.int64)) + c

    data = evaluate_and_probe(
        StubAgent(q_fn), seeds=(1, 2, 3), gamma=GAMMA, train_horizon=1_000,
        probe_horizon=1_500, env_fn=chain_env_fn(length))

    bias = float(np.mean(data.arrays["q"][0] - data.arrays["G"]))
    assert bias == pytest.approx(c, abs=1e-6)
    per_point = np.abs(data.arrays["q"][0] - data.arrays["G"] - c)
    assert per_point.max() < 1e-5, "float32 storage of q must not move the bias"


def test_non_terminating_chain_horizon_bookkeeping():
    """Only t < train_horizon is stored, G uses the whole rollout, and the
    episode counts as a truncation, not a termination."""
    train_horizon, probe_horizon = 30, 50
    data = evaluate_and_probe(
        StubAgent(), seeds=(1, 2), gamma=GAMMA, train_horizon=train_horizon,
        probe_horizon=probe_horizon, env_fn=chain_env_fn(None))
    a = data.arrays

    assert len(a["t"]) == 2 * train_horizon
    assert list(a["t"][:train_horizon]) == list(range(train_horizon))
    assert list(a["episode"]) == [0] * train_horizon + [1] * train_horizon

    t = a["t"].astype(np.int64)
    np.testing.assert_allclose(a["G"], _closed_form(probe_horizon, t), rtol=0, atol=1e-12)
    np.testing.assert_allclose(a["ep_G_at_train_horizon"],
                               _closed_form(probe_horizon, np.int64(train_horizon)))
    np.testing.assert_allclose(a["ep_return_1000"], float(train_horizon))
    np.testing.assert_array_equal(a["ep_len"], [probe_horizon, probe_horizon])
    assert not a["ep_terminated"].any()
    assert [row["outcome"] for row in data.episodes] == ["timeout", "timeout"]
    assert [row["length_1000"] for row in data.episodes] == [train_horizon, train_horizon]


def test_a_terminal_episode_ends_at_its_last_reward():
    """G is the discounted sum to the end: the last point has G = r, not a tail."""
    data = evaluate_and_probe(
        StubAgent(), seeds=(1,), gamma=GAMMA, train_horizon=1_000,
        probe_horizon=1_500, env_fn=chain_env_fn(3))
    np.testing.assert_allclose(data.arrays["G"], _closed_form(3, np.arange(3)), rtol=0,
                               atol=1e-12)
    assert data.arrays["G"][-1] == pytest.approx(1.0)
    np.testing.assert_allclose(data.arrays["ep_G_at_train_horizon"], 0.0)


def test_q_is_evaluated_at_the_policy_action_on_the_visited_states():
    """I8: the stored q pairs each state with the action the policy took there."""
    recorded: dict[str, np.ndarray] = {}
    holder: dict[str, ChainEnv] = {}

    class RecordingChain(ChainEnv):
        def step(self, action):
            self.actions.append(np.asarray(action, dtype=np.float32))
            return super().step(action)

    class RecordingAgent(StubAgent):
        def act(self, obs: np.ndarray) -> np.ndarray:
            return np.full(ACT_DIM, (obs[0] % 3) - 1.0, dtype=np.float32)

        def q_values(self, obs: Tensor, act: Tensor) -> Tensor:
            recorded["obs"] = obs.numpy().copy()
            recorded["act"] = act.numpy().copy()
            return super().q_values(obs, act)

    def env_fn(horizon: int):
        env = RecordingChain(4)
        env.actions = []
        holder["env"] = env
        return TimeLimit(env, max_episode_steps=horizon)

    data = evaluate_and_probe(
        RecordingAgent(), seeds=(1, 2), gamma=GAMMA, train_horizon=1_000,
        probe_horizon=1_500, env_fn=env_fn)

    env_actions = np.asarray(holder["env"].actions, dtype=np.float32)
    assert env_actions.shape == (2 * 4, ACT_DIM)
    np.testing.assert_array_equal(recorded["act"], env_actions)
    assert not np.allclose(env_actions, 0.0), "the stub must take non-trivial actions"
    # the observations are the pre-step ones, in order: obs[:, 0] is the step index
    np.testing.assert_array_equal(recorded["obs"][:, 0], data.arrays["t"])
    assert data.arrays["q"].shape == (1, len(env_actions))


# --- real environment ------------------------------------------------------


#: `heuristic` only reads `env.unwrapped.continuous`, so a namespace is enough.
_ENV_SHIM = types.SimpleNamespace(unwrapped=types.SimpleNamespace(continuous=True))


class _HeuristicAgent:
    """Gymnasium's heuristic controller, as an agent, for the probe pipeline."""

    def act(self, obs: np.ndarray) -> np.ndarray:
        return np.asarray(heuristic(_ENV_SHIM, obs), dtype=np.float32)

    def q_values(self, obs: Tensor, act: Tensor) -> Tensor:
        return torch.zeros(1, obs.shape[0])


def _independent_return(seed: int, horizon: int) -> float:
    env = make_env(horizon)
    obs, _ = env.reset(seed=seed)
    rewards: list[float] = []
    terminated = truncated = False
    while not (terminated or truncated):
        obs, reward, terminated, truncated, _ = env.step(_HeuristicAgent().act(obs))
        rewards.append(float(reward))
    env.close()
    return float(np.sum(rewards[:1_000]))


def test_lunarlander_with_the_heuristic_controller_is_reproducible():
    """Two probe passes on the same seeds give identical raw arrays."""
    seeds = (1_000_000, 1_000_001)
    first = evaluate_and_probe(_HeuristicAgent(), seeds, GAMMA, 1_000, 1_500)
    second = evaluate_and_probe(_HeuristicAgent(), seeds, GAMMA, 1_000, 1_500)
    assert set(first.arrays) == set(second.arrays)
    for key, value in first.arrays.items():
        np.testing.assert_array_equal(value, second.arrays[key], err_msg=key)
    assert [row["outcome"] for row in first.episodes] == [
        row["outcome"] for row in second.episodes]

    a = first.arrays
    for i, row in enumerate(first.episodes):
        independent = _independent_return(row["seed"], 1_500)
        assert row["return_1000"] == pytest.approx(independent, abs=1e-6)
        # every step t < min(T, 1000) is stored, and nothing beyond it
        n = min(int(a["ep_len"][i]), 1_000)
        assert int((a["episode"] == i).sum()) == n
        np.testing.assert_array_equal(a["t"][a["episode"] == i], np.arange(n))
        assert row["length_1000"] == n


def test_returns_to_go_matches_a_manual_sum():
    rewards = np.array([1.0, -2.0, 3.0, 0.5], dtype=np.float64)
    expected = np.array([1 - 2 * 0.99 + 3 * 0.99**2 + 0.5 * 0.99**3,
                         -2 + 3 * 0.99 + 0.5 * 0.99**2,
                         3 + 0.5 * 0.99,
                         0.5])
    np.testing.assert_allclose(returns_to_go(rewards, 0.99), expected, rtol=0, atol=1e-15)
    assert returns_to_go(np.zeros(0), 0.99).dtype == np.float64
