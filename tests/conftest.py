"""Shared test doubles: a chain environment with closed-form values, and a stub
critic whose Q-values are a function of the observation.

They exist so that the probe pipeline can be checked against a known truth
(V3) without a neural network, on deviations of the order of the float64
rounding of the return recursion.
"""

from __future__ import annotations

from collections.abc import Callable

import gymnasium as gym
import numpy as np
import torch
from gymnasium.wrappers import TimeLimit
from torch import Tensor

from lunar_bias.envs import ACT_DIM, OBS_DIM


class ChainEnv(gym.Env):
    """Reward 1.0 per step, observation `[t, 0, ..., 0]`, optional termination.

    `length=None` never terminates (the TimeLimit wrapper then ends it). The
    returned observation carries the number of steps taken so far, so a stored
    probe point at index `k` has `t = k`: the probe's horizon bookkeeping can be
    checked against the closed form (1 - gamma**(L - t)) / (1 - gamma).
    """

    def __init__(self, length: int | None = None, obs_dim: int = OBS_DIM):
        self.length = length
        self.t = 0
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(ACT_DIM,), dtype=np.float32)

    def _obs(self) -> np.ndarray:
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        obs[0] = self.t
        return obs

    def reset(self, *, seed: int | None = None,
              options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.t = 0
        return self._obs(), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        self.t += 1
        terminated = self.length is not None and self.t >= self.length
        return self._obs(), 1.0, terminated, False, {}


def chain_env_fn(length: int | None) -> Callable[[int], gym.Env]:
    """An `env_fn` for `evaluate_and_probe`: chain + time limit."""
    def make(horizon: int) -> gym.Env:
        return TimeLimit(ChainEnv(length), max_episode_steps=horizon)
    return make


class StubAgent:
    """`n_critics` critics whose Q-values come from `q_fn(obs)`.

    `q_fn` receives the observation batch as float64 `[N, obs_dim]` and returns
    `[N]` (broadcast to all critics) or `[n_critics, N]`.
    """

    def __init__(self, q_fn: Callable[[np.ndarray], np.ndarray] | None = None,
                 n_critics: int = 1, action: float = 0.0):
        self.q_fn = q_fn if q_fn is not None else (lambda obs: np.zeros(len(obs)))
        self.n_critics = n_critics
        self.action = action

    def act(self, obs: np.ndarray) -> np.ndarray:
        return np.full(ACT_DIM, self.action, dtype=np.float32)

    def q_values(self, obs: Tensor, act: Tensor) -> Tensor:
        q = np.asarray(self.q_fn(obs.detach().numpy().astype(np.float64)), dtype=np.float32)
        if q.ndim == 1:
            q = np.tile(q, (self.n_critics, 1))
        assert q.shape == (self.n_critics, obs.shape[0])
        return torch.from_numpy(q)
