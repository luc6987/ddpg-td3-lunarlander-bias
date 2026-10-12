"""Replay buffer: a numpy ring buffer that stores `terminated` only.

A time-limit truncation is *not* a termination: the critic bootstraps through
it (invariant I1, `docs/protocol.md` §1.2). The caller therefore passes the
`terminated` flag from `env.step`, never `terminated or truncated`.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import torch
from torch import Tensor


class Batch(NamedTuple):
    obs: Tensor       # [B, 8] float32
    act: Tensor       # [B, 2] float32
    rew: Tensor       # [B]    float32
    next_obs: Tensor  # [B, 8] float32
    term: Tensor      # [B]    float32, 1.0 only for true termination (never truncation)


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, act_dim: int, rng: np.random.Generator):
        self.obs = np.zeros((capacity, obs_dim), np.float32)
        self.act = np.zeros((capacity, act_dim), np.float32)
        self.rew = np.zeros(capacity, np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), np.float32)
        self.term = np.zeros(capacity, np.float32)
        self.capacity, self.size, self.ptr, self.rng = capacity, 0, 0, rng

    def add(self, obs: np.ndarray, act: np.ndarray, rew: float, next_obs: np.ndarray,
            terminated: bool) -> None:
        i = self.ptr
        self.obs[i], self.act[i], self.rew[i] = obs, act, rew
        self.next_obs[i], self.term[i] = next_obs, float(terminated)
        self.ptr = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Batch:
        idx = self.rng.integers(0, self.size, size=batch_size)
        return Batch(*(torch.from_numpy(a[idx])
                       for a in (self.obs, self.act, self.rew, self.next_obs, self.term)))

    def __len__(self) -> int:
        return self.size
