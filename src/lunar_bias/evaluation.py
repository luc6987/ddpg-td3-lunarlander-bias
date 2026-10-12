"""Evaluation *is* the bias probe: one pass, on the 20 fixed seeds.

Per checkpoint the deterministic policy runs the 20 evaluation seeds, each
episode starting with `reset(seed=seed)`, for up to `probe_horizon` steps
(invariant I7). Performance uses only the first `train_horizon` steps; the
probe stores every point $t < \\min(T, train\\_horizon)$ together with
$G_t$ computed over the whole rollout.

$G_t$ is ONE sample of $Q^\\pi(s_t, a_t)$: LunarLander's transitions are
stochastic (two `dispersion` uniforms per step), so it is neither exact nor
noise-free (`docs/protocol.md` §1.1, §1.3). Raw data only — every statistic is
computed afterwards in `analysis/`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import gymnasium as gym
import numpy as np
import torch
from torch import Tensor

from lunar_bias.envs import make_env

#: "terminated" is for non-LunarLander test environments, which have no lander.
OUTCOMES = ("landed", "crashed", "timeout", "terminated")


class PolicyAgent(Protocol):
    """What the probe needs from an agent: a deterministic policy and Q values."""

    def act(self, obs: np.ndarray) -> np.ndarray: ...

    def q_values(self, obs: Tensor, act: Tensor) -> Tensor: ...


def returns_to_go(rewards: np.ndarray, gamma: float) -> np.ndarray:
    """G[t] = sum_k gamma**k * rewards[t + k], float64, backward recursion."""
    out = np.empty(len(rewards), dtype=np.float64)
    acc = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        acc = float(rewards[t]) + gamma * acc
        out[t] = acc
    return out


def _outcome(env: gym.Env, terminated: bool, length: int, train_horizon: int) -> str:
    if not terminated or length > train_horizon:
        return "timeout"  # a standard 1 000-step episode would have been truncated
    lander = getattr(env.unwrapped, "lander", None)
    if lander is None:
        return "terminated"
    # mirrors LunarLander.step's +100 for a landing and -100 for a crash
    return "landed" if not lander.awake else "crashed"


@dataclass
class ProbeData:
    episodes: list[dict[str, Any]]  # rows of eval_episodes.csv (without `step`)
    arrays: dict[str, np.ndarray]   # contents of probe/step_XXXXXXX.npz


def evaluate_and_probe(agent: PolicyAgent, seeds: Sequence[int], gamma: float,
                       train_horizon: int, probe_horizon: int,
                       env_fn: Callable[[int], gym.Env] = make_env) -> ProbeData:
    """Run the deterministic policy once per fixed seed.

    Returns performance rows (first `train_horizon` steps) and raw probe points
    (every t < min(T, train_horizon)), with G_t computed over the whole rollout,
    which is allowed to run to `probe_horizon`.
    """
    env = env_fn(probe_horizon)
    rows: list[dict[str, Any]] = []
    obs_pts, act_pts, ep_idx, t_idx, g_pts = [], [], [], [], []
    ep_len, ep_term, ep_g_h = [], [], []
    for i, seed in enumerate(seeds):
        obs, _ = env.reset(seed=int(seed))
        observations, actions, rewards = [], [], []
        terminated = truncated = False
        while not (terminated or truncated):
            action = agent.act(obs)
            observations.append(obs)
            actions.append(action)
            obs, reward, terminated, truncated, _ = env.step(action)
            rewards.append(float(reward))
        length = len(rewards)
        r = np.asarray(rewards, dtype=np.float64)
        g = returns_to_go(r, gamma)
        n = min(length, train_horizon)
        rows.append({
            "episode_index": i,
            "seed": int(seed),
            "return_1000": float(r[:train_horizon].sum()),
            "length_1000": n,
            "outcome": _outcome(env, terminated, length, train_horizon),
        })
        obs_pts.append(np.asarray(observations[:n], dtype=np.float32))
        act_pts.append(np.asarray(actions[:n], dtype=np.float32))
        ep_idx.append(np.full(n, i, dtype=np.int16))
        t_idx.append(np.arange(n, dtype=np.int16))
        g_pts.append(g[:n])
        ep_len.append(length)
        ep_term.append(bool(terminated))
        ep_g_h.append(g[train_horizon] if length > train_horizon else 0.0)
    env.close()
    q = agent.q_values(torch.from_numpy(np.concatenate(obs_pts)),
                       torch.from_numpy(np.concatenate(act_pts))).numpy()
    arrays = {
        "episode": np.concatenate(ep_idx),
        "t": np.concatenate(t_idx),
        "G": np.concatenate(g_pts),  # float64
        "q": q.astype(np.float32),  # [n_critics, N]; row 0 = Q1
        "ep_seed": np.asarray(seeds, dtype=np.int64),
        "ep_len": np.asarray(ep_len, dtype=np.int32),
        "ep_terminated": np.asarray(ep_term, dtype=bool),
        "ep_return_1000": np.asarray([x["return_1000"] for x in rows], dtype=np.float64),
        "ep_outcome": np.asarray([OUTCOMES.index(x["outcome"]) for x in rows],
                                 dtype=np.int8),
        "ep_G_at_train_horizon": np.asarray(ep_g_h, dtype=np.float64),
        "gamma": np.float64(gamma),
        "train_horizon": np.int32(train_horizon),
        "probe_horizon": np.int32(probe_horizon),
    }
    return ProbeData(rows, arrays)
