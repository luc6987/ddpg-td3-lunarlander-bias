"""The training loop: seeding, uniform warm-up, UTD updates, evaluation.

Invariants (`docs/protocol.md` §5, `docs/implementation_plan.md` §9):
- I1: the buffer stores `terminated` only, so the target bootstraps through a
  time-limit truncation;
- I2: steps 1..learning_starts use uniform random actions in [-1, 1]^2, and the
  first gradient update happens at step == learning_starts;
- I3: after warm-up, action = clip(pi(s) + N(0, 0.1^2), -1, 1), with UTD
  gradient steps per environment step on a single environment;
- I10: one torch thread (set here, so a worker cannot change it);
- I11: a non-finite critic loss raises `Diverged`, which is a result, not a bug.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch

from lunar_bias.agents import make_agent
from lunar_bias.buffer import ReplayBuffer
from lunar_bias.config import RunConfig
from lunar_bias.envs import ACT_DIM, OBS_DIM, make_env

if TYPE_CHECKING:  # avoid a circular import: io.py imports this module's types
    from lunar_bias.io import RunWriter


class Diverged(Exception):
    """A non-finite critic loss: the run stops and is reported as `diverged`."""

    def __init__(self, step: int):
        super().__init__(f"non-finite critic loss at step {step}")
        self.step = step


def _nanmean(values: Sequence[float]) -> float:
    """Mean ignoring NaNs, or NaN when there is nothing to average."""
    if not values:
        return float("nan")
    with warnings.catch_warnings():  # an all-NaN window is expected, not a problem
        warnings.simplefilter("ignore", RuntimeWarning)
        return float(np.nanmean(values))


class Window:
    """Running window of the training loop's statistics, flushed at `log_every`."""

    def __init__(self) -> None:
        self._reset()

    def _reset(self) -> None:
        self._critic_loss: list[float] = []
        self._actor_loss: list[float] = []
        self._q1_batch_mean: list[float] = []
        self._q_abs_max: list[float] = []
        self._critic_grad_norm: list[float] = []
        self._episodes: list[float] = []

    def add_update(self, stats: Mapping[str, float]) -> None:
        self._critic_loss.append(stats["critic_loss"])
        self._actor_loss.append(stats["actor_loss"])  # NaN until the first actor step
        self._q1_batch_mean.append(stats["q1_batch_mean"])
        self._q_abs_max.append(stats["q_abs_max"])
        self._critic_grad_norm.append(stats["critic_grad_norm"])

    def add_episode(self, ret: float) -> None:
        self._episodes.append(float(ret))

    def flush(self) -> dict[str, float]:
        """Return the window's statistics and start a new window.

        Means are NaN when no update (and no episode) happened in the window;
        `train_episodes` is a count, so it is 0 then.
        """
        out = {
            "critic_loss": _nanmean(self._critic_loss),
            "actor_loss": _nanmean(self._actor_loss),
            "q1_batch_mean": _nanmean(self._q1_batch_mean),
            "q_abs_max": max(self._q_abs_max) if self._q_abs_max else float("nan"),
            "critic_grad_norm": _nanmean(self._critic_grad_norm),
            "train_episodes": len(self._episodes),
            "train_return_mean": _nanmean(self._episodes),
        }
        self._reset()
        return out


def train(cfg: RunConfig, out: RunWriter) -> None:
    torch.set_num_threads(1)  # I10: one thread per process, whatever the caller did
    torch.manual_seed(cfg.seed)  # network init + TD3 target noise
    act_rng, buf_rng = (np.random.default_rng(s)
                        for s in np.random.SeedSequence(cfg.seed).spawn(2))
    agent = make_agent(cfg)
    buffer = ReplayBuffer(cfg.algo.buffer_size, OBS_DIM, ACT_DIM, buf_rng)
    env = make_env(cfg.evaluation.train_horizon)
    obs, _ = env.reset(seed=cfg.seed)  # later resets continue the env's own RNG stream
    window = Window()  # running means for metrics.csv
    ep_return = 0.0
    out.evaluation(0, agent)
    for step in range(1, cfg.max_steps + 1):
        if step <= cfg.algo.learning_starts:
            action = act_rng.uniform(-1.0, 1.0, ACT_DIM).astype(np.float32)
        else:
            noise = cfg.algo.action_noise * act_rng.standard_normal(ACT_DIM)
            action = np.clip(agent.act(obs) + noise, -1.0, 1.0).astype(np.float32)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        # truncation is NOT terminal: only `terminated` is stored (I1)
        buffer.add(obs, action, float(reward), next_obs, terminated)
        ep_return += float(reward)
        obs = next_obs
        if terminated or truncated:
            window.add_episode(ep_return)
            ep_return = 0.0
            obs, _ = env.reset()
        if step >= cfg.algo.learning_starts:
            for _ in range(cfg.algo.utd):
                stats = agent.update(buffer.sample(cfg.algo.batch_size))
                if not math.isfinite(stats["critic_loss"]):
                    raise Diverged(step)
                window.add_update(stats)
        if step % cfg.log_every == 0:
            out.metrics(step, agent, window.flush())
        if step % cfg.evaluation.every == 0:
            out.evaluation(step, agent)
        if step % cfg.evaluation.checkpoint_every == 0 or step == cfg.max_steps:
            out.checkpoint(step, agent)
    env.close()
