"""DDPG and TD3, with TD3 written as DDPG plus exactly three hooks.

Invariants (`docs/protocol.md` §5):
- I4: DDPG has a target actor *and* a target critic, both Polyak-updated after
  every gradient step;
- I5: TD3 = DDPG + (1) two critics regressed to ONE reduced target built from
  the two *target* critics, (2) target policy smoothing, (3) delayed actor and
  target updates;
- I8: probe Q-values come from the ONLINE critics, at the action the policy
  took. Row 0 of `q_values` is Q1, the critic the actor ascends.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from lunar_bias.buffer import Batch
from lunar_bias.config import AlgoConfig, Reduction, RunConfig
from lunar_bias.envs import ACT_DIM, OBS_DIM
from lunar_bias.networks import Actor, QNetwork


def reduce_q(qs: Tensor, mode: Reduction) -> Tensor:
    """Reduce stacked critic values [n_critics, B] -> [B]."""
    if mode == "min":
        return qs.min(dim=0).values
    if mode == "mean":
        return qs.mean(dim=0)
    if mode == "max":
        return qs.max(dim=0).values
    raise ValueError(f"unknown reduction {mode!r}")


def soft_update(src: nn.Module, dst: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for p, tp in zip(src.parameters(), dst.parameters(), strict=True):
            tp.mul_(1.0 - tau).add_(p, alpha=tau)


def grad_norm(module: nn.Module) -> float:
    sq = [p.grad.pow(2).sum() for p in module.parameters() if p.grad is not None]
    return float(torch.sqrt(torch.stack(sq).sum())) if sq else 0.0


class DDPGAgent:
    """DDPG with a target actor and a target critic (Lillicrap et al., 2016).

    TD3Agent overrides exactly three hooks: `target_value` (twin critics with a
    reduced target), `target_action` (target policy smoothing) and `actor_due`
    (delayed actor and target updates). Nothing else differs.
    """

    n_critics = 1

    def __init__(self, cfg: RunConfig, obs_dim: int = OBS_DIM, act_dim: int = ACT_DIM):
        self.cfg: AlgoConfig = cfg.algo
        self.actor = Actor(obs_dim, act_dim, cfg.actor)
        self.critics = nn.ModuleList(
            [QNetwork(obs_dim, act_dim, cfg.critic) for _ in range(self.n_critics)])
        self.actor_target = copy.deepcopy(self.actor).requires_grad_(False)
        self.critic_targets = copy.deepcopy(self.critics).requires_grad_(False)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.cfg.lr_actor)
        self.critic_opt = torch.optim.Adam(self.critics.parameters(), lr=self.cfg.lr_critic)
        self.n_updates = 0
        self.n_actor_updates = 0
        self.last_actor_loss = float("nan")

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> np.ndarray:
        """Deterministic action pi(s) in [-1, 1] (no exploration noise)."""
        x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        return self.actor(x).squeeze(0).numpy()

    @torch.no_grad()
    def q_values(self, obs: Tensor, act: Tensor) -> Tensor:
        """ONLINE critics at (obs, act): shape [n_critics, N]; row 0 is Q1."""
        return torch.stack([q(obs, act) for q in self.critics])

    # --- the three hooks TD3 overrides --------------------------------------
    def target_action(self, next_obs: Tensor) -> Tensor:
        return self.actor_target(next_obs)

    def target_value(self, next_obs: Tensor) -> Tensor:
        return self.critic_targets[0](next_obs, self.target_action(next_obs))

    def actor_due(self) -> bool:
        return True

    @torch.no_grad()
    def compute_target(self, b: Batch) -> Tensor:
        # only true termination stops the bootstrap; truncations were stored as term = 0
        return b.rew + self.cfg.gamma * (1.0 - b.term) * self.target_value(b.next_obs)

    def update(self, b: Batch) -> dict[str, float]:
        y = self.compute_target(b)
        qs = [q(b.obs, b.act) for q in self.critics]
        # every critic regresses to the same target
        critic_loss = sum(F.mse_loss(q, y) for q in qs)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        stats = {
            "critic_loss": float(critic_loss),
            "q1_batch_mean": float(qs[0].mean()),
            "q_abs_max": max(float(q.abs().max()) for q in qs),
            "critic_grad_norm": grad_norm(self.critics),
        }
        self.critic_opt.step()
        self.n_updates += 1
        if self.actor_due():
            # the actor ascends Q1
            actor_loss = -self.critics[0](b.obs, self.actor(b.obs)).mean()
            self.actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_opt.step()
            self.last_actor_loss = float(actor_loss)
            self.n_actor_updates += 1
            soft_update(self.actor, self.actor_target, self.cfg.tau)
            soft_update(self.critics, self.critic_targets, self.cfg.tau)
        # the previous value on delayed steps, not 0.0
        stats["actor_loss"] = self.last_actor_loss
        return stats

    def state_dict(self) -> dict[str, Any]:
        return {"actor": self.actor.state_dict(), "critics": self.critics.state_dict()}

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        self.actor.load_state_dict(sd["actor"])
        self.critics.load_state_dict(sd["critics"])


class TD3Agent(DDPGAgent):
    """TD3 = DDPG + (1) twin critics with one reduced target, (2) target policy
    smoothing, (3) delayed actor and target updates."""

    n_critics = 2

    def target_action(self, next_obs: Tensor) -> Tensor:  # (2)
        a = self.actor_target(next_obs)
        noise = (torch.randn_like(a) * self.cfg.target_noise).clamp(
            -self.cfg.target_noise_clip, self.cfg.target_noise_clip)
        return (a + noise).clamp(-1.0, 1.0)

    def target_value(self, next_obs: Tensor) -> Tensor:  # (1)
        a = self.target_action(next_obs)
        qs = torch.stack([qt(next_obs, a) for qt in self.critic_targets])
        return reduce_q(qs, self.cfg.target_reduction)

    def actor_due(self) -> bool:  # (3)
        return self.n_updates % self.cfg.policy_delay == 0


def make_agent(cfg: RunConfig) -> DDPGAgent:
    return {"ddpg": DDPGAgent, "td3": TD3Agent}[cfg.algo.name](cfg)
