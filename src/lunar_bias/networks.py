"""Actor and critic MLPs, with the study's only architectural knob: critic LN.

`norm="layer"` inserts a `LayerNorm` after every hidden `Linear` and before the
ReLU. It is never used on the output layer (a LayerNorm over a 1-d output would
make it constant) and never in the actor (`docs/protocol.md` §5).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from lunar_bias.config import NetConfig, Norm


def mlp(in_dim: int, hidden: Sequence[int], out_dim: int, norm: Norm = "none",
        out_act: nn.Module | None = None) -> nn.Sequential:
    """Linear -> [LayerNorm] -> ReLU per hidden layer, then a plain Linear output."""
    layers: list[nn.Module] = []
    d = in_dim
    for h in hidden:
        layers.append(nn.Linear(d, h))
        if norm == "layer":
            layers.append(nn.LayerNorm(h))
        layers.append(nn.ReLU())
        d = h
    layers.append(nn.Linear(d, out_dim))
    if out_act is not None:
        layers.append(out_act)
    return nn.Sequential(*layers)


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, net: NetConfig):
        super().__init__()
        self.net = mlp(obs_dim + act_dim, net.hidden, 1, net.norm)

    def forward(self, obs: Tensor, act: Tensor) -> Tensor:  # [B, o], [B, a] -> [B]
        return self.net(torch.cat([obs, act], dim=-1)).squeeze(-1)


class Actor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, net: NetConfig):
        super().__init__()
        self.net = mlp(obs_dim, net.hidden, act_dim, net.norm, out_act=nn.Tanh())

    def forward(self, obs: Tensor) -> Tensor:  # [B, o] -> [B, a] in [-1, 1]
        return self.net(obs)
