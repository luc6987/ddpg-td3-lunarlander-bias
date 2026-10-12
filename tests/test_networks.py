"""`networks.py`: LN only where the protocol puts it (§5), and usable shapes.

The LN treatment is Linear -> LayerNorm -> ReLU for each hidden layer. No
LayerNorm on the output layer (it would make the output constant), none in the
actor.
"""

from __future__ import annotations

import torch
from torch import nn

from lunar_bias.config import NetConfig
from lunar_bias.envs import ACT_DIM, OBS_DIM
from lunar_bias.networks import Actor, QNetwork, mlp


def _kinds(module: nn.Sequential) -> list[type[nn.Module]]:
    return [type(m) for m in module]


def test_mlp_module_order_with_layer_norm():
    net = mlp(8, (64, 32), 1, "layer")
    assert _kinds(net) == [
        nn.Linear, nn.LayerNorm, nn.ReLU,
        nn.Linear, nn.LayerNorm, nn.ReLU,
        nn.Linear,
    ]
    assert net[1].normalized_shape == (64,)
    assert net[4].normalized_shape == (32,)


def test_mlp_module_order_without_norm():
    net = mlp(8, (64, 32), 1, "none")
    assert _kinds(net) == [nn.Linear, nn.ReLU, nn.Linear, nn.ReLU, nn.Linear]
    assert not any(isinstance(m, nn.LayerNorm) for m in net)


def test_no_layer_norm_after_the_last_linear():
    """A LayerNorm over a 1-d output would make it constant."""
    for norm in ("none", "layer"):
        net = mlp(8, (64, 64), 1, norm, out_act=nn.Tanh())
        linears = [i for i, m in enumerate(net) if isinstance(m, nn.Linear)]
        norms = [i for i, m in enumerate(net) if isinstance(m, nn.LayerNorm)]
        assert max(norms, default=-1) < max(linears)
        assert isinstance(net[-1], nn.Tanh)
        assert isinstance(net[-2], nn.Linear)


def test_mlp_shapes_and_out_act():
    x = torch.zeros(5, 8)
    assert mlp(8, (64, 64), 1, "none")(x).shape == (5, 1)
    assert mlp(8, (64, 64), 2, "layer")(x).shape == (5, 2)
    assert mlp(8, (16,), 3, "none", out_act=nn.Tanh())(x).shape == (5, 3)


def test_qnetwork_shapes():
    q = QNetwork(OBS_DIM, ACT_DIM, NetConfig())
    out = q(torch.zeros(7, OBS_DIM), torch.zeros(7, ACT_DIM))
    assert out.shape == (7,)
    assert out.dtype == torch.float32


def test_actor_output_is_bounded():
    actor = Actor(OBS_DIM, ACT_DIM, NetConfig())
    obs = torch.randn(256, OBS_DIM) * 50.0  # far outside the observation box
    out = actor(obs)
    assert out.shape == (256, ACT_DIM)
    assert torch.all(out >= -1.0) and torch.all(out <= 1.0)


def test_every_parameter_gets_a_gradient():
    q = QNetwork(OBS_DIM, ACT_DIM, NetConfig(norm="layer"))
    actor = Actor(OBS_DIM, ACT_DIM, NetConfig())
    obs, act = torch.randn(16, OBS_DIM), torch.randn(16, ACT_DIM)
    (q(obs, act).pow(2).sum() + actor(obs).pow(2).sum()).backward()
    for name, module in (("q", q), ("actor", actor)):
        for pname, p in module.named_parameters():
            assert p.grad is not None, f"{name}.{pname} has no gradient"
            assert torch.isfinite(p.grad).all(), f"{name}.{pname} has a non-finite gradient"


def test_defaults_match_the_protocol():
    cfg = NetConfig()
    assert cfg.hidden == (64, 64)
    assert cfg.norm == "none"
