"""`agents.py`: the TD3 mechanisms and the DDPG baseline (invariants I4, I5, I8).

Every hook TD3 overrides is checked directly, plus the target computation of
DDPG, which must bootstrap through a time-limit truncation (I1).
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

from lunar_bias.agents import DDPGAgent, TD3Agent, make_agent, reduce_q, soft_update
from lunar_bias.buffer import Batch
from lunar_bias.config import AlgoConfig, RunConfig
from lunar_bias.envs import ACT_DIM, OBS_DIM

B = 16


def _cfg(algo: str, **kwargs: object) -> RunConfig:
    base = RunConfig()
    return dataclasses.replace(
        base, algo=dataclasses.replace(AlgoConfig(name=algo), **kwargs))


def _batch(rng: torch.Generator) -> Batch:
    return Batch(
        obs=torch.randn(B, OBS_DIM, generator=rng),
        act=torch.rand(B, ACT_DIM, generator=rng) * 2 - 1,
        rew=torch.randn(B, generator=rng),
        next_obs=torch.randn(B, OBS_DIM, generator=rng),
        term=(torch.rand(B, generator=rng) < 0.2).float(),
    )


@pytest.fixture
def rng() -> torch.Generator:
    return torch.Generator().manual_seed(0)


@pytest.fixture
def batch(rng: torch.Generator) -> Batch:
    return _batch(rng)


# --- DDPG ------------------------------------------------------------------


def test_ddpg_n_critics_and_make_agent():
    assert DDPGAgent.n_critics == 1
    assert TD3Agent.n_critics == 2
    assert isinstance(make_agent(_cfg("ddpg")), DDPGAgent)
    assert isinstance(make_agent(_cfg("td3")), TD3Agent)


def test_ddpg_has_a_target_actor_and_a_target_critic():
    agent = DDPGAgent(_cfg("ddpg"))
    assert isinstance(agent.actor_target, type(agent.actor))
    assert len(agent.critic_targets) == 1
    obs = torch.randn(4, OBS_DIM, generator=torch.Generator().manual_seed(1))
    assert torch.equal(agent.actor_target(obs), agent.actor(obs)), "targets start as a copy"
    assert all(not p.requires_grad for p in agent.actor_target.parameters())
    assert all(not p.requires_grad for p in agent.critic_targets.parameters())


def test_ddpg_target_equals_r_plus_gamma_one_minus_term_q_prime(batch: Batch):
    """y = r + gamma (1 - d) Q'(s', mu'(s')) — the Bellman target, I4."""
    cfg = _cfg("ddpg")
    agent = DDPGAgent(cfg)
    y = agent.compute_target(batch)
    with torch.no_grad():
        q_prime = agent.critic_targets[0](batch.next_obs,
                                          agent.actor_target(batch.next_obs))
        expected = batch.rew + cfg.algo.gamma * (1.0 - batch.term) * q_prime
    torch.testing.assert_close(y, expected)


def test_termination_stops_the_bootstrap(batch: Batch):
    """term = 1 => y = r; a truncation (term = 0) keeps bootstrapping."""
    agent = DDPGAgent(_cfg("ddpg"))
    done = batch._replace(term=torch.ones(B))
    torch.testing.assert_close(agent.compute_target(done), batch.rew)

    truncated = batch._replace(term=torch.zeros(B))
    y = agent.compute_target(truncated)
    assert not torch.allclose(y, batch.rew), "a truncation must still bootstrap"


def test_q_values_come_from_the_online_critics(batch: Batch):
    """Shape [n_critics, N], row 0 = Q1 (I8), and no target network is used."""
    ddpg = DDPGAgent(_cfg("ddpg"))
    td3 = TD3Agent(_cfg("td3"))
    assert ddpg.q_values(batch.obs, batch.act).shape == (1, B)
    qs = td3.q_values(batch.obs, batch.act)
    assert qs.shape == (2, B)
    with torch.no_grad():
        torch.testing.assert_close(qs[0], td3.critics[0](batch.obs, batch.act))
        torch.testing.assert_close(qs[1], td3.critics[1](batch.obs, batch.act))

    td3.update(batch)  # moves the online critics; n_updates = 1 is not an actor step
    with torch.no_grad():
        qs = td3.q_values(batch.obs, batch.act)
        q_target = td3.critic_targets[0](batch.obs, batch.act)
        torch.testing.assert_close(qs[0], td3.critics[0](batch.obs, batch.act))
    assert not torch.allclose(qs[0], q_target), "q_values must not read the target critics"


def test_actor_ascends_q1(batch: Batch):
    """Actor loss = -Q1(s, pi(s)) (TD3 standard, I8)."""
    # lr = 0 freezes both networks, so the recorded loss is comparable to a manual one
    agent = TD3Agent(_cfg("td3", lr_actor=0.0, lr_critic=0.0))
    with torch.no_grad():
        expected = -float(agent.critics[0](batch.obs, agent.actor(batch.obs)).mean())
    agent.update(batch)  # n_updates = 1: not an actor step, so the loss stays NaN
    assert agent.last_actor_loss != agent.last_actor_loss
    agent.update(batch)  # n_updates = 2: actor step
    assert agent.last_actor_loss == pytest.approx(expected, rel=1e-6)


def test_a_gradient_step_increases_q1_at_the_policy(batch: Batch):
    """The actor maximises Q1: one actor step must raise Q1(s, pi(s))."""
    agent = TD3Agent(_cfg("td3", lr_critic=0.0))  # the critic cannot confound the test
    with torch.no_grad():
        before = float(agent.critics[0](batch.obs, agent.actor(batch.obs)).mean())
    agent.update(batch)
    agent.update(batch)  # the second one is an actor step
    with torch.no_grad():
        after = float(agent.critics[0](batch.obs, agent.actor(batch.obs)).mean())
    assert after > before


def test_act_is_deterministic_and_bounded():
    agent = DDPGAgent(_cfg("ddpg"))
    obs = torch.randn(OBS_DIM, generator=torch.Generator().manual_seed(2)).numpy()
    a1, a2 = agent.act(obs), agent.act(obs)
    assert a1.shape == (ACT_DIM,)
    assert a1.dtype.name == "float32"
    torch.testing.assert_close(torch.from_numpy(a1), torch.from_numpy(a2))
    assert (a1 >= -1.0).all() and (a1 <= 1.0).all()


# --- the three TD3 mechanisms ----------------------------------------------


@pytest.mark.parametrize("reduction", ["min", "mean", "max"])
def test_td3_target_without_noise_is_the_reduction_of_the_target_critics(
        reduction: str, batch: Batch, rng: torch.Generator):
    agent = TD3Agent(_cfg("td3", target_noise=0.0, target_reduction=reduction))
    with torch.no_grad():
        a = agent.actor_target(batch.next_obs)
        qs = torch.stack([qt(batch.next_obs, a) for qt in agent.critic_targets])
        expected_target_value = reduce_q(qs, reduction)  # type: ignore[arg-type]
        expected = batch.rew + agent.cfg.gamma * (1.0 - batch.term) * expected_target_value
    torch.testing.assert_close(agent.compute_target(batch), expected)
    # the two target critics really differ, so min/mean/max are distinguishable
    assert not torch.allclose(qs[0], qs[1])


def test_td3_reductions_are_ordered_pointwise():
    qs = torch.tensor([[1.0, 5.0, -2.0], [3.0, 4.0, -1.0]])
    torch.testing.assert_close(reduce_q(qs, "min"), torch.tensor([1.0, 4.0, -2.0]))
    torch.testing.assert_close(reduce_q(qs, "mean"), torch.tensor([2.0, 4.5, -1.5]))
    torch.testing.assert_close(reduce_q(qs, "max"), torch.tensor([3.0, 5.0, -1.0]))
    with pytest.raises(ValueError, match="unknown reduction"):
        reduce_q(qs, "median")  # type: ignore[arg-type]


def test_target_action_is_smoothed_and_clipped(batch: Batch):
    """(2) target policy smoothing: noise in +-target_noise_clip, action in [-1, 1]."""
    clip = 0.25
    agent = TD3Agent(_cfg("td3", target_noise=1e6, target_noise_clip=clip))
    with torch.no_grad():
        a = agent.actor_target(batch.next_obs)
        noisy = agent.target_action(batch.next_obs)
    assert noisy.shape == a.shape
    assert torch.all(noisy >= -1.0) and torch.all(noisy <= 1.0)
    # a huge sigma with a tiny clip saturates the clamp: |noisy - a| == clip
    # wherever a +- clip is still inside the box, and is smaller at the box edge
    dev = (noisy - a).abs()
    assert float(dev.max()) <= clip + 1e-6, "the noise was not clipped"
    assert float(dev.max()) > clip - 1e-3, "the noise was not applied"


def test_target_action_without_noise_is_the_target_actor(batch: Batch):
    agent = TD3Agent(_cfg("td3", target_noise=0.0))
    with torch.no_grad():
        torch.testing.assert_close(
            agent.target_action(batch.next_obs), agent.actor_target(batch.next_obs))


def test_td3_delays_the_actor_and_the_targets(batch: Batch):
    """(3) policy_delay = 2: 6 gradient steps => 3 actor steps; DDPG does 6."""
    td3 = TD3Agent(_cfg("td3"))
    for _ in range(6):
        td3.update(batch)
    assert td3.n_updates == 6
    assert td3.n_actor_updates == 3

    ddpg = DDPGAgent(_cfg("ddpg"))
    for _ in range(6):
        ddpg.update(batch)
    assert ddpg.n_updates == 6
    assert ddpg.n_actor_updates == 6


def test_td3_targets_are_unchanged_by_a_non_actor_update(batch: Batch):
    agent = TD3Agent(_cfg("td3"))
    before_actor = [p.detach().clone() for p in agent.actor_target.parameters()]
    before_critics = [p.detach().clone() for p in agent.critic_targets.parameters()]

    agent.update(batch)  # n_updates = 1: delayed, so nothing but the online critics moves
    assert agent.n_actor_updates == 0
    for p, q in zip(agent.actor_target.parameters(), before_actor, strict=True):
        assert torch.equal(p, q), "the target actor moved on a delayed step"
    for p, q in zip(agent.critic_targets.parameters(), before_critics, strict=True):
        assert torch.equal(p, q), "the target critics moved on a delayed step"

    agent.update(batch)  # n_updates = 2: the actor and both targets move
    assert agent.n_actor_updates == 1
    assert any(not torch.equal(p, q) for p, q in
               zip(agent.actor_target.parameters(), before_actor, strict=True))
    assert any(not torch.equal(p, q) for p, q in
               zip(agent.critic_targets.parameters(), before_critics, strict=True))


def test_ddpg_updates_its_targets_on_every_step(batch: Batch):
    agent = DDPGAgent(_cfg("ddpg"))
    agent.update(batch)
    before = [p.detach().clone() for p in agent.actor_target.parameters()]
    agent.update(batch)
    assert any(not torch.equal(p, q) for p, q in
               zip(agent.actor_target.parameters(), before, strict=True))


# --- soft_update -----------------------------------------------------------


def test_soft_update_tau_one_copies_and_tau_zero_keeps():
    src = DDPGAgent(_cfg("ddpg")).actor
    dst = DDPGAgent(_cfg("ddpg")).actor
    with torch.no_grad():
        for p in src.parameters():
            p.add_(1.0)

    keep = [p.detach().clone() for p in dst.parameters()]
    soft_update(src, dst, 0.0)
    for p, q in zip(dst.parameters(), keep, strict=True):
        torch.testing.assert_close(p, q)

    soft_update(src, dst, 1.0)
    for p, q in zip(dst.parameters(), src.parameters(), strict=True):
        torch.testing.assert_close(p, q)

    # tau = 0.5 interpolates
    with torch.no_grad():
        for p in src.parameters():
            p.add_(2.0)
    half = [p.detach().clone() for p in dst.parameters()]
    soft_update(src, dst, 0.5)
    for p, q, s in zip(dst.parameters(), half, src.parameters(), strict=True):
        torch.testing.assert_close(p, 0.5 * q + 0.5 * s)


def test_soft_update_rejects_mismatched_modules():
    """A silently skipped parameter pair would corrupt the target networks."""
    with pytest.raises(RuntimeError):  # same count, different shapes
        soft_update(torch.nn.Linear(2, 2), torch.nn.Linear(3, 3), 0.1)
    with pytest.raises(ValueError):  # different parameter counts: strict zip
        soft_update(torch.nn.Sequential(torch.nn.Linear(2, 2)),
                    torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)), 0.1)


# --- misc ------------------------------------------------------------------


def test_state_dict_round_trip():
    agent = TD3Agent(_cfg("td3"))
    other = TD3Agent(_cfg("td3"))
    other.load_state_dict(agent.state_dict())
    obs = torch.randn(3, OBS_DIM, generator=torch.Generator().manual_seed(3))
    torch.testing.assert_close(agent.actor(obs), other.actor(obs))


def test_update_returns_finite_stats(batch: Batch):
    agent = TD3Agent(_cfg("td3"))
    stats = agent.update(batch)
    assert set(stats) == {
        "critic_loss", "q1_batch_mean", "q_abs_max", "critic_grad_norm", "actor_loss"}
    assert all(isinstance(v, float) for v in stats.values())
    assert stats["critic_grad_norm"] > 0.0
