"""Validate the assumption the whole bias estimator rests on.

The Monte-Carlo estimator in `docs/protocol.md` treats a single deterministic
rollout as an *exact* return-to-go, and forks trajectories by re-seeding and
replaying. Both are only legitimate if a seeded `LunarLander-v3` is
deterministic. These tests check that directly rather than assuming it.

What is actually random in `LunarLander.reset`, and therefore what has to be
reproduced by a replay:

* the terrain profile, `height = np_random.uniform(0, H / 2, size=(CHUNKS + 1,))`
  — the lander lands on this, so it materially changes the return;
* the initial random impulse applied to the lander.

Both are drawn from `self.np_random`, which `reset(seed=...)` re-seeds, so a
replay rebuilds them. Note also that `reset` ends by taking one step with a
zero action, so the observation it returns is already $s_1$.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

ENV_ID = "LunarLander-v3"
ENV_KWARGS = {"continuous": True, "enable_wind": False}
SEED = 12345
N_STEPS = 400
#: Points at which trajectories are forked. Kept well below the episode length
#: the fixture actions produce (the lander survives ~70 steps before crashing).
FORK_POINTS = (10, 30, 60)


def _make_env():
    return gym.make(ENV_ID, **ENV_KWARGS)


def _rollout(actions: np.ndarray, seed: int = SEED):
    """Replay `actions` from a freshly seeded environment, recording everything
    until the episode ends (or the actions run out)."""
    env = _make_env()
    obs, _ = env.reset(seed=seed)
    obs_list, reward_list = [obs], []
    for a in actions:
        obs, r, terminated, truncated, _ = env.step(a)
        obs_list.append(obs)
        reward_list.append(r)
        if terminated or truncated:
            break
    env.close()
    return np.stack(obs_list), np.asarray(reward_list)


@pytest.fixture(scope="module")
def actions() -> np.ndarray:
    """A fixed, non-trivial action sequence.

    Not all zeros: it has to exercise the dynamics, including the leg-contact
    regime. Small enough that the lander usually survives past `FORK_POINTS`,
    which the fork test needs.
    """
    rng = np.random.default_rng(0)
    return (0.4 * rng.uniform(-1.0, 1.0, size=(N_STEPS, 2))).astype(np.float32)


def test_reset_is_reproducible():
    """Same seed => bit-identical initial observation."""
    env = _make_env()
    obs_a, _ = env.reset(seed=SEED)
    obs_b, _ = env.reset(seed=SEED)
    env.close()
    np.testing.assert_array_equal(obs_a, obs_b)


def test_randomised_terrain_is_reproducible():
    """The terrain profile (the genuinely random part of a reset) must be
    rebuilt identically, since the lander lands on it."""
    env = _make_env()
    env.reset(seed=SEED)
    terrain_a = np.asarray([tuple(p) for p in env.unwrapped.sky_polys], dtype=np.float64)
    env.reset(seed=SEED)
    terrain_b = np.asarray([tuple(p) for p in env.unwrapped.sky_polys], dtype=np.float64)
    env.close()
    np.testing.assert_array_equal(terrain_a, terrain_b)


def test_replay_is_bit_identical(actions):
    """The core assumption: replaying the same actions from the same seed
    reproduces the trajectory exactly, step by step."""
    obs_a, rew_a = _rollout(actions)
    obs_b, rew_b = _rollout(actions)
    np.testing.assert_array_equal(obs_a, obs_b)
    np.testing.assert_array_equal(rew_a, rew_b)


def test_fork_lands_on_the_same_state(actions):
    """Replaying the first `t` actions from the same seed must land exactly on
    $s_t$ of the original trajectory.

    This is the property the fork probe relies on: it is what lets us evaluate
    $Q(s_t, a')$ for an action $a'$ the policy would *not* have taken, without
    touching Box2D internals.
    """
    obs_full, rew_full = _rollout(actions)
    assert len(rew_full) > max(FORK_POINTS), (
        f"the fixture actions end the episode after only {len(rew_full)} steps, "
        f"so the fork points {FORK_POINTS} are unreachable"
    )

    for t in FORK_POINTS:
        obs_fork, rew_fork = _rollout(actions[:t])
        assert len(rew_fork) == t, "the fork ran past the fork point"
        np.testing.assert_array_equal(obs_fork[t], obs_full[t])
        np.testing.assert_array_equal(rew_fork, rew_full[:t])


def test_fork_reproduces_the_return_to_go(actions):
    """The property the estimator actually uses.

    Replay the prefix in one environment, then continue with the original
    remaining actions *in the same environment* — the sequence of rewards must
    match the original trajectory's, and so must the return-to-go at the fork
    point. Note that the continuation is essential: a fork that stops at `t`
    has no tail, and $G_t$ sums over the whole tail.
    """
    gamma = 0.99
    _, rew_full = _rollout(actions)

    def return_to_go(rewards: np.ndarray, t: int) -> float:
        return float(sum(gamma**k * r for k, r in enumerate(rewards[t:])))

    for t in FORK_POINTS:
        env = _make_env()
        obs, _ = env.reset(seed=SEED)
        rewards: list[float] = []
        # phase 1: replay the stored prefix (this is the "fork")
        for a in actions[:t]:
            obs, r, terminated, truncated, _ = env.step(a)
            rewards.append(r)
            if terminated or truncated:
                break
        assert not (terminated or truncated), f"episode ended before fork point {t}"
        assert len(rewards) == t
        # phase 2: continue with the original tail
        for a in actions[t:]:
            obs, r, terminated, truncated, _ = env.step(a)
            rewards.append(r)
            if terminated or truncated:
                break
        env.close()

        rew_fork = np.asarray(rewards)
        np.testing.assert_array_equal(rew_fork, rew_full[: len(rew_fork)])
        np.testing.assert_allclose(
            return_to_go(rew_fork, t), return_to_go(rew_full, t), rtol=1e-12, atol=1e-12
        )


def test_determinism_is_not_trivial(actions):
    """Guard against a vacuous pass: a different seed must give a different
    trajectory, otherwise the tests above would hold for a constant env."""
    obs_a, _ = _rollout(actions, seed=SEED)
    obs_b, _ = _rollout(actions, seed=SEED + 1)
    assert not np.array_equal(obs_a, obs_b)
