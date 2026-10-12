"""V2 — the stochasticity negative control (docs/protocol.md §1.1, §4).

`tests/test_lunarlander_determinism.py` shows that a seeded replay is
bit-identical. That is reproducibility of the RNG stream, not a deterministic
MDP: from the *same* state s_t, re-seeding only the environment's RNG changes
the trajectory and hence the return-to-go.

So a stored $G_t$ is one sample of $Q^\\pi(s_t, a_t)$: unbiased but noisy. This
test is the negative control for anyone who would call it "exact".
"""

from __future__ import annotations

import numpy as np
from gymnasium.envs.box2d.lunar_lander import heuristic

from lunar_bias.envs import make_env
from lunar_bias.evaluation import returns_to_go

GAMMA = 0.99
SEED = 12345
FORK_STEP = 20  #: early enough that the heuristic controller is still flying
PROBE_HORIZON = 1_500
#: Re-seeding values used by the fork probe (scripts/probe_noise.py, X5).
NOISE_SEEDS = tuple(10_000_000 + k for k in range(5))


def _act(env, obs) -> np.ndarray:
    return np.asarray(heuristic(env, obs), dtype=np.float32)


def _rollout(seed: int, t_fork: int, noise_seed: int | None = None):
    """Replay the heuristic controller for `t_fork` steps, then, optionally,
    re-seed the transition noise before continuing to the horizon.

    Returns (observation at the fork, rewards collected after the fork).
    """
    env = make_env(PROBE_HORIZON)
    obs, _ = env.reset(seed=seed)
    for t in range(t_fork):
        obs, _, terminated, truncated, _ = env.step(_act(env, obs))
        if terminated or truncated:
            env.close()
            raise AssertionError(f"episode ended at step {t + 1} < fork step {t_fork}")
    obs_fork = np.array(obs, copy=True)

    if noise_seed is not None:
        # Only the transition noise changes; everything else is replayed.
        env.unwrapped.np_random = np.random.default_rng(noise_seed)

    rewards: list[float] = []
    terminated = truncated = False
    while not (terminated or truncated):
        obs, reward, terminated, truncated, _ = env.step(_act(env, obs))
        rewards.append(float(reward))
    env.close()
    return obs_fork, np.asarray(rewards, dtype=np.float64)


def test_v2_reseeding_the_transition_noise_changes_the_return_to_go():
    obs_fork, rewards_plain = _rollout(SEED, FORK_STEP)
    assert len(rewards_plain) > 0

    g_at_fork: list[float] = []
    for noise_seed in NOISE_SEEDS:
        obs_k, rewards_k = _rollout(SEED, FORK_STEP, noise_seed=noise_seed)
        # the replay lands exactly on s_t: only the noise differs
        np.testing.assert_array_equal(obs_k, obs_fork)
        g_at_fork.append(float(returns_to_go(rewards_k, GAMMA)[0]))

    g = np.asarray(g_at_fork, dtype=np.float64)
    assert len(np.unique(g)) >= 2, f"the transition noise did nothing: G = {g}"
    assert g.std(ddof=1) > 0.1, f"the spread of G is suspiciously small: {g}"

    print(f"\nV2: G_{{{FORK_STEP}}} over {len(NOISE_SEEDS)} re-seeded continuations: "
          f"min={g.min():.3f} max={g.max():.3f} std={g.std(ddof=1):.3f}")


def test_v2_without_reseeding_the_return_is_reproduced():
    """Control: the spread above comes from re-seeding, not from the replay."""
    _, rewards_a = _rollout(SEED, FORK_STEP)
    _, rewards_b = _rollout(SEED, FORK_STEP)
    np.testing.assert_array_equal(rewards_a, rewards_b)
