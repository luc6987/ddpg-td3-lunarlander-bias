"""The single environment: continuous LunarLander, no wind.

One environment per process, so that UTD (gradient steps per environment step)
is exact (`docs/implementation_plan.md` §4). Never `gymnasium.vector`.
"""

from __future__ import annotations

import gymnasium as gym

ENV_ID = "LunarLander-v3"
OBS_DIM = 8
ACT_DIM = 2


def make_env(max_episode_steps: int) -> gym.Env:
    """Continuous LunarLander, no wind, with the given time limit.

    Transitions are stochastic: two `dispersion` uniforms are drawn from the env RNG
    at every step. reset(seed=s) + the same actions reproduces a trajectory exactly.
    """
    env = gym.make(ENV_ID, continuous=True, enable_wind=False,
                   max_episode_steps=max_episode_steps)
    if env.unwrapped.enable_wind:
        raise RuntimeError("wind must be disabled (docs/protocol.md §1.1)")
    return env
