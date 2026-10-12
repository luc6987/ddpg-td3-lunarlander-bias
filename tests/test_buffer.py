"""`buffer.py`: ring semantics, dtypes and reproducibility (invariant I1)."""

from __future__ import annotations

import numpy as np
import torch

from lunar_bias.buffer import ReplayBuffer
from lunar_bias.envs import ACT_DIM, OBS_DIM


def _buffer(capacity: int, seed: int = 0) -> ReplayBuffer:
    return ReplayBuffer(capacity, OBS_DIM, ACT_DIM, np.random.default_rng(seed))


def _add(buffer: ReplayBuffer, value: float, terminated: bool = False) -> None:
    obs = np.full(OBS_DIM, value, np.float32)
    act = np.full(ACT_DIM, value, np.float32)
    buffer.add(obs, act, value, obs, terminated)


def test_shapes_and_dtypes():
    buffer = _buffer(10)
    _add(buffer, 1.0)
    batch = buffer.sample(4)
    assert batch.obs.shape == (4, OBS_DIM)
    assert batch.act.shape == (4, ACT_DIM)
    assert batch.rew.shape == (4,)
    assert batch.next_obs.shape == (4, OBS_DIM)
    assert batch.term.shape == (4,)
    for tensor in batch:
        assert tensor.dtype == torch.float32


def test_terminated_is_stored_as_float():
    buffer = _buffer(4)
    _add(buffer, 1.0, terminated=True)
    _add(buffer, 2.0, terminated=False)
    assert buffer.term.tolist()[:2] == [1.0, 0.0]


def test_len_grows_then_caps():
    buffer = _buffer(3)
    for i in range(3):
        assert len(buffer) == i
        _add(buffer, float(i))
    assert len(buffer) == 3
    _add(buffer, 99.0)
    assert len(buffer) == 3


def test_wrap_around_overwrites_the_oldest():
    buffer = _buffer(3)
    for i in range(5):
        _add(buffer, float(i))
    stored = set(buffer.obs[:, 0].tolist())
    assert stored == {2.0, 3.0, 4.0}
    assert buffer.ptr == 2


def test_samples_only_from_the_filled_part():
    buffer = _buffer(64)
    for i in range(5):
        _add(buffer, float(i))
    seen = set()
    for _ in range(50):
        seen |= set(buffer.sample(16).obs[:, 0].tolist())
    assert seen <= {0.0, 1.0, 2.0, 3.0, 4.0}


def test_same_rng_seed_gives_the_same_indices():
    a, b = _buffer(32, seed=7), _buffer(32, seed=7)
    for i in range(20):
        _add(a, float(i))
        _add(b, float(i))
    sa, sb = a.sample(8), b.sample(8)
    torch.testing.assert_close(sa.obs, sb.obs)
    torch.testing.assert_close(sa.rew, sb.rew)


def test_different_rng_seed_gives_different_indices():
    a, b = _buffer(64, seed=0), _buffer(64, seed=1)
    for i in range(64):
        _add(a, float(i))
        _add(b, float(i))
    assert not torch.equal(a.sample(32).rew, b.sample(32).rew)
