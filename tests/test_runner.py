"""`runner.py` and `io.py`: the artifacts of a run, and the resumability rules.

A run is a pure function of its config: `done` and `diverged` are final, while
`crashed` and stale `running` runs start again from a wiped directory
(docs/implementation_plan.md §10).
"""

from __future__ import annotations

import dataclasses
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from lunar_bias import grids, runner, train
from lunar_bias.agents import make_agent
from lunar_bias.config import RunConfig
from lunar_bias.io import EVAL_COLUMNS, METRICS_COLUMNS, PROBE_SUMMARY_COLUMNS, read_json
from lunar_bias.train import Diverged

#: Every key of probe/step_XXXXXXX.npz, with the dtype it must be stored in.
PROBE_DTYPES = {
    "episode": np.int16, "t": np.int16, "G": np.float64, "q": np.float32,
    "ep_seed": np.int64, "ep_len": np.int32, "ep_terminated": np.bool_,
    "ep_return_1000": np.float64, "ep_outcome": np.int8,
    "ep_G_at_train_horizon": np.float64, "gamma": np.float64,
    "train_horizon": np.int32, "probe_horizon": np.int32,
}
META_KEYS = {"run_id", "python", "torch", "gymnasium", "numpy", "platform", "machine",
             "hostname", "cpu_count", "torch_threads", "git_commit", "git_dirty",
             "start", "start_epoch", "end", "end_epoch", "wall_s"}


def tiny_config(**overrides) -> RunConfig:
    """A TD3/LN run of a few hundred steps: every artifact, in a few seconds."""
    cfg = grids.smoke()[1]
    cfg = dataclasses.replace(
        cfg, max_steps=200, log_every=100,
        algo=dataclasses.replace(cfg.algo, learning_starts=50),
        evaluation=dataclasses.replace(cfg.evaluation, every=100, checkpoint_every=100))
    return dataclasses.replace(cfg, **overrides) if overrides else cfg


@pytest.fixture
def runs_dir(tmp_path: Path) -> Path:
    return tmp_path / "runs"


def snapshot(run_dir: Path) -> dict[str, int]:
    """Every path in the run directory, with its modification time."""
    return {str(p.relative_to(run_dir)): p.stat().st_mtime_ns
            for p in [run_dir, *sorted(run_dir.rglob("*"))]}


def dead_pid() -> int:
    """A pid that is certainly dead: this host's, from a process that has exited."""
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


def test_run_produces_every_documented_artifact(runs_dir: Path):
    cfg = tiny_config()
    assert runner.run(cfg, runs_dir) == "done"
    run_dir = runs_dir / cfg.run_id

    # the directory is named by run_id, and config.json rebuilds the same run
    assert run_dir.is_dir()
    assert RunConfig.from_dict(json.loads((run_dir / "config.json").read_text())) == cfg
    assert not (run_dir / ".lock").exists(), "the lock must be released"

    status = read_json(run_dir / "status.json")
    assert status["state"] == "done" and status["step"] == cfg.max_steps
    assert status["error"] is None and status["wall_s"] > 0

    meta = read_json(run_dir / "meta.json")
    assert set(meta) == META_KEYS
    assert meta["end"] is not None and meta["wall_s"] > 0
    assert meta["torch_threads"] == 1  # I10

    # metrics.csv: one window mean per log_every steps, in order, with the agent's counters
    metrics = (run_dir / "metrics.csv").read_text().strip().splitlines()
    assert metrics[0] == ",".join(METRICS_COLUMNS)
    assert [int(line.split(",")[0]) for line in metrics[1:]] == [100, 200]
    updates = [int(line.split(",")[2]) for line in metrics[1:]]
    # warm-up is steps 1..50, and the first update is at step == learning_starts (I2)
    assert updates == [51, 151], "one gradient update per environment step after warm-up"
    assert float(metrics[1].split(",")[1]) <= float(metrics[2].split(",")[1])

    # eval_episodes.csv: one row per episode and checkpoint, with the step column
    evals = (run_dir / "eval_episodes.csv").read_text().strip().splitlines()
    assert evals[0] == ",".join(EVAL_COLUMNS)
    assert len(evals) == 1 + 3 * len(cfg.evaluation.seeds)  # steps 0, 100, 200
    assert [int(line.split(",")[0]) for line in evals[1:]] == (
        [0, 0] + [100, 100] + [200, 200])
    assert [int(line.split(",")[2]) for line in evals[1:3]] == list(cfg.evaluation.seeds)

    # probe npz: keys, dtypes and shapes
    probes = sorted((run_dir / "probe").glob("step_*.npz"))
    assert [p.name for p in probes] == ["step_0000000.npz", "step_0000100.npz",
                                        "step_0000200.npz"]
    with np.load(probes[-1]) as data:
        assert set(data.files) == set(PROBE_DTYPES)
        for key, dtype in PROBE_DTYPES.items():
            assert data[key].dtype == dtype, key
        n_points = data["t"].shape[0]
        assert data["episode"].shape == (n_points,)
        assert data["G"].shape == (n_points,)
        assert data["q"].shape == (2, n_points)  # TD3: [n_critics, N], row 0 = Q1
        for key in ("ep_seed", "ep_len", "ep_terminated", "ep_return_1000", "ep_outcome",
                    "ep_G_at_train_horizon"):
            assert data[key].shape == (len(cfg.evaluation.seeds),), key
        assert int(data["train_horizon"]) == cfg.evaluation.train_horizon
        assert int(data["probe_horizon"]) == cfg.evaluation.probe_horizon
        assert float(data["gamma"]) == cfg.algo.gamma
        contiguous = np.diff(data["t"])[np.diff(data["episode"]) == 0]
        assert (contiguous == 1).all(), "t is contiguous inside an episode"

    summary = (run_dir / "probe_summary.csv").read_text().strip().splitlines()
    assert summary[0] == ",".join(PROBE_SUMMARY_COLUMNS)
    assert len(summary) == 4

    # checkpoints every 50 k steps (here: every 100) and at the end
    ckpts = sorted((run_dir / "checkpoints").glob("step_*.pt"))
    assert [p.name for p in ckpts] == ["step_0000100.pt", "step_0000200.pt"]
    saved = torch.load(ckpts[-1], weights_only=True)
    assert set(saved) == {"actor", "critics", "step"} and saved["step"] == cfg.max_steps
    agent = make_agent(cfg)
    agent.load_state_dict({"actor": saved["actor"], "critics": saved["critics"]})


def test_a_rerun_is_skipped(runs_dir: Path):
    cfg = tiny_config()
    assert runner.run(cfg, runs_dir) == "done"
    run_dir = runs_dir / cfg.run_id

    before = snapshot(run_dir)
    assert runner.run(cfg, runs_dir) == "skipped"
    assert snapshot(run_dir) == before, "a skipped run must not touch its artifacts"


def test_a_run_held_by_a_live_process_is_locked(runs_dir: Path):
    cfg = tiny_config()
    run_dir = runs_dir / cfg.run_id
    run_dir.mkdir(parents=True)
    (run_dir / ".lock").write_text(json.dumps(  # this process is alive: the lock holds
        {"host": socket.gethostname(), "pid": os.getpid(), "time": time.time()}))
    assert runner.run(cfg, runs_dir) == "locked"
    assert not (run_dir / "metrics.csv").exists()


def test_the_first_update_happens_at_learning_starts(runs_dir: Path):
    """I2: uniform random actions up to learning_starts, then one update per step."""
    base = tiny_config()
    cfg = dataclasses.replace(
        base, max_steps=60, log_every=10,
        algo=dataclasses.replace(base.algo, learning_starts=50),
        evaluation=dataclasses.replace(base.evaluation, seeds=base.evaluation.seeds[:1]))
    assert runner.run(cfg, runs_dir) == "done"
    rows = (runs_dir / cfg.run_id / "metrics.csv").read_text().strip().splitlines()[1:]
    steps = [int(line.split(",")[0]) for line in rows]
    updates = [int(line.split(",")[2]) for line in rows]
    assert steps == [10, 20, 30, 40, 50, 60]
    assert updates[:5] == [0, 0, 0, 0, 1], "no gradient update before learning_starts"
    assert updates[5] == 11, "one update per environment step from learning_starts on"


def test_a_stale_lock_is_taken_over(runs_dir: Path):
    cfg = tiny_config()
    run_dir = runs_dir / cfg.run_id
    run_dir.mkdir(parents=True)
    (run_dir / ".lock").write_text(json.dumps(
        {"host": socket.gethostname(), "pid": 999_999_999, "time": time.time()}))
    assert runner.run(cfg, runs_dir) == "done"
    assert not (run_dir / ".lock").exists()


def test_acquire_lock_reports_a_dead_pid_as_stale(tmp_path: Path):
    lock = runner.acquire_lock(tmp_path)
    assert lock is not None and lock.exists()
    lock.unlink()
    assert runner._pid_alive(999_999_999) is False
    assert runner._pid_alive("not a pid") is False


def test_a_stale_lock_from_a_finished_process_is_taken_over(runs_dir: Path):
    """The dead pid is real: this host's, from a process that has already exited."""
    cfg = tiny_config()
    run_dir = runs_dir / cfg.run_id
    run_dir.mkdir(parents=True)
    (run_dir / ".lock").write_text(json.dumps(
        {"host": socket.gethostname(), "pid": dead_pid(), "time": time.time()}))
    assert runner.run(cfg, runs_dir) == "done", "a dead pid on this host is stale"
    assert read_json(run_dir / "status.json")["state"] == "done"
    assert not (run_dir / ".lock").exists()


def test_a_fresh_lock_from_another_host_blocks_until_it_is_old(runs_dir: Path):
    """Another host's lock is respected while fresh, and taken over once too old."""
    run_dir = runs_dir / "another-hosts-run"
    lock = run_dir / ".lock"
    run_dir.mkdir(parents=True)
    lock.write_text(json.dumps({"host": "another-host", "pid": 1, "time": time.time()}))
    assert runner.acquire_lock(run_dir) is None, "a fresh lock elsewhere is not stale"
    taken = runner.acquire_lock(run_dir, stale_after_s=0.0)
    assert taken is not None and taken.exists()
    assert json.loads(lock.read_text())["host"] == socket.gethostname()


def test_a_crashed_run_restarts_from_a_wiped_directory(runs_dir: Path, monkeypatch):
    cfg = tiny_config()
    run_dir = runs_dir / cfg.run_id

    def boom(cfg_: RunConfig, out: object) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(runner, "train", boom)
    assert runner.run(cfg, runs_dir) == "crashed"
    status = read_json(run_dir / "status.json")
    assert status["state"] == "crashed" and "RuntimeError: boom" in status["error"]

    marker = run_dir / "stale.txt"
    marker.write_text("left over from the crashed attempt")
    monkeypatch.undo()
    assert runner.run(cfg, runs_dir) == "done"
    assert not marker.exists(), "a restart must wipe the run directory"
    assert read_json(run_dir / "status.json")["state"] == "done"


def test_a_crash_inside_the_training_loop_restarts_from_a_wiped_directory(
        runs_dir: Path, monkeypatch):
    """A crash after the step-0 evaluation leaves real artifacts; the restart wipes them."""
    cfg = tiny_config()
    run_dir = runs_dir / cfg.run_id

    def broken(cfg_: RunConfig):
        agent = make_agent(cfg_)
        count = {"n": 0}
        update = agent.update

        def crash_after_two(batch):
            count["n"] += 1
            if count["n"] >= 3:  # the first gradient step is at step 50
                raise RuntimeError("boom")
            return update(batch)

        agent.update = crash_after_two
        return agent

    monkeypatch.setattr(train, "make_agent", broken)
    assert runner.run(cfg, runs_dir) == "crashed"
    assert read_json(run_dir / "status.json")["state"] == "crashed"
    crashed = (run_dir / "eval_episodes.csv").read_text().strip().splitlines()
    assert len(crashed) == 1 + len(cfg.evaluation.seeds), "the step-0 evaluation did run"

    monkeypatch.undo()  # the restart is a different process, with a working agent
    assert runner.run(cfg, runs_dir) == "done"
    evals = (run_dir / "eval_episodes.csv").read_text().strip().splitlines()
    assert [int(line.split(",")[0]) for line in evals[1:]] == [0, 0, 100, 100, 200, 200], \
        "no episode row of the crashed attempt may survive the wipe"
    metrics = (run_dir / "metrics.csv").read_text().strip().splitlines()
    assert [int(line.split(",")[0]) for line in metrics[1:]] == [100, 200]


def test_a_stale_running_run_restarts(runs_dir: Path, monkeypatch):
    cfg = tiny_config()
    run_dir = runs_dir / cfg.run_id
    run_dir.mkdir(parents=True)
    (run_dir / "status.json").write_text(json.dumps(
        {"state": "running", "step": 123, "wall_s": 1.0, "error": None}))
    assert runner.run(cfg, runs_dir) == "done"
    assert read_json(run_dir / "status.json")["state"] == "done"


def test_a_diverged_run_is_never_rerun(runs_dir: Path, monkeypatch):
    cfg = tiny_config()
    calls = []

    def diverge(cfg_: RunConfig, out: object) -> None:
        calls.append(1)
        raise Diverged(42)

    monkeypatch.setattr(runner, "train", diverge)
    assert runner.run(cfg, runs_dir) == "diverged"
    run_dir = runs_dir / cfg.run_id
    status = read_json(run_dir / "status.json")
    assert status["state"] == "diverged" and status["step"] == 42
    assert status["error"] == "non-finite critic loss at step 42"
    # a later sweep must leave it alone, and must not touch its artifacts
    before = snapshot(run_dir)
    assert runner.run(cfg, runs_dir) == "skipped"
    assert snapshot(run_dir) == before
    assert len(calls) == 1, "a diverged run is final: it is never re-run"


def test_a_crash_can_be_raised_outside_sweeps(runs_dir: Path, monkeypatch):
    cfg = tiny_config()

    def boom(cfg_: RunConfig, out: object) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(runner, "train", boom)
    with pytest.raises(RuntimeError, match="boom"):
        runner.run(cfg, runs_dir, raise_on_crash=True)
    assert read_json(runs_dir / cfg.run_id / "status.json")["state"] == "crashed"
    assert not (runs_dir / cfg.run_id / ".lock").exists(), "the lock must be released"


@pytest.mark.parametrize("index", [0, 1])
def test_the_smoke_grid_runs_end_to_end(index: int, runs_dir: Path):
    """The grid used by `./run.sh smoke`, both of its conditions."""
    cfg = grids.smoke()[index]
    assert runner.run(cfg, runs_dir) == "done"
    run_dir = runs_dir / cfg.run_id
    assert read_json(run_dir / "status.json")["state"] == "done"
    assert (run_dir / "probe" / f"step_{cfg.max_steps:07d}.npz").exists()
    assert (run_dir / "checkpoints" / f"step_{cfg.max_steps:07d}.pt").exists()
    assert runner.run(cfg, runs_dir) == "skipped"
