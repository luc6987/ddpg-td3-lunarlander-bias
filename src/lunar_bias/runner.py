"""One run end to end: lock, status, meta, wipe-and-restart.

`runner.run(cfg, runs_dir)` is idempotent by design (`docs/implementation_plan.md`
§10). `done` and `diverged` are final and are never re-run; `crashed` and stale
`running` runs restart from a wiped directory, because a run is a pure function
of its config and there is nothing worth resuming in 10–40 minutes of training.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import socket
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any

import gymnasium
import numpy as np
import torch

from lunar_bias.config import RunConfig
from lunar_bias.io import RunWriter, atomic_write_json, atomic_write_text, read_json
from lunar_bias.train import Diverged, train

#: status.json states. `done` and `diverged` are final; the others are retried.
STATES = ("running", "done", "diverged", "crashed")
FINAL_STATES = ("done", "diverged")


def _pid_alive(pid: object) -> bool:
    if not isinstance(pid, int):
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def acquire_lock(run_dir: Path, stale_after_s: float = 6 * 3600) -> Path | None:
    """Take the per-run lock, or return None if another live process holds it."""
    run_dir.mkdir(parents=True, exist_ok=True)
    lock = run_dir / ".lock"
    for _ in range(2):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                info = json.loads(lock.read_text() or "{}")
            except (OSError, json.JSONDecodeError):
                info = {}
            same_host = info.get("host") == socket.gethostname()
            alive = same_host and _pid_alive(info.get("pid"))
            fresh = time.time() - lock.stat().st_mtime < stale_after_s
            if alive or (not same_host and fresh):
                return None
            lock.unlink(missing_ok=True)  # stale: dead pid on this host, or too old
            continue
        with os.fdopen(fd, "w") as f:
            json.dump({"host": socket.gethostname(), "pid": os.getpid(),
                       "time": time.time()}, f)
        return lock
    return None


def _git(args: list[str]) -> str | None:
    """Run a read-only git command in the current repository, or return None."""
    try:
        proc = subprocess.run(["git", *args], capture_output=True, text=True,
                              check=False, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout.strip() if proc.returncode == 0 else None


def _meta(cfg: RunConfig, start: float, end: float | None = None) -> dict[str, Any]:
    """Versions, machine, git state and timing: enough to reproduce or explain a run."""
    return {
        "run_id": cfg.run_id,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "gymnasium": gymnasium.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "hostname": socket.gethostname(),
        "cpu_count": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "git_commit": _git(["rev-parse", "HEAD"]),
        "git_dirty": bool(_git(["status", "--porcelain"])),
        "start": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(start)),
        "start_epoch": start,
        "end": None if end is None else time.strftime("%Y-%m-%dT%H:%M:%S",
                                                      time.localtime(end)),
        "end_epoch": end,
        "wall_s": None if end is None else round(end - start, 3),
    }


def _wipe(run_dir: Path) -> None:
    """Remove everything but the lock: every (re)start is from scratch."""
    for entry in run_dir.iterdir():
        if entry.name == ".lock":
            continue
        if entry.is_dir() and not entry.is_symlink():
            shutil.rmtree(entry)
        else:
            entry.unlink()


def run(cfg: RunConfig, runs_dir: Path | str = "runs", *,
        raise_on_crash: bool = False) -> str:
    """Run one config.

    Returns `skipped` (already final), `locked` (someone else is running it),
    `done`, `diverged` or `crashed`.
    """
    run_dir = Path(runs_dir) / cfg.run_id
    status = read_json(run_dir / "status.json") or {}
    if status.get("state") in FINAL_STATES:
        return "skipped"

    lock = acquire_lock(run_dir)
    if lock is None:
        return "locked"

    start = time.time()
    try:
        _wipe(run_dir)
        atomic_write_text(run_dir / "config.json", cfg.to_json())
        atomic_write_json(run_dir / "meta.json", _meta(cfg, start))
        writer = RunWriter(run_dir, cfg, started_at=start)
        writer.status("running", 0)
        try:
            train(cfg, writer)
        except Diverged as exc:
            writer.status("diverged", exc.step, error=str(exc))
            _finish_meta(run_dir, cfg, start)
            return "diverged"
        except Exception:  # KeyboardInterrupt and SystemExit keep the run "running"
            last = read_json(run_dir / "status.json") or {}
            writer.status("crashed", int(last.get("step", 0)),
                          error=traceback.format_exc())
            _finish_meta(run_dir, cfg, start)
            if raise_on_crash:
                raise
            return "crashed"
        writer.status("done", cfg.max_steps)
        _finish_meta(run_dir, cfg, start)
        return "done"
    finally:
        lock.unlink(missing_ok=True)


def _finish_meta(run_dir: Path, cfg: RunConfig, start: float) -> None:
    atomic_write_json(run_dir / "meta.json", _meta(cfg, start, end=time.time()))
