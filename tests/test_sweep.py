"""`sweep.py`: the union of the grids, dry runs, status and the exit code (§11).

The sweep is resumable because it is a pure function of the grid configs plus the
states already under `--runs-dir`. Every test here passes its own `--runs-dir`
under `tmp_path`: the real `runs/` is never touched.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from lunar_bias import grids
from lunar_bias.config import RunConfig


def load_script(name: str) -> ModuleType:
    """Import a CLI script by path: `scripts/` is not an importable package."""
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sweep = load_script("sweep")


def write_status(run_dir: Path, state: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "status.json").write_text(json.dumps(
        {"state": state, "step": 0, "wall_s": 1.0, "error": None}))


@pytest.mark.parametrize(("grids_spec", "n_runs"), [
    ("pilot", 6), ("e1", 40), ("e1,e2", 60), ("e1,e2,e3,e4", 80),
])
def test_a_dry_run_lists_the_unique_runs_and_writes_nothing(
        grids_spec: str, n_runs: int, tmp_path: Path, capsys, monkeypatch):
    def must_not_run(payload: dict) -> dict:
        raise AssertionError("a dry run must not run anything")

    monkeypatch.setattr(sweep, "run_one", must_not_run)
    runs_dir = tmp_path / "runs"
    assert sweep.main(["--dry-run", "--grid", grids_spec, "--runs-dir", str(runs_dir)]) == 0

    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == n_runs + 1, "one line per unique run, plus the summary"
    assert lines[-1].startswith(f"{n_runs} unique run(s) in grids {grids_spec} (")
    seeds = [int(line.split("seed=")[1].split()[0]) for line in lines[:-1]]
    assert seeds == sorted(seeds), "the order is seed-major"
    assert all(line.split()[-1] == "missing" for line in lines[:-1])
    if grids_spec == "e1,e2":
        assert seeds[:6] == [0] * 6 and seeds[-6:] == [9] * 6, \
            "an interrupted sweep leaves every condition balanced"
    assert not runs_dir.exists(), "a dry run must not create the runs directory"


def test_status_counts_done_diverged_crashed_running_and_missing(tmp_path: Path, capsys):
    runs_dir = tmp_path / "runs"
    smoke = grids.unique(grids.smoke())
    write_status(runs_dir / smoke[0].run_id, "done")
    write_status(runs_dir / smoke[1].run_id, "diverged")
    pilot = grids.unique(grids.pilot())
    write_status(runs_dir / pilot[0].run_id, "crashed")
    write_status(runs_dir / pilot[1].run_id, "running")

    assert sweep.main(["--status", "--grid", "smoke", "--runs-dir", str(runs_dir)]) == 0
    assert capsys.readouterr().out.strip() == (
        "smoke  n=  2  done=1  diverged=1  crashed=0  running=0  missing=0")

    assert sweep.main(["--status", "--grid", "pilot", "--runs-dir", str(runs_dir)]) == 0
    assert capsys.readouterr().out.strip() == (
        "pilot  n=  6  done=0  diverged=0  crashed=1  running=1  missing=4")


def test_the_exit_code_is_one_when_a_run_crashed_but_not_when_it_diverged(
        tmp_path: Path, capsys, monkeypatch):
    states: list[str] = []

    def fake_run_one(payload: dict) -> dict:
        cfg = RunConfig.from_dict(payload["cfg"])
        return {"run_id": cfg.run_id, "condition": grids.describe(cfg), "seed": cfg.seed,
                "state": states.pop(0), "wall_s": 0.5}

    monkeypatch.setattr(sweep, "run_one", fake_run_one)
    runs_dir = tmp_path / "runs"

    states[:] = ["done", "diverged"]
    assert sweep.main(["--grid", "smoke", "--jobs", "1", "--runs-dir", str(runs_dir)]) == 0
    assert "diverged=1" in capsys.readouterr().out, "diverged is a result, not an error"

    states[:] = ["done", "crashed"]
    assert sweep.main(["--grid", "smoke", "--jobs", "1", "--runs-dir", str(runs_dir)]) == 1
    assert "crashed=1" in capsys.readouterr().out
