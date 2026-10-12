# DDPG vs TD3 on continuous LunarLander

Layer normalisation, overestimation bias and performance: the code of a small
empirical study (MS2A reinforcement-learning course project).

## What it does

Two questions on `LunarLander-v3` (continuous actions, no wind):

1. Does **layer normalisation** (LN) in the critic change the **overestimation
   bias** of DDPG and TD3?
2. Does the bias affect **performance**? This is tested by manipulation: TD3 is
   kept fixed and only the reduction over its two target critics (min, mean or
   max) is changed.

The bias is estimated by comparing the critic, $Q_1(s_t, \pi(s_t))$, with the
Monte-Carlo return-to-go $G_t$ of the same rollout. The rollouts are 20 fixed
evaluation episodes, shared by every run and checkpoint, so one pass measures
both performance and bias. LunarLander's transitions are stochastic, so each
$G_t$ is a single unbiased sample of $Q^\pi$, not an exact value.

| experiment | conditions | runs |
| --- | --- | --- |
| E1 | algorithm {DDPG, TD3} × critic normalisation {none, LayerNorm} | 10 seeds each |
| E2 | TD3 × target reduction {min, mean, max} (min is E1's TD3 without LN) | 10 seeds each |

Every run has 300 000 environment steps and two hidden layers of 256 units. Two
more grids, `e3` (LN × update-to-data ratio) and `e4` (LN × depth), are defined
but were not run for the report.

## Setup

Python 3.12 and [uv](https://docs.astral.sh/uv/). CPU only, macOS arm64 or
Linux. The environment is locked in `uv.lock`.

```bash
uv sync --frozen
uv run pytest -q
```

Notes: on Linux, torch comes from the PyTorch CPU index (see `pyproject.toml`).
LunarLander needs `gymnasium[box2d]`, which pulls `box2d==2.3.10` (wheels for
macOS arm64 and Linux); do not pin `box2d-py`. `requirements.txt` is generated
from the lock with `uv export`; do not edit it by hand.

## Usage

One run. Its configuration is the same as the same condition inside a grid.

```bash
uv run python scripts/train.py --algo td3 --critic-norm layer --seed 0 --runs-dir runs
```

A grid, resumable and in parallel (one torch thread per process). `smoke` is two
very short runs to check the set-up; `e1,e2` are the 60 runs of the study.

```bash
uv run python scripts/sweep.py --grid smoke --jobs 2 --runs-dir runs
uv run python scripts/sweep.py --grid e1,e2 --jobs 8 --runs-dir runs
uv run python scripts/sweep.py --status --grid e1,e2 --runs-dir runs
```

The analysis reads `runs/`, never trains, and writes the figures, tables and
`numbers.json` to `--out`.

```bash
uv run python scripts/analyze.py --runs runs --out report
```

The 60 runs of E1 and E2 took 43.6 CPU-hours in total. The raw run data (about
0.5 GB) are not in this repository; `scripts/sweep.py` regenerates them.

## Layout

| path | content |
| --- | --- |
| `src/lunar_bias/` | `config` (frozen dataclasses, `run_id`), `envs`, `networks` (MLP with optional LayerNorm), `buffer`, `agents` (DDPG, and TD3 as DDPG plus three mechanisms), `evaluation` (the evaluation episodes are the bias probe), `train`, `io` and `runner` (resumable runs), `grids`, `analysis/` (summaries, statistics, figures, tables) |
| `scripts/` | command-line entry points: `train`, `sweep`, `analyze`, plus `pilot_report`, `probe_noise`, `q_range` and `check_report` |
| `tests/` | the pytest suite |
| `pyproject.toml`, `uv.lock`, `requirements.txt` | dependencies and the locked environment |

One run is one directory `runs/<run_id>/` with `config.json`, `status.json`,
`metrics.csv`, `eval_episodes.csv`, the raw probe arrays `probe/step_*.npz` and
small checkpoints. `run_id` is a hash of the configuration without the free-text
`tag`, so a condition shared by two grids is a single run.
