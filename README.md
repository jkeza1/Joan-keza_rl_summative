# SMEEF_RL — Assignment-ready project

This repository contains a custom mission-based environment (SMEEF) and training/evaluation scripts for four RL methods required by the assignment: Value-based (DQN) and Policy-based (REINFORCE, PPO, A2C). The primary demo/entry point for interactive visualization and playing back saved models is `smeef.py` (see below). The code and documentation are organized to make it straightforward to run experiments, repeat hyperparameter sweeps (10+ runs per algorithm), produce recordings, and compile a short report.

## Repository layout

project_root/

- environment/
	- `smeef_env.py`        # Custom Gymnasium environment implementation (dict observation)
	- `obs_wrappers.py`     # `NormalizeFlattenObs` to flatten dict -> 12-D Box for MLP policies

- agents/
	- `reinforce_agent.py`  # PyTorch policy network used by the vanilla REINFORCE trainer

- training/
	- `dqn_training.py`     # DQN (SB3) training script
	- `a2c_ultra_fast.py`   # A2C (SB3) quick runner
	- `ppo_demo.py`         # PPO (SB3) example runner
	- `reinforce_vanilla.py`# Vanilla PyTorch REINFORCE implementation
	- `reinforce_training.py` # Sweep-capable runner for REINFORCE (10-config grid)
	- `compare_all.py`      # Loads saved models, evaluates, and creates a comparison plot

- models/
	- `dqn/`, `ppo/`, `a2c/`, `reinforce/`  # Saved model artifacts

- outputs/
	- `plots/`, `metrics/`, `videos/`, `logs/` # Generated experiment outputs

# SMEEF_RL — Assignment-ready Reinforcement Learning Project

An assignment-ready reinforcement learning project featuring a custom mission-based environment (SMEEF) and implementations of four RL algorithms: DQN, PPO, A2C and REINFORCE. This repository includes interactive visualization, static demos, training scripts, hyperparameter sweep tooling, saved models, and plotting utilities for evaluation and submission.

## Quick Start

Prerequisites

- Python 3.10+ (virtual environment recommended)
- Windows PowerShell examples shown below

Installation & setup

```powershell
# Create virtual environment
python -m venv .venv

# Activate (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

Run interactive demo

```powershell
python smeef.py
```

Run static demo (save frames / video)

```powershell
python run_random_demo.py --save-frames outputs/videos/random_demo
```

## Repository structure

Top-level layout (important files and folders):

```
SMEEF_RL/
│
├── README.md
├── requirements.txt
├── smeef.py                  # Main interactive demo (pygame visualization)
│
├── environment/
│   ├── __init__.py
	├── smeef_env.py          # Custom Gymnasium environment
	└── obs_wrappers.py       # NormalizeFlattenObs (Dict → Box)
│
├── agents/
│   ├── __init__.py
│   └── reinforce_agent.py    # PyTorch policy used by REINFORCE
│
├── training/
│   ├── __init__.py
	├── dqn_training.py       # DQN (SB3) training script
	├── ppo_demo.py           # PPO example runner (SB3)
	├── a2c_ultra_fast.py     # A2C minimal fast script
	├── reinforce_vanilla.py  # Vanilla REINFORCE implementation
	├── reinforce_training.py # REINFORCE sweeps, 10+ configs
	└── compare_all.py        # Evaluation + comparison plot
│
├── models/
│   ├── dqn/                 # Saved DQN models
│   ├── ppo/                 # Saved PPO models
│   ├── a2c/                 # Saved A2C models
# SMEEF_RL

A reinforcement-learning project that implements a custom mission-based environment (SMEEF) plus training and demo utilities for several RL algorithms (DQN, PPO, A2C, REINFORCE). The repository contains the environment code, agent implementations, training scripts, saved models, plotting utilities and example demos/visualizations.

This README documents: quick setup, how to run demos and training scripts, where artifacts are stored, and a short repo map.

## Quick start

Requirements

- Python 3.10–3.12 (recommended)
- Create and activate a virtual environment before installing dependencies.

Windows PowerShell example:

```powershell
python -m venv .venv311
.venv311\\Scripts\\Activate.ps1
pip install -r requirements.txt
```

Notes

- There is a local virtual environment directory `.venv311` in this workspace — consider adding it to `.gitignore` if you push this repo.
- The project depends on packages listed in `requirements.txt`.

## Quick examples

- Run the interactive demo (pygame visualization):

```powershell
python smeef.py
```

- Run static / headless demo (saves frames or video):

```powershell
python run_random_demo.py --save-frames outputs/videos/random_demo
```

- Run a PPO example/demo:

```powershell
python ppo_demo.py
```

## Training scripts

Top training scripts live in the `training/` folder. Examples:

- `training/dqn_training.py` — DQN training (SB3)
- `training/ppo_training.py` — PPO training (SB3)
- `training/a2c_training.py` — A2C training (SB3)
- `training/reinforce_training.py` — REINFORCE (PyTorch) and sweep tooling
- `training/compare_all.py` — evaluate multiple saved models and produce comparison plots

Run (example):

```powershell
python training/dqn_training.py
python training/reinforce_training.py --total-episodes 500
python training/compare_all.py
```

See `config/training_config.yaml` for default training hyperparameters.

## Project layout (important files)

- `smeef.py` — interactive demo and model playback (main entrypoint for visualization)
- `smeef_demo.py`, `ppo_demo.py`, `enhanced_demo.py`, `run_random_demo.py` — demo scripts
- `environment/` — custom environment implementation and wrappers
  - `smeef_env.py` — core environment
  - `obs_wrappers.py` — helper wrappers (NormalizeFlattenObs, etc.)
  - `rendering.py` — rendering helpers
- `agents/` — agent policy code
  - `a2c_agent.py`, `dqn_agent.py`, `ppo_agent.py`, `reinforce_agent.py`
- `training/` — training runners and utilities
- `models/` — saved model artifacts (SB3 `.zip`, PyTorch `.pt`/`.pth`)
- `outputs/` — generated outputs: `logs/`, `metrics/`, `plots/`, `videos/`
- `config/` — configuration YAMLs (`env_config.yaml`, `training_config.yaml`)
- `requirements.txt` — pinned Python dependencies
- `scripts/` — plotting and analysis scripts (plotting helpers used to build figures)

## Models & outputs

- Trained models are stored under `models/<algorithm>/`. The repo contains several saved runs (zip/pt files).
- Experiment artifacts (metrics, plots, TensorBoard logs, videos) are under `outputs/` (e.g. `outputs/metrics/`, `outputs/plots/`, `outputs/videos/`).

## Environment summary

The SMEEF environment (`environment/smeef_env.py`) exposes a mission-based grid-like task. Observations are provided as a dict; for training with standard MLP policies use the wrapper in `environment/obs_wrappers.py` to flatten/normalize the observation into a Box.

Reward and termination logic are implemented in `smeef_env.py`. Use the `info` dict returned on each step for diagnostics (reward components, mission status, etc.).

## Usage notes & troubleshooting

- Observation-shape mismatch: ensure you apply the same `NormalizeFlattenObs` wrapper at training and inference.
- Model compatibility: SB3 models are `.zip` files; PyTorch policies are `.pt`/`.pth` files. Check `smeef.py` for the `MODEL_PATHS` constants to point playback to a specific file.
- Missing packages: install via `pip install -r requirements.txt` into the activated venv.

## Suggested submission checklist

If preparing this repository for a submission or external sharing, include:

1. Source code (all `.py` files under repo root, `environment/`, `agents/`, `training/`)
2. Trained model checkpoints under `models/` for the algorithms you want to demonstrate
3. `outputs/` with sample `plots/`, `metrics/summary.csv`, and `videos/`
4. A short report in `report/` containing environment description, reward shaping decisions, hyperparameter choices and key figures

## Next steps I can help with

- Shorten or expand this README
- Add a CONTRIBUTING.md, LICENSE, or .gitignore that excludes local venvs
- Create a one-page PDF report template in `report/`
- Add automated scripts to export `report/figures/` from `outputs/plots/`

---

If you'd like any of the suggested follow-ups implemented (e.g., `.gitignore`, report template, or moving/renaming files for clearer structure), tell me which and I'll proceed.
Common issues and fixes



- Observation shape errors

