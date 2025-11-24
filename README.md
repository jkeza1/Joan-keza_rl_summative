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
│   └── reinforce/           # Saved REINFORCE models
│
├── outputs/
│   ├── logs/                # TensorBoard, console transcripts
│   ├── metrics/             # summary.csv + per-run stats
│   ├── plots/               # learning curves, comparison plots
│   └── videos/              # random demo + agent recordings
│
├── demos/
│   ├── run_random_demo.py   # Static frames/random movement (wrapper)
│   └── manual_control.py    # Optional: keyboard-controlled agent (placeholder)
│
├── report/
│   ├── figures/             # Exported plots for the PDF
  └── SMEEF_RL_Report.pdf   # Optional: final write-up (add before submission)
│
└── utils/
	 ├── __init__.py
	 ├── seeds.py             # Global seed helper
	 ├── file_paths.py        # Centralized paths for models/outputs/
	 └── plot_helpers.py      # Reusable plotting utilities

```

## Environment overview

Custom SMEEF environment (see `environment/smeef_env.py`) — mission-based gridworld with a mixed observation dictionary and a compact flattened wrapper.

Observation space (Dict):

- `position` — (2 ints) grid coordinates
- `resources` — (4 floats) [Money, Energy, Skills, Support]
- `needs` — (4 floats) [Childcare, Financial, Emotional, Career]
- `child_status` — (2 floats) [Health, Happiness]

Action space (Discrete enum):

- Movement: MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT
- Mission actions: USE_SERVICE, WORK_PART_TIME, ATTEND_TRAINING, SEEK_SUPPORT, etc.

Reward structure

- Shaped rewards based on resource deltas and need-reduction
- Additional bonuses for service usage and mission completion
- Reward breakdown exposed via `info['reward_components']` for diagnostics

Terminal conditions

- Reaching a goal location
- Critical resource depletion
- Exceeding maximum steps

Tip: for training with ML-friendly MLP policies, wrap the env with `NormalizeFlattenObs` from `environment/obs_wrappers.py` to convert the dict observation into a flat Box.

## Algorithms implemented

Implemented algorithms and where to find training code:

- DQN (value-based) — `training/dqn_training.py` (Stable-Baselines3)
- PPO (policy optimization) — `training/ppo_demo.py` (Stable-Baselines3)
- A2C (actor-critic) — `training/a2c_ultra_fast.py` (Stable-Baselines3 wrapper)
- REINFORCE (vanilla policy gradient) — `training/reinforce_training.py` (PyTorch)

Each SB3 training script uses the `NormalizeFlattenObs` wrapper to ensure the observation matches the policy network input shape.

## Usage examples

Training & evaluation

```powershell
# Quick model comparison
python training/compare_all.py

# REINFORCE smoke test (200 episodes)
python training/reinforce_training.py --total-episodes 200

# REINFORCE hyperparameter sweep (10 configurations)
python training/reinforce_training.py --sweep --total-episodes 1000

# Individual algorithm training
python training/dqn_training.py
python training/a2c_ultra_fast.py
python training/ppo_demo.py
```

Visualization & recording

```powershell
# Cinematic interactive demo
python smeef.py

# Static demo with frame saving
python run_random_demo.py --save-frames outputs/videos/random_demo
```

Model files

Place trained checkpoints in `models/<algorithm>/` (SB3 `.zip` or PyTorch `.pt`/`.pth`). `smeef.py` reads `MODEL_PATHS` constants to load models for visualization.

## Assignment requirements checklist

This project includes the items requested for assignment submission:

- Custom Gym/N mission-based environment with complex dictionary observations and mission actions
- Interactive PyGame visualization (`smeef.py`) and static demo saving (`run_random_demo.py`)
- Implementations of DQN, PPO, A2C (SB3) and REINFORCE (PyTorch)
- Hyperparameter sweep tooling for REINFORCE (10+ configs)
- Metrics logging to `outputs/metrics/summary.csv` and reusable plot helpers

## Submission artifacts

Ensure your submission includes these items:

- Full repository tree and source code
- Trained model checkpoints in `models/` for each algorithm
- `outputs/` containing `metrics/summary.csv`, `plots/`, `videos/`, and `logs/`
- Short PDF report (2–4 pages) in `report/` describing environment design, reward shaping, hyperparameter choices, and results — include exported figures from `report/figures/`

## Troubleshooting

Common issues and fixes

- Observation shape errors

```python
# Use wrapper for MLP policies
from environment.obs_wrappers import NormalizeFlattenObs
env = NormalizeFlattenObs(env)
```

- Action conversion issues

```python
# Convert numpy actions to Python ints
action = int(action[0]) if hasattr(action, '__len__') else action
```

- Model compatibility

Ensure that the observation space used during training matches the one used during evaluation/visualization. Check `MODEL_PATHS` in `smeef.py` and point them to the trained files.

If you see editor warnings about missing optional packages (numpy, torch, matplotlib), install them into the repo virtual environment:

```powershell
pip install numpy torch matplotlib stable-baselines3 gymnasium
```

## Notes & next steps

- I added lightweight wrappers and small helper files to expose the requested layout while keeping original code intact.
- If you want me to actually move/rename files (instead of wrappers), I can perform a safe refactor (git-move + update imports) and run quick syntax checks.

If you'd like that refactor (move the real implementations into `demos/` and `training/`), tell me and I'll proceed with the moves and run validation.

---

Good luck with the assignment — tell me if you want the README tweaked (shorter/longer), a submission PDF template, or automation to export figures into `report/figures/`.


