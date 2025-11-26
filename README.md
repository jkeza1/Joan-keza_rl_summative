
 # SMEEF Reinforcement Learning Project

## Project Overview
This project implements and compares four reinforcement learning algorithms (DQN, REINFORCE, A2C, PPO) on a custom environment called SMEEF (Single Mother Economic Empowerment Framework). The environment simulates decision-making for single mothers balancing resources, services, and child well-being in a grid-based world.

## Environment Description

### Mission
Maximize cumulative well-being by managing resources (money, energy, skills, social support) while reducing needs (childcare, financial, emotional, career) and improving child health/happiness.

### Action Space
Discrete with 8 actions:

- MOVE_UP (0), MOVE_DOWN (1), MOVE_LEFT (2), MOVE_RIGHT (3)

- USE_SERVICE (4), WORK_PART_TIME (5), ATTEND_TRAINING (6), SEEK_SUPPORT (7)

### Observation Space
Dictionary containing:

- position: 2D coordinates in grid

- resources: [money, energy, skills, social_support] ∈ [0,100]

- needs: [childcare, financial, emotional, career] ∈ [0,100]

- child_status: [health, happiness] ∈ [0,100]

### Reward Structure
Complex reward function that:

- Rewards improvements in resources and child status

- Penalizes energy consumption and invalid actions

- Provides goal bonuses for reaching target locations

- Applies penalties for critical failures (energy depletion, child health issues)

## Project Structure

text

```text
smeef_RL/
├── agents/                 # RL agent implementations
│   ├── a2c_agent.py
│   ├── dqn_agent.py
│   ├── ppo_agent.py
│   └── reinforce_agent.py
├── config/                # Configuration files
│   ├── env_config.py
│   └── training_config.py
├── environment/           # Custom Gym environment
│   ├── observation.py
│   ├── rendering.py
│   └── smee_env.py
├── models/               # Saved model weights
│   ├── a2c/
│   ├── dqn/
│   ├── ppo/
│   └── reinforce/
├── outputs/              # Training outputs
│   ├── logs/
│   ├── metrics/
│   ├── plots/
│   └── videos/
├── scripts/              # Utility scripts
│   ├── generate_report.py
│   ├── plot_comparison.py
│   ├── plot_single_run.py
│   └── plot_training.py
├── training/             # Training scripts
│   ├── a2c_training.py
│   ├── compare_algorithms.py
│   ├── dqn_training.py
│   ├── ppo_training.py
│   └── reinforce_training.py
├── create_diagram.py     # Environment visualization
├── enhanced_demo.py      # Enhanced demonstration
├── ppo_demo.py          # PPO-specific demo
├── requirements.txt      # Project dependencies
├── smeef.py             # Main environment file
├── smeef_demo.py        # Basic demonstration
└── README.md            # This file
```

## Installation
Clone the repository:

```bash
git clone https://github.com/jkeza1/smeef_RL.git
cd smeef_RL
```
Create and activate a virtual environment:

```bash
python -m venv .venv311
source .venv311/bin/activate  # On Windows: .venv311\Scripts\activate
```
Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Training Agents

#### DQN Training:

```bash
python training/dqn_training.py
```

#### PPO Training:

```bash
python training/ppo_training.py
```

#### A2C Training:

```bash
python training/a2c_training.py
```

#### REINFORCE Training:

```bash
python training/reinforce_training.py
```

### Running Demos

#### Basic Environment Demo:

```bash
python smeef_demo.py
```

#### PPO Trained Agent Demo:

```bash
python ppo_demo.py
```

#### Enhanced Visualization:

```bash
python enhanced_demo.py
```

### Generating Reports and Plots

#### Compare All Algorithms:

```bash
python training/compare_algorithms.py
```

#### Generate Training Plots:

```bash
python scripts/plot_training.py
```

#### Create Comparison Charts:

```bash
python scripts/plot_comparison.py
```

## Algorithm Performance Summary
Based on extensive hyperparameter tuning (10+ runs per algorithm):

| Algorithm | Best Mean Reward | Convergence | Stability | Generalization |
|---|---:|---|---|---|
| PPO | -24.00 | Fast | High | Excellent |
| DQN | -14.50 | Medium | Very High | Very Good |
| A2C | -24.40 | Fast | Medium | Good |
| REINFORCE | -16.47 | Variable | Low | Poor |

### Key Findings:

- PPO achieved the best overall performance with excellent generalization

- DQN showed remarkable stability and consistent learning

- A2C learned quickly but was sensitive to hyperparameters

- REINFORCE demonstrated high variance but educational value

## Hyperparameter Tuning
Each algorithm underwent extensive hyperparameter optimization:

- DQN: Learning rate, buffer size, exploration schedule

- PPO: Learning rate, batch size, clip range, epochs

- A2C: Learning rate, n-steps, entropy coefficient

- REINFORCE: Learning rate, gamma, hidden layer sizes

## Visualization
The environment features PyGame-based visualization showing:

- Agent position (red square)

- Special locations (home, work, services)

- Resource status overlay

- Real-time reward feedback

## Requirements
Key dependencies:

- gymnasium>=0.28.1

- stable-baselines3>=2.0.0

- pygame>=2.5.0

- numpy>=1.24.0

- matplotlib>=3.7.0

- torch>=2.0.0

See requirements.txt for complete list.

## Results
Comprehensive results including:

- Training curves for all algorithms

- Hyperparameter sensitivity analysis

- Generalization performance on unseen states

- Comparative analysis of sample efficiency

- Robustness evaluation across multiple seeds

### performance graph


 *Figure: PPO actual vs expected performance across hyperparameter sweeps.*
                          

<img width="1366" height="655" alt="ppo_graph" src="https://github.com/user-attachments/assets/67f21233-4100-497e-8d2d-06bb2cd66215" />


 *Figure: Algorithm Comparison — Cumulative Reward*
                                 

<img width="1366" height="655" alt="cumulative reward" src="https://github.com/user-attachments/assets/f3ccf6c4-c58a-4de4-bdfa-4d7a462827cc" />


 *Training Stability — Reward & Loss (A2C, PPO, REINFORCE, DQN)*

                              
<img width="1366" height="655" alt="reward-loss" src="https://github.com/user-attachments/assets/49a662e0-15fb-49bd-87d7-3bf4694e52c9" />




## Video Demonstration
Project video available at: https://youtu.be/9XwhzTRiBbo

## Author
Joan Keza

GitHub name: jkeza1
Project Repository: Joan-keza_rl_summative

## License

This project is for educational purposes as part of a reinforcement learning summative assignment.





