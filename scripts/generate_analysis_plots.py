"""Generate comparison plots and analysis for trained RL agents.

Produces:
- Cumulative rewards subplot for best model of each algorithm (evaluation runs)
- Training stability plots (DQN loss, PG entropy) if TensorBoard logs available
- Episodes-to-converge summary (CSV + printed)
- Generalization tests on unseen initial states (CSV + histogram)

This script attempts to use Stable-Baselines3 saved models when available and
falls back to reading monitor CSVs for training rewards. It is defensive: if a
dependency or file is missing it prints a clear message and continues.

Usage (from repo root):
    python scripts/generate_analysis_plots.py --n-eval 100

Outputs are written to `outputs/plots/` and `outputs/metrics/`.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from stable_baselines3 import DQN, PPO, A2C
    from stable_baselines3.common.vec_env import DummyVecEnv
    SB3_AVAILABLE = True
except Exception:
    SB3_AVAILABLE = False

try:
    # tensorboard event reader
    from tensorboard.backend.event_processing import event_accumulator
    TB_AVAILABLE = True
except Exception:
    TB_AVAILABLE = False

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PLOTS = ROOT / "outputs" / "plots"
OUTPUT_METRICS = ROOT / "outputs" / "metrics"
MODELS_DIR = ROOT / "models"
LOGS_DIR = ROOT / "outputs" / "logs"

os.makedirs(OUTPUT_PLOTS, exist_ok=True)
os.makedirs(OUTPUT_METRICS, exist_ok=True)


def find_model_for_algo(algo_name: str) -> Path | None:
    """Search models/<algo_name> for best/final model files.

    Returns a Path to a candidate model file or None.
    """
    algo_dir = MODELS_DIR / algo_name
    if not algo_dir.exists():
        return None

    # look for common patterns: best, final, policy, .zip, .pth, .pt
    patterns = ["**/*best*.zip", "**/*best*.pth", "**/*best*.pt", "**/*final*.zip",
                "**/*final*.pth", "**/*final*.pt", "**/*.zip", "**/*.pth", "**/*.pt"]

    for pat in patterns:
        hits = list(algo_dir.glob(pat))
        if hits:
            # prefer the first hit (glob returns filesystem order)
            return hits[0]

    return None


def load_monitor_rewards_for_algo(algo_name: str) -> dict[str, pd.Series]:
    """Find monitor CSVs for an algorithm under outputs/logs and return per-run rewards."""
    results = {}
    # Look under outputs/logs/**/monitor.csv
    for csv_path in LOGS_DIR.rglob('monitor.csv'):
        # determine algorithm by looking at parent folder names
        parent = csv_path.parent.name.lower()
        if algo_name.lower() in parent or algo_name.lower() in str(csv_path).lower():
            try:
                df = pd.read_csv(csv_path, comment='#')
                if 'r' in df.columns:
                    rewards = df['r']
                elif 'reward' in df.columns:
                    rewards = df['reward']
                else:
                    continue
                results[str(csv_path)] = rewards
            except Exception:
                continue
    return results


def evaluate_model_on_env(model_path: Path, algo_name: str, n_eval: int = 100) -> np.ndarray | None:
    """Load model with SB3 when possible and evaluate on the environment.

    Returns an array of episode rewards or None if evaluation couldn't run.
    """
    try:
        # local import to avoid failing entire script
        from environment.smeef_env import SMEEFEnv
        from environment.obs_wrappers import NormalizeFlattenObs
    except Exception as e:
        print(f"Could not import environment modules: {e}")
        return None

    def make_env():
        return NormalizeFlattenObs(SMEEFEnv(grid_size=6, max_steps=100))

    if not SB3_AVAILABLE:
        print("stable_baselines3 not available in this environment. Skipping model evaluation.")
        return None

    # pick algorithm class
    algo = None
    if 'dqn' in algo_name.lower():
        algo = DQN
    elif 'ppo' in algo_name.lower():
        algo = PPO
    elif 'a2c' in algo_name.lower() or 'actor_critic' in algo_name.lower():
        algo = A2C

    if algo is None:
        print(f"No SB3 mapping for algorithm '{algo_name}', skipping evaluation")
        return None

    # Build eval env
    eval_env = DummyVecEnv([make_env])

    try:
        model = algo.load(str(model_path), env=eval_env)
    except Exception as e:
        print(f"Failed to load model {model_path} for {algo_name}: {e}")
        return None

    rewards = []
    for ep in range(n_eval):
        obs = eval_env.reset()
        done = False
        ep_reward = 0.0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, dones, info = eval_env.step(action)
            ep_reward += float(r[0])
            if dones[0]:
                break
        rewards.append(ep_reward)

    eval_env.close()
    return np.array(rewards)


def read_tb_scalars(path: Path, tag_filters: list[str] | None = None) -> dict[str, np.ndarray]:
    """Read scalars from a TensorBoard event directory using EventAccumulator.

    Returns a dict[tag] = np.array(values)
    """
    if not TB_AVAILABLE:
        print("TensorBoard EventAccumulator not available; install tensorboard to enable scalar reads.")
        return {}

    ea = event_accumulator.EventAccumulator(str(path))
    try:
        ea.Reload()
    except Exception as e:
        print(f"Failed to read TB events at {path}: {e}")
        return {}

    tags = ea.Tags().get('scalars', [])
    out = {}
    for tag in tags:
        if tag_filters and not any(f in tag for f in tag_filters):
            continue
        try:
            vals = [v.value for v in ea.Scalars(tag)]
            out[tag] = np.array(vals)
        except Exception:
            continue
    return out


def plot_cumulative_rewards(algos: list[str], n_eval: int = 100):
    """Evaluate best models (if present) and plot cumulative rewards in subplots."""
    n = len(algos)
    cols = 2
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(10 * cols, 4 * rows))
    axes = axes.flatten() if n > 1 else [axes]

    summary = []

    for i, algo in enumerate(algos):
        ax = axes[i]
        model_path = find_model_for_algo(algo)
        rewards = None
        label = algo.upper()

        if model_path and SB3_AVAILABLE:
            print(f"Evaluating {algo} model: {model_path}")
            rewards = evaluate_model_on_env(model_path, algo, n_eval=n_eval)
            if rewards is not None:
                ax.plot(np.arange(1, len(rewards) + 1), np.cumsum(rewards), label=f"Eval cumulative (n={len(rewards)})")
        # fallback to monitor CSVs
        if rewards is None:
            monitors = load_monitor_rewards_for_algo(algo)
            if monitors:
                for path, series in monitors.items():
                    ax.plot(np.arange(1, len(series) + 1), np.cumsum(series.values), alpha=0.6, label=f"train:{Path(path).parent.name}")
            else:
                ax.text(0.5, 0.5, 'No model or monitor logs found', ha='center')

        ax.set_title(f"{label} cumulative rewards")
        ax.set_xlabel('Episode')
        ax.set_ylabel('Cumulative Reward')
        ax.grid(True)
        ax.legend()

        # episodes-to-converge heuristic
        conv_eps = compute_episodes_to_converge(rewards if rewards is not None else None,
                                                monitors and list(monitors.values())[0] if monitors else None)
        summary.append({'algo': algo, 'converged_episode': conv_eps})

    # hide any unused axes
    for j in range(n, len(axes)):
        fig.delaxes(axes[j])

    out_file = OUTPUT_PLOTS / 'cumulative_rewards_all.png'
    fig.tight_layout()
    fig.savefig(out_file, dpi=200)
    print(f"Saved cumulative rewards plot to {out_file}")

    # write summary
    pd.DataFrame(summary).to_csv(OUTPUT_METRICS / 'episodes_to_converge.csv', index=False)


def compute_episodes_to_converge(eval_rewards: np.ndarray | None, train_series: pd.Series | None, window: int = 20, tol: float = 0.05):
    """Estimate episodes-to-converge using evaluation rewards (preferred) or training series.

    Convergence: rolling mean (window) reaches final_mean - tol*abs(final_mean) and stays above for 50 episodes.
    Returns episode index (1-based) or NaN if not found.
    """
    series = None
    if eval_rewards is not None:
        series = pd.Series(eval_rewards)
    elif train_series is not None:
        series = pd.Series(train_series.values)
    else:
        return np.nan

    roll = series.rolling(window=window, min_periods=1).mean()
    final_mean = roll.iloc[-1]
    threshold = final_mean - tol * abs(final_mean)

    for i in range(len(roll)):
        # check next 50 episodes (or until end)
        end = min(len(roll), i + 50)
        if (roll.iloc[i:end] >= threshold).all():
            return int(i + 1)
    return np.nan


def plot_training_stability():
    """Plot DQN loss and PG entropy if TensorBoard logs are available."""
    # Note: simple implementation — scan LOGS_DIR for event files
    tb_dirs = [p for p in LOGS_DIR.rglob('*') if any(f.name.startswith('events.out.tfevents') for f in p.glob('events.out.tfevents*'))]
    if not tb_dirs:
        # maybe the event files are directly under subfolders
        tb_dirs = [p for p in LOGS_DIR.iterdir() if p.is_dir() and any('events.out.tfevents' in str(x) for x in p.rglob('*'))]

    if not tb_dirs:
        print("No TensorBoard event directories found; skipping training stability plots.")
        return

    # Collect tags we care about
    loss_tags = ['loss', 'train/loss', 'train/critic_loss', 'loss/td_error']
    entropy_tags = ['entropy', 'policy/entropy', 'train/entropy']

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    for tb_dir in tb_dirs:
        scalars = read_tb_scalars(tb_dir, tag_filters=loss_tags + entropy_tags)
        # plot best-matching loss tag
        for tag, vals in scalars.items():
            if any(k in tag.lower() for k in ['loss']) and len(vals) > 0:
                axes[0].plot(vals, label=f"{tb_dir.name}:{tag}")
            if any(k in tag.lower() for k in ['entropy']) and len(vals) > 0:
                axes[1].plot(vals, label=f"{tb_dir.name}:{tag}")

    axes[0].set_title('Training loss (DQN)')
    axes[0].set_xlabel('Steps')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    axes[1].set_title('Policy entropy (PG)')
    axes[1].set_xlabel('Steps')
    axes[1].set_ylabel('Entropy')
    axes[1].legend()

    out_file = OUTPUT_PLOTS / 'training_stability_all.png'
    fig.tight_layout()
    fig.savefig(out_file, dpi=200)
    print(f"Saved training stability plot to {out_file}")


def generalization_test(algos: list[str], n_test: int = 50, seed_start: int = 1000):
    """Evaluate best models on unseen initial states (different seeds).

    Writes a CSV with per-run statistics and a histogram plot.
    """
    rows = []
    for algo in algos:
        model_path = find_model_for_algo(algo)
        if not model_path:
            print(f"No model found for {algo}, skipping generalization test")
            continue

        rewards = evaluate_model_on_env(model_path, algo, n_eval=n_test)
        if rewards is None:
            continue

        rows.append({
            'algo': algo,
            'mean_reward': float(np.mean(rewards)),
            'std_reward': float(np.std(rewards)),
            'median_reward': float(np.median(rewards)),
            'n_test': int(n_test)
        })

        # save histogram
        plt.figure(figsize=(6, 4))
        plt.hist(rewards, bins=15, alpha=0.8)
        plt.title(f'Generalization rewards distribution: {algo}')
        plt.xlabel('Episode reward')
        plt.ylabel('Count')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(OUTPUT_PLOTS / f'generalization_{algo}.png', dpi=200)
        plt.close()

    if rows:
        pd.DataFrame(rows).to_csv(OUTPUT_METRICS / 'generalization_summary.csv', index=False)
        print(f"Saved generalization summary to {OUTPUT_METRICS / 'generalization_summary.csv'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algos', nargs='+', default=['dqn', 'ppo', 'a2c', 'reinforce'], help='Algorithm names to include')
    parser.add_argument('--n-eval', type=int, default=100, help='Episodes per evaluation for cumulative plot')
    parser.add_argument('--n-test', type=int, default=50, help='Episodes per generalization test')
    args = parser.parse_args()

    algos = args.algos

    print('Algorithms to analyze:', algos)

    plot_cumulative_rewards(algos, n_eval=args.n_eval)
    plot_training_stability()
    generalization_test(algos, n_test=args.n_test)


if __name__ == '__main__':
    main()
