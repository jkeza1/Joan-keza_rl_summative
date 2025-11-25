"""Plot cumulative rewards per episode for best models of each method.

This script searches for per-algorithm episode logs in `outputs/logs/` and
attempts to extract episode rewards. It prefers Gym Monitor CSV files
(monitor.csv or .monitor.csv). If TensorBoard event files are present and the
`tensorboard` package is installed, it can also read scalar summaries from
event files.

Outputs a PNG to `outputs/plots/cumulative_rewards_comparison.png`.
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = ROOT / "outputs" / "logs"
OUT_DIR = ROOT / "outputs" / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def read_monitor_csv(path: Path) -> Optional[np.ndarray]:
    try:
        # Gym monitor CSVs often have comment lines starting with '#'
        df = pd.read_csv(path, comment="#")
        # Common conventions: column 'r' or 'reward' stores episode reward
        for col in ["r", "reward", "episode_reward", "value"]:
            if col in df.columns:
                return df[col].to_numpy()
        # Some monitor files have the reward as first unnamed column
        if df.shape[1] >= 1:
            return df.iloc[:, 0].to_numpy()
    except Exception:
        return None
    return None

def read_tensorboard_scalar_from_dir(tb_dir: Path, scalar_keys: List[str]) -> Optional[np.ndarray]:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except Exception:
        return None
    event_files = list(tb_dir.glob("events.*"))
    if not event_files:
        return None
    ea = EventAccumulator(str(tb_dir))
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    # try to find a matching scalar key
    for key in scalar_keys:
        if key in tags:
            events = ea.Scalars(key)
            return np.array([e.value for e in events])
    # fallback: try any scalar that looks like a reward
    for tag in tags:
        if "reward" in tag.lower() or "episode" in tag.lower():
            events = ea.Scalars(tag)
            return np.array([e.value for e in events])
    return None

def collect_rewards() -> Dict[str, np.ndarray]:
    results = {}
    # look for algorithm folders under outputs/logs
    if not LOGS_DIR.exists():
        print(f"No logs directory found at {LOGS_DIR}")
        return results
    for alg_dir in LOGS_DIR.iterdir():
        if not alg_dir.is_dir():
            continue
        # check for monitor csv files
        monitor_candidates = list(alg_dir.glob("monitor*.csv")) + list(alg_dir.glob("*.csv"))
        series = None
        for cand in monitor_candidates:
            series = read_monitor_csv(cand)
            if series is not None and len(series) > 0:
                results[alg_dir.name] = series
                break
        if series is not None:
            continue
        # check for tensorboard event files inside this folder or nested folders
        tb_series = None
        for sub in alg_dir.rglob("*"):
            if sub.is_dir():
                tb_series = read_tensorboard_scalar_from_dir(sub, ["rollout/ep_rew_mean", "eval/episode_reward", "episode_reward", "reward"])
                if tb_series is not None:
                    results[alg_dir.name] = tb_series
                    break
        # no data found -> skip
    return results

def plot_rewards(all_rewards: Dict[str, np.ndarray]):
    if not all_rewards:
        print("No reward series found. Check outputs/logs/ for monitor CSVs or install tensorboard to read event files.")
        return
    n = len(all_rewards)
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(8, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, (alg, series) in zip(axes, all_rewards.items()):
        episodes = np.arange(1, len(series) + 1)
        ax.plot(episodes, series, label=f"{alg} per-episode reward")
        # also plot running average
        window = max(1, len(series)//50)
        running = pd.Series(series).rolling(window=window).mean()
        ax.plot(episodes, running, label=f"running mean (window={window})", color="orange")
        ax.set_title(alg)
        ax.set_ylabel("Episode Reward")
        ax.legend()
    axes[-1].set_xlabel("Episode")
    fig.tight_layout()
    out_path = OUT_DIR / "cumulative_rewards_comparison.png"
    fig.savefig(out_path)
    print(f"Saved plot to {out_path}")

def main():
    data = collect_rewards()
    plot_rewards(data)

if __name__ == "__main__":
    main()
