"""Plot training stability indicators: DQN loss and policy entropy for PG methods.

This script looks for TensorBoard event files (preferred) under
`outputs/logs/<algorithm>/` to extract training scalars. If `tensorboard` is
not installed it will attempt to read any CSV logs that contain 'loss' or
'entropy' columns.

Produces `outputs/plots/training_stability.png` containing subplots per
available metric.
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Optional

ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = ROOT / "outputs" / "logs"
OUT_DIR = ROOT / "outputs" / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def read_tensorboard_scalars(tb_dir: Path, tag: str) -> Optional[np.ndarray]:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except Exception:
        return None
    try:
        ea = EventAccumulator(str(tb_dir))
        ea.Reload()
        tags = ea.Tags().get('scalars', [])
        if tag in tags:
            events = ea.Scalars(tag)
            return np.array([e.value for e in events])
        # fallback: search for similar tags
        for t in tags:
            if tag.lower() in t.lower() or tag.split('/')[-1].lower() in t.lower():
                events = ea.Scalars(t)
                return np.array([e.value for e in events])
    except Exception:
        return None
    return None

def read_csv_loss_entropy(csv_path: Path, colname: str) -> Optional[np.ndarray]:
    try:
        df = pd.read_csv(csv_path, comment='#')
        if colname in df.columns:
            return df[colname].to_numpy()
        # try lowercase
        for c in df.columns:
            if colname.lower() in str(c).lower():
                return df[c].to_numpy()
    except Exception:
        return None
    return None

def collect_metrics() -> Dict[str, Dict[str, np.ndarray]]:
    metrics = {}
    if not LOGS_DIR.exists():
        print("No logs/ directory found")
        return metrics
    for alg_dir in LOGS_DIR.iterdir():
        if not alg_dir.is_dir():
            continue
        alg_name = alg_dir.name
        metrics[alg_name] = {}
        # try tensorboard in any subdir
        found_loss = None
        found_entropy = None
        for sub in alg_dir.rglob('*'):
            if sub.is_dir():
                if found_loss is None:
                    found_loss = read_tensorboard_scalars(sub, 'loss')
                if found_entropy is None:
                    found_entropy = read_tensorboard_scalars(sub, 'policy/entropy')
                if found_loss is not None and found_entropy is not None:
                    break
        # fallback to CSV files
        if found_loss is None:
            for csv in alg_dir.glob('*.csv'):
                found_loss = read_csv_loss_entropy(csv, 'loss')
                if found_loss is not None:
                    break
        if found_entropy is None:
            for csv in alg_dir.glob('*.csv'):
                found_entropy = read_csv_loss_entropy(csv, 'entropy')
                if found_entropy is not None:
                    break
        if found_loss is not None:
            metrics[alg_name]['loss'] = found_loss
        if found_entropy is not None:
            metrics[alg_name]['entropy'] = found_entropy
        # if neither found, remove empty entry
        if not metrics[alg_name]:
            metrics.pop(alg_name, None)
    return metrics

def plot_metrics(metrics: Dict[str, Dict[str, np.ndarray]]):
    if not metrics:
        print("No stability metrics found. Install tensorboard and ensure training logs are saved, or provide CSVs with 'loss'/'entropy' columns.")
        return
    # determine subplots: one for DQN loss, one for policy entropy (per-alg)
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), sharex=False)
    ax_loss, ax_ent = axes
    for alg, data in metrics.items():
        if 'loss' in data:
            series = data['loss']
            steps = np.arange(1, len(series)+1)
            ax_loss.plot(steps, series, label=alg)
        if 'entropy' in data:
            series = data['entropy']
            steps = np.arange(1, len(series)+1)
            ax_ent.plot(steps, series, label=alg)
    ax_loss.set_title('Training Loss (DQN / others)')
    ax_loss.set_ylabel('Loss')
    ax_loss.legend()
    ax_ent.set_title('Policy Entropy (PG methods)')
    ax_ent.set_ylabel('Entropy')
    ax_ent.legend()
    fig.tight_layout()
    out_path = OUT_DIR / 'training_stability.png'
    fig.savefig(out_path)
    print(f"Saved training stability plot to {out_path}")

def main():
    metrics = collect_metrics()
    plot_metrics(metrics)

if __name__ == '__main__':
    main()
