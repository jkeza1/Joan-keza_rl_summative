import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import json
from pathlib import Path

def load_training_logs():
    """Load training logs from all algorithms"""
    logs_dir = Path("outputs/logs")
    data = {}
    
    # DQN logs
    dqn_logs = logs_dir / "dqn_training_log.csv"
    if dqn_logs.exists():
        df = pd.read_csv(dqn_logs)
        data['DQN'] = {
            'episodes': df['episode'].values,
            'rewards': df['reward'].values,
            'moving_avg': df['reward'].rolling(window=50, min_periods=1).mean()
        }
    
    # PPO logs
    ppo_logs = logs_dir / "ppo_training_log.csv"
    if ppo_logs.exists():
        df = pd.read_csv(ppo_logs)
        data['PPO'] = {
            'episodes': df['episode'].values,
            'rewards': df['reward'].values,
            'moving_avg': df['reward'].rolling(window=50, min_periods=1).mean()
        }
    
    # A2C logs
    a2c_logs = logs_dir / "a2c_training_log.csv"
    if a2c_logs.exists():
        df = pd.read_csv(a2c_logs)
        data['A2C'] = {
            'episodes': df['episode'].values,
            'rewards': df['reward'].values,
            'moving_avg': df['reward'].rolling(window=50, min_periods=1).mean()
        }
    
    # REINFORCE logs
    reinforce_logs = logs_dir / "reinforce_training_log.csv"
    if reinforce_logs.exists():
        df = pd.read_csv(reinforce_logs)
        data['REINFORCE'] = {
            'episodes': df['episode'].values,
            'rewards': df['reward'].values,
            'moving_avg': df['reward'].rolling(window=50, min_periods=1).mean()
        }
    
    return data

def plot_cumulative_rewards_comparison():
    """Create subplot comparison of all methods"""
    data = load_training_logs()
    
    if not data:
        print("No training logs found. Run training scripts first.")
        return
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    axes = [ax1, ax2, ax3, ax4]
    
    colors = {'DQN': 'blue', 'PPO': 'green', 'A2C': 'red', 'REINFORCE': 'purple'}
    
    for idx, (algo, algo_data) in enumerate(data.items()):
        ax = axes[idx]
        episodes = algo_data['episodes']
        rewards = algo_data['rewards']
        moving_avg = algo_data['moving_avg']
        
        # Calculate cumulative rewards
        cumulative_rewards = np.cumsum(rewards)
        
        # Plot raw rewards (light) and moving average (bold)
        ax.plot(episodes, rewards, alpha=0.3, color=colors[algo], label='Raw Reward')
        ax.plot(episodes, moving_avg, linewidth=2, color=colors[algo], label='Moving Avg (50)')
        
        ax.set_title(f'{algo} - Training Progress', fontsize=14, fontweight='bold')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add performance statistics
        avg_reward = np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
        max_reward = np.max(rewards)
        ax.text(0.02, 0.98, f'Avg (last 100): {avg_reward:.1f}\nMax: {max_reward:.1f}', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('outputs/plots/cumulative_rewards_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('outputs/plots/cumulative_rewards_comparison.pdf', bbox_inches='tight')
    plt.show()
    
    print("✓ Cumulative rewards comparison plot saved to outputs/plots/")

if __name__ == "__main__":
    plot_cumulative_rewards_comparison()