import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
from pathlib import Path

def load_stability_metrics():
    """Load stability metrics from training logs"""
    logs_dir = Path("outputs/logs")
    stability_data = {}
    
    # DQN Stability Metrics (Loss curves)
    dqn_logs = logs_dir / "dqn_training_log.csv"
    if dqn_logs.exists():
        df = pd.read_csv(dqn_logs)
        stability_data['DQN'] = {
            'episodes': df['episode'].values,
            'loss': df.get('loss', np.zeros(len(df))),  # Q-loss
            'epsilon': df.get('epsilon', np.zeros(len(df))),  # Exploration
            'q_values': df.get('q_value', np.zeros(len(df)))  # Average Q-values
        }
    
    # PPO Stability Metrics
    ppo_logs = logs_dir / "ppo_training_log.csv"
    if ppo_logs.exists():
        df = pd.read_csv(ppo_logs)
        stability_data['PPO'] = {
            'episodes': df['episode'].values,
            'policy_loss': df.get('policy_loss', np.zeros(len(df))),
            'value_loss': df.get('value_loss', np.zeros(len(df))),
            'entropy': df.get('entropy', np.zeros(len(df)))
        }
    
    # A2C Stability Metrics
    a2c_logs = logs_dir / "a2c_training_log.csv"
    if a2c_logs.exists():
        df = pd.read_csv(a2c_logs)
        stability_data['A2C'] = {
            'episodes': df['episode'].values,
            'policy_loss': df.get('policy_loss', np.zeros(len(df))),
            'value_loss': df.get('value_loss', np.zeros(len(df))),
            'entropy': df.get('entropy', np.zeros(len(df)))
        }
    
    # REINFORCE Stability Metrics
    reinforce_logs = logs_dir / "reinforce_training_log.csv"
    if reinforce_logs.exists():
        df = pd.read_csv(reinforce_logs)
        stability_data['REINFORCE'] = {
            'episodes': df['episode'].values,
            'policy_loss': df.get('policy_loss', np.zeros(len(df))),
            'entropy': df.get('entropy', np.zeros(len(df))),
            'grad_norm': df.get('grad_norm', np.zeros(len(df)))
        }
    
    return stability_data

def plot_training_stability():
    """Create comprehensive training stability analysis"""
    stability_data = load_stability_metrics()
    
    if not stability_data:
        print("No stability metrics found. Ensure training scripts log these metrics.")
        return
    
    # Create comprehensive stability plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: DQN Stability (Loss and Q-values)
    if 'DQN' in stability_data:
        ax1 = axes[0, 0]
        dqn_data = stability_data['DQN']
        
        if np.any(dqn_data['loss'] > 0):
            ax1.plot(dqn_data['episodes'], dqn_data['loss'], 'b-', alpha=0.7, label='Q-Loss')
        if np.any(dqn_data['q_values'] > 0):
            ax1.plot(dqn_data['episodes'], dqn_data['q_values'], 'r-', alpha=0.7, label='Avg Q-Value')
        
        ax1.set_title('DQN: Q-Loss and Value Stability', fontweight='bold')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Loss / Q-Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add epsilon decay if available
        if np.any(dqn_data['epsilon'] > 0):
            ax1_eps = ax1.twinx()
            ax1_eps.plot(dqn_data['episodes'], dqn_data['epsilon'], 'g--', alpha=0.5, label='Epsilon')
            ax1_eps.set_ylabel('Exploration Rate', color='green')
            ax1_eps.legend(loc='lower right')
    
    # Plot 2: Policy Gradient Entropy (PPO, A2C, REINFORCE)
    ax2 = axes[0, 1]
    for algo in ['PPO', 'A2C', 'REINFORCE']:
        if algo in stability_data and np.any(stability_data[algo]['entropy'] > 0):
            data = stability_data[algo]
            ax2.plot(data['episodes'], data['entropy'], label=f'{algo} Entropy', alpha=0.8)
    
    ax2.set_title('Policy Gradient: Policy Entropy', fontweight='bold')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Entropy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Policy Loss Comparison
    ax3 = axes[1, 0]
    for algo in ['PPO', 'A2C', 'REINFORCE']:
        if algo in stability_data and np.any(stability_data[algo]['policy_loss'] > 0):
            data = stability_data[algo]
            # Smooth the loss for better visualization
            smoothed_loss = pd.Series(data['policy_loss']).rolling(window=20, min_periods=1).mean()
            ax3.plot(data['episodes'], smoothed_loss, label=f'{algo} Policy Loss', alpha=0.8)
    
    ax3.set_title('Policy Gradient: Policy Loss (Smoothed)', fontweight='bold')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Policy Loss')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Value Loss Comparison (PPO vs A2C)
    ax4 = axes[1, 1]
    for algo in ['PPO', 'A2C']:
        if algo in stability_data and np.any(stability_data[algo]['value_loss'] > 0):
            data = stability_data[algo]
            smoothed_loss = pd.Series(data['value_loss']).rolling(window=20, min_periods=1).mean()
            ax4.plot(data['episodes'], smoothed_loss, label=f'{algo} Value Loss', alpha=0.8)
    
    ax4.set_title('Actor-Critic: Value Loss (Smoothed)', fontweight='bold')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Value Loss')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/plots/training_stability_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig('outputs/plots/training_stability_analysis.pdf', bbox_inches='tight')
    plt.show()
    
    print("✓ Training stability analysis plot saved to outputs/plots/")

def generate_stability_analysis():
    """Generate textual analysis of training stability"""
    stability_data = load_stability_metrics()
    
    analysis = "## Training Stability Analysis\n\n"
    
    for algo, data in stability_data.items():
        analysis += f"### {algo}\n"
        
        if algo == 'DQN':
            if np.any(data['loss'] > 0):
                final_loss = data['loss'][-100:].mean()
                loss_std = data['loss'][-100:].std()
                analysis += f"- **Q-Loss Stability**: Final 100 episodes: μ={final_loss:.3f}, σ={loss_std:.3f}\n"
                analysis += f"- **Convergence**: {'Stable' if loss_std < 0.1 else 'Volatile'} loss pattern\n"
        
        elif algo in ['PPO', 'A2C', 'REINFORCE']:
            if np.any(data['entropy'] > 0):
                final_entropy = data['entropy'][-100:].mean()
                entropy_trend = "decreasing" if data['entropy'][-1] < data['entropy'][0] else "stable/increasing"
                analysis += f"- **Policy Entropy**: Final={final_entropy:.3f}, Trend={entropy_trend}\n"
                analysis += f"- **Exploration**: {'Adequate' if final_entropy > 0.1 else 'Limited'} exploration\n"
            
            if np.any(data.get('policy_loss', []) > 0):
                policy_loss_std = data['policy_loss'][-100:].std()
                analysis += f"- **Policy Loss Stability**: σ={policy_loss_std:.3f}\n"
        
        analysis += "\n"
    
    # Save analysis to file
    with open('outputs/plots/stability_analysis.md', 'w') as f:
        f.write(analysis)
    
    print("✓ Stability analysis saved to outputs/plots/stability_analysis.md")
    print("\n" + analysis)

if __name__ == "__main__":
    plot_training_stability()
    generate_stability_analysis()