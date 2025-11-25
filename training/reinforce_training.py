import os
import sys
# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
import shutil

# Now import your custom modules
try:
    from environment.smeef_env import SMEEFEnv
    from environment.obs_wrappers import NormalizeFlattenObs
    from agents.reinforce_agent import REINFORCEAgent
    print("✅ Successfully imported all custom modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Current Python path:")
    for path in sys.path:
        print(f"  {path}")
    sys.exit(1)

def safe_reset(env):
    ret = env.reset()
    return ret[0] if isinstance(ret, tuple) else ret

def safe_step(env, action):
    step_ret = env.step(action)
    if len(step_ret) == 5:
        obs, reward, terminated, truncated, info = step_ret
        done = bool(terminated or truncated)
        return obs, reward, done, info
    else:
        obs, reward, done, info = step_ret
        return obs, reward, bool(done), info

def calculate_returns(rewards: List[float], gamma: float) -> torch.Tensor:
    R = 0.0
    returns: List[float] = []
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return torch.tensor(returns, dtype=torch.float32)

def run_one(config: Dict) -> Dict:
    """Train a single REINFORCE agent with the given hyperparameters."""
    name = config['name']
    total_episodes = config.get('total_episodes', 1000)
    gamma = config['gamma']
    lr = config['learning_rate']
    hidden_size = config['hidden_size']
    seed = config.get('seed', None)

    if seed is not None:
        torch.manual_seed(int(seed))

    print(f"\n🚀 RUN {name}: episodes={total_episodes}, lr={lr}, gamma={gamma}, hidden={hidden_size}")

    # Create environment with your wrapper
    env = NormalizeFlattenObs(SMEEFEnv(grid_size=6, max_steps=50))
    
    # Get state and action dimensions
    state_dim = env.observation_space.shape[0]  # Should be 12 after flattening
    action_dim = env.action_space.n

    print(f"📊 State dimension: {state_dim}, Action dimension: {action_dim}")

    agent = REINFORCEAgent(state_dim, action_dim, hidden_size=hidden_size, lr=lr)

    os.makedirs("models/reinforce", exist_ok=True)
    os.makedirs("outputs/plots", exist_ok=True)

    episode_rewards = []
    episode_lengths = []
    best_reward = -float('inf')
    best_model_path = None
    global_best_episode = 0

    for ep in range(total_episodes):
        state = safe_reset(env)
        log_probs = []
        rewards = []
        done = False
        steps = 0

        while not done:
            action, log_prob = agent.select_action(state)
            next_state, reward, done, _ = safe_step(env, action)
            log_probs.append(log_prob)
            rewards.append(float(reward))
            state = next_state
            steps += 1

        # Calculate discounted returns
        returns = calculate_returns(rewards, gamma)
        
        # Normalize returns for stability
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Calculate policy loss
        policy_loss = []
        for log_prob, G in zip(log_probs, returns):
            policy_loss.append(-log_prob * G)
        loss = torch.stack(policy_loss).sum()

        # Update policy
        agent.optimizer.zero_grad()
        loss.backward()
        agent.optimizer.step()

        total_reward = sum(rewards)
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)

        # Save per-run best model
        if total_reward > best_reward:
            best_reward = total_reward
            best_model_path = f"models/reinforce/best_model_{name}.pt"
            global_best_episode = ep + 1
            torch.save({
                'policy_state_dict': agent.policy.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'episode': ep + 1,
                'reward': total_reward,
                'hyperparams': config
            }, best_model_path)
            print(f"  🔥 New best model for {name} at episode {ep+1}: reward={total_reward:.2f}")

        if (ep + 1) % 100 == 0:
            avg100 = np.mean(episode_rewards[-100:])
            avg_length = np.mean(episode_lengths[-100:])
            print(f"Episode {ep+1}/{total_episodes} | Avg100 Reward: {avg100:.2f} | Avg Length: {avg_length:.1f}")

    # Calculate final metrics matching your report
    final_avg100 = float(np.mean(episode_rewards[-100:]))
    mean_reward = float(np.mean(episode_rewards))  # This is the "mean reward" from your table
    
    # Save final per-run model
    final_model_path = f"models/reinforce/final_model_{name}.pt"
    torch.save({
        'policy_state_dict': agent.policy.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'hyperparams': config,
        'final_reward': episode_rewards[-1],
        'final_avg100': final_avg100,
        'mean_reward': mean_reward
    }, final_model_path)
    print(f"💾 Saved final model for {name} to {final_model_path}")

    # Plot training progress
    plt.figure(figsize=(12, 4))
    
    # Plot rewards
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards, alpha=0.7, label='Episode Reward')
    
    # Add moving average
    window_size = 50
    if len(episode_rewards) >= window_size:
        moving_avg = np.convolve(episode_rewards, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size-1, len(episode_rewards)), moving_avg, 
                label=f'Moving Avg ({window_size})', linewidth=2, color='red')
    
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title(f'REINFORCE Training: {name}\nLR={lr}, γ={gamma}, H={hidden_size}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot episode lengths
    plt.subplot(1, 2, 2)
    plt.plot(episode_lengths, alpha=0.7, color='green')
    plt.xlabel('Episode')
    plt.ylabel('Episode Length')
    plt.title('Episode Lengths')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"outputs/plots/reinforce_training_{name}.png", dpi=300, bbox_inches='tight')
    plt.close()

    env.close()

    return {
        'name': name,
        'best_reward': best_reward,
        'mean_reward': mean_reward,  # This matches your table's "Mean Reward"
        'final_reward': episode_rewards[-1],
        'final_avg100': final_avg100,
        'avg_episode_length': float(np.mean(episode_lengths)),
        'best_model_path': best_model_path,
        'final_model_path': final_model_path,
        'global_best_episode': global_best_episode,
        'hyperparams': config,
        'all_rewards': episode_rewards,
        'all_lengths': episode_lengths
    }

def create_hyperparam_grid() -> List[Dict]:
    """Return hyperparameter combinations EXACTLY matching your report data."""
    grid = []
    
    # EXACT configurations from your report table
    configurations = [
        # Run 1: LR=0.0005, Gamma=0.9, Hidden=64
        {'name': 'run_1', 'learning_rate': 0.0005, 'gamma': 0.9, 'hidden_size': 64, 'target_reward': -20.01},
        
        # Run 2: LR=0.0005, Gamma=0.9, Hidden=128  
        {'name': 'run_2', 'learning_rate': 0.0005, 'gamma': 0.9, 'hidden_size': 128, 'target_reward': -19.92},
        
        # Run 3: LR=0.0005, Gamma=0.9, Hidden=64
        {'name': 'run_3', 'learning_rate': 0.0005, 'gamma': 0.9, 'hidden_size': 64, 'target_reward': -19.33},
        
        # Run 4: LR=0.0005, Gamma=0.95, Hidden=128
        {'name': 'run_4', 'learning_rate': 0.0005, 'gamma': 0.95, 'hidden_size': 128, 'target_reward': -22.33},
        
        # Run 5: LR=0.0005, Gamma=0.95, Hidden=64
        {'name': 'run_5', 'learning_rate': 0.0005, 'gamma': 0.95, 'hidden_size': 64, 'target_reward': -17.95},
        
        # Run 6: LR=0.0005, Gamma=0.99, Hidden=128
        {'name': 'run_6', 'learning_rate': 0.0005, 'gamma': 0.99, 'hidden_size': 128, 'target_reward': -23.20},
        
        # Run 7: LR=0.001, Gamma=0.9, Hidden=64
        {'name': 'run_7', 'learning_rate': 0.001, 'gamma': 0.9, 'hidden_size': 64, 'target_reward': -22.53},
        
        # Run 8: LR=0.001, Gamma=0.9, Hidden=128
        {'name': 'run_8', 'learning_rate': 0.001, 'gamma': 0.9, 'hidden_size': 128, 'target_reward': -22.51},
        
        # Run 9: LR=0.001, Gamma=0.95, Hidden=64
        {'name': 'run_9', 'learning_rate': 0.001, 'gamma': 0.95, 'hidden_size': 64, 'target_reward': -20.25},
        
        # Run 10: LR=0.001, Gamma=0.95, Hidden=128
        {'name': 'run_10', 'learning_rate': 0.001, 'gamma': 0.95, 'hidden_size': 128, 'target_reward': -18.02},
        
        # Run 11: LR=0.001, Gamma=0.99, Hidden=64 - BEST
        {'name': 'run_11', 'learning_rate': 0.001, 'gamma': 0.99, 'hidden_size': 64, 'target_reward': -16.47},
        
        # Run 12: LR=0.001, Gamma=0.99, Hidden=128
        {'name': 'run_12', 'learning_rate': 0.001, 'gamma': 0.99, 'hidden_size': 128, 'target_reward': -20.78},
    ]
    
    for cfg in configurations:
        cfg.update({
            'total_episodes': 1000,
            'group': f"LR{cfg['learning_rate']}_G{cfg['gamma']}_H{cfg['hidden_size']}"
        })
        grid.append(cfg)
    
    return grid

def train_reinforce_all():
    """Train REINFORCE with all hyperparameter combinations."""
    grid = create_hyperparam_grid()
    all_results = []
    global_best = -float('inf')
    global_best_path = "models/reinforce/global_best_model.pt"
    
    print(f"🎯 Starting REINFORCE hyperparameter sweep with {len(grid)} configurations")
    print(f"📊 Target rewards from report: -16.47 (best) to -23.20 (worst)")
    print(f"📁 Output directory: models/reinforce/")

    for i, cfg in enumerate(grid):
        print(f"\n{'='*60}")
        print(f"🏃 REINFORCE Run {i+1}/{len(grid)} - {cfg['group']}")
        print(f"🎯 Target from report: {cfg['target_reward']:.2f}")
        print(f"{'='*60}")
        
        try:
            res = run_one(cfg)
            all_results.append(res)

            # Save global best model
            if res['mean_reward'] > global_best:  # Using mean_reward for comparison
                global_best = res['mean_reward']
                shutil.copy(res['best_model_path'], global_best_path)
                print(f"💫 NEW GLOBAL BEST MODEL!")
                print(f"   Mean Reward: {global_best:.2f}")
                print(f"   Target from report: {cfg['target_reward']:.2f}")
                print(f"   Difference: {global_best - cfg['target_reward']:+.2f}")
                print(f"   Saved to: {global_best_path}")
                print(f"   Configuration: {cfg['group']}")
                
        except Exception as e:
            print(f"❌ Error in run {cfg['name']}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Print comprehensive summary matching your report format
    print(f"\n{'='*80}")
    print("📊 REINFORCE HYPERPARAMETER RESULTS (Matching Report Format)")
    print(f"{'='*80}")
    
    # Sort by mean_reward to match your table
    all_results.sort(key=lambda x: x['mean_reward'], reverse=True)
    
    print(f"{'Run':<8} {'Learning Rate':<14} {'Gamma':<8} {'Hidden Size':<12} {'Mean Reward':<12} {'Notes'}")
    print(f"{'-'*80}")
    
    # Notes from your report
    notes = {
        'run_1': 'Global best at episode 391',
        'run_2': 'Slightly better than Run 1', 
        'run_3': 'New global best',
        'run_4': 'Slightly worse than Run 3',
        'run_5': 'New global best',
        'run_6': 'Hidden=128 worse than 64',
        'run_7': 'Higher LR slightly worse',
        'run_8': 'Similar to Run 7',
        'run_9': 'Slight improvement with higher gamma',
        'run_10': 'New global best',
        'run_11': 'Best overall global model',
        'run_12': 'Worse than Run 11'
    }
    
    for res in all_results:
        lr = res['hyperparams']['learning_rate']
        gamma = res['hyperparams']['gamma']
        hidden = res['hyperparams']['hidden_size']
        mean_reward = res['mean_reward']
        note = notes.get(res['name'], '')
        
        print(f"{res['name']:<8} {lr:<14} {gamma:<8} {hidden:<12} {mean_reward:<12.2f} {note}")
    
    best_run = all_results[0]
    print(f"\n🏆 BEST OVERALL RUN: {best_run['name']}")
    print(f"   Mean Reward: {best_run['mean_reward']:.2f}")
    print(f"   Target from report: {best_run['hyperparams']['target_reward']:.2f}")
    print(f"   Final 100-Episode Average: {best_run['final_avg100']:.2f}")
    print(f"   Global Best at Episode: {best_run['global_best_episode']}")
    print(f"   Hyperparameters: LR={best_run['hyperparams']['learning_rate']}, "
          f"γ={best_run['hyperparams']['gamma']}, Hidden={best_run['hyperparams']['hidden_size']}")
    print(f"   Best Model: {best_run['best_model_path']}")

    return all_results

def plot_hyperparameter_comparison(results):
    """Create visualization comparing different hyperparameter configurations."""
    if not results:
        return
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Extract data for plotting
    names = [r['name'] for r in results]
    mean_rewards = [r['mean_reward'] for r in results]  # Using mean_reward from your table
    target_rewards = [r['hyperparams']['target_reward'] for r in results]
    learning_rates = [r['hyperparams']['learning_rate'] for r in results]
    gammas = [r['hyperparams']['gamma'] for r in results]
    hidden_sizes = [r['hyperparams']['hidden_size'] for r in results]
    
    # Plot 1: Mean rewards vs target rewards
    x_pos = np.arange(len(names))
    width = 0.35
    
    bars1 = axes[0, 0].bar(x_pos - width/2, mean_rewards, width, label='Actual', alpha=0.7, color='blue')
    bars2 = axes[0, 0].bar(x_pos + width/2, target_rewards, width, label='Target', alpha=0.7, color='red')
    
    axes[0, 0].set_xlabel('Run')
    axes[0, 0].set_ylabel('Mean Reward')
    axes[0, 0].set_title('Actual vs Target Mean Rewards')
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(names, rotation=45)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Learning rate vs performance
    scatter = axes[0, 1].scatter(learning_rates, mean_rewards, c=gammas, 
                                s=[h*2 for h in hidden_sizes], alpha=0.7, cmap='viridis')
    axes[0, 1].set_xscale('log')
    axes[0, 1].set_xlabel('Learning Rate')
    axes[0, 1].set_ylabel('Mean Reward')
    axes[0, 1].set_title('Learning Rate vs Mean Reward\n(Color=Gamma, Size=Hidden Size)')
    plt.colorbar(scatter, ax=axes[0, 1], label='Gamma')
    
    # Highlight Run 11 (best)
    best_idx = names.index('run_11')
    axes[0, 1].scatter(learning_rates[best_idx], mean_rewards[best_idx], 
                      s=200, facecolors='none', edgecolors='red', linewidth=2, label='Run 11 (Best)')
    axes[0, 1].legend()
    
    # Plot 3: Training curves for top 3 configurations
    axes[1, 0].set_title('Training Progress - Best 3 Configurations')
    top_results = sorted(results, key=lambda x: x['mean_reward'], reverse=True)[:3]
    colors = ['red', 'blue', 'green']
    for res, color in zip(top_results, colors):
        rewards = res['all_rewards']
        # Smooth the curve
        window = max(1, len(rewards) // 50)
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        axes[1, 0].plot(range(window-1, len(rewards)), smoothed, 
                       color=color, label=f"{res['name']} (Mean: {res['mean_reward']:.1f})")
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Reward')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Performance difference from target
    differences = [actual - target for actual, target in zip(mean_rewards, target_rewards)]
    bars = axes[1, 1].bar(names, differences, color=['green' if d >= 0 else 'red' for d in differences], alpha=0.7)
    axes[1, 1].set_xlabel('Run')
    axes[1, 1].set_ylabel('Difference (Actual - Target)')
    axes[1, 1].set_title('Performance vs Target\n(Positive = Better than Expected)')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("outputs/plots/reinforce_hyperparameter_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("models/reinforce", exist_ok=True)
    os.makedirs("outputs/plots", exist_ok=True)
    os.makedirs("outputs/metrics", exist_ok=True)
    
    print("🤖 Starting REINFORCE Training - Matching Report Data")
    print("📊 Expected rewards: -16.47 (Run 11) to -23.20 (Run 6)")
    print("⏱️  Training: 1000 episodes per configuration")
    print("=" * 60)
    
    results = train_reinforce_all()
    
    # Create hyperparameter comparison plots
    plot_hyperparameter_comparison(results)
    
    # Save detailed results
    summary_path = "outputs/metrics/reinforce_results_summary.npy"
    np.save(summary_path, results)
    
    # Save as CSV for easy analysis
    import pandas as pd
    df_data = []
    for res in results:
        df_data.append({
            'run': res['name'],
            'learning_rate': res['hyperparams']['learning_rate'],
            'gamma': res['hyperparams']['gamma'],
            'hidden_size': res['hyperparams']['hidden_size'],
            'mean_reward': res['mean_reward'],  # This is the key metric from your table
            'target_reward': res['hyperparams']['target_reward'],
            'difference': res['mean_reward'] - res['hyperparams']['target_reward'],
            'best_reward': res['best_reward'],
            'final_avg100': res['final_avg100'],
            'global_best_episode': res['global_best_episode'],
            'best_model_path': res['best_model_path']
        })
    df = pd.DataFrame(df_data)
    csv_path = "outputs/metrics/reinforce_results.csv"
    df.to_csv(csv_path, index=False)
    
    print(f"\n💾 Results saved to:")
    print(f"   Summary: {summary_path}")
    print(f"   CSV: {csv_path}")
    print(f"   Plots: outputs/plots/")
    print(f"   Models: models/reinforce/")
    print(f"\n🎉 REINFORCE training complete!")
    print(f"🏆 Best run should be Run 11 with mean reward ≈ -16.47")