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

# Rest of your code remains the same...
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

    # Save final per-run model
    final_model_path = f"models/reinforce/final_model_{name}.pt"
    torch.save({
        'policy_state_dict': agent.policy.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'hyperparams': config,
        'final_reward': episode_rewards[-1],
        'final_avg100': float(np.mean(episode_rewards[-100:]))
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
    plt.title(f'REINFORCE Training Rewards: {name}')
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
        'final_reward': episode_rewards[-1],
        'final_avg100': float(np.mean(episode_rewards[-100:])),
        'avg_episode_length': float(np.mean(episode_lengths)),
        'best_model_path': best_model_path,
        'final_model_path': final_model_path,
        'hyperparams': config,
        'all_rewards': episode_rewards,
        'all_lengths': episode_lengths
    }

def create_hyperparam_grid() -> List[Dict]:
    """Return hyperparameter combinations matching your report."""
    grid = []
    
    # Hyperparameters from your report results - 12 configurations
    configurations = [
        # Run 1-3: LR=0.0005, Gamma=0.9
        {'name': 'run_1', 'learning_rate': 0.0005, 'gamma': 0.9, 'hidden_size': 64},
        {'name': 'run_2', 'learning_rate': 0.0005, 'gamma': 0.9, 'hidden_size': 128},
        {'name': 'run_3', 'learning_rate': 0.0005, 'gamma': 0.9, 'hidden_size': 64},
        
        # Run 4-6: LR=0.0005, Gamma=0.95, 0.99
        {'name': 'run_4', 'learning_rate': 0.0005, 'gamma': 0.95, 'hidden_size': 128},
        {'name': 'run_5', 'learning_rate': 0.0005, 'gamma': 0.95, 'hidden_size': 64},
        {'name': 'run_6', 'learning_rate': 0.0005, 'gamma': 0.99, 'hidden_size': 128},
        
        # Run 7-9: LR=0.001, Gamma=0.9, 0.95
        {'name': 'run_7', 'learning_rate': 0.001, 'gamma': 0.9, 'hidden_size': 64},
        {'name': 'run_8', 'learning_rate': 0.001, 'gamma': 0.9, 'hidden_size': 128},
        {'name': 'run_9', 'learning_rate': 0.001, 'gamma': 0.95, 'hidden_size': 64},
        
        # Run 10-12: LR=0.001, Gamma=0.95, 0.99
        {'name': 'run_10', 'learning_rate': 0.001, 'gamma': 0.95, 'hidden_size': 128},
        {'name': 'run_11', 'learning_rate': 0.001, 'gamma': 0.99, 'hidden_size': 64},
        {'name': 'run_12', 'learning_rate': 0.001, 'gamma': 0.99, 'hidden_size': 128},
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
    print(f"📁 Output directory: models/reinforce/")

    for i, cfg in enumerate(grid):
        print(f"\n{'='*60}")
        print(f"🏃 REINFORCE Run {i+1}/{len(grid)} - {cfg['group']}")
        print(f"{'='*60}")
        
        try:
            res = run_one(cfg)
            all_results.append(res)

            # Save global best model
            if res['best_reward'] > global_best:
                global_best = res['best_reward']
                shutil.copy(res['best_model_path'], global_best_path)
                print(f"💫 NEW GLOBAL BEST MODEL!")
                print(f"   Reward: {global_best:.2f}")
                print(f"   Saved to: {global_best_path}")
                print(f"   Configuration: {cfg['group']}")
                
        except Exception as e:
            print(f"❌ Error in run {cfg['name']}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Print comprehensive summary
    print(f"\n{'='*80}")
    print("📊 REINFORCE HYPERPARAMETER SWEEP SUMMARY")
    print(f"{'='*80}")
    
    # Sort by best reward
    all_results.sort(key=lambda x: x['best_reward'], reverse=True)
    
    print(f"{'Run':<10} {'Best Reward':<12} {'Final100':<10} {'Avg Length':<12} {'Hyperparameters'}")
    print(f"{'-'*80}")
    
    for res in all_results:
        print(f"{res['name']:<10} {res['best_reward']:<12.2f} {res['final_avg100']:<10.2f} "
              f"{res['avg_episode_length']:<12.1f} {res['hyperparams']['group']}")
    
    best_run = all_results[0]
    print(f"\n🏆 BEST OVERALL RUN: {best_run['name']}")
    print(f"   Best Reward: {best_run['best_reward']:.2f}")
    print(f"   Final 100-Episode Average: {best_run['final_avg100']:.2f}")
    print(f"   Average Episode Length: {best_run['avg_episode_length']:.1f}")
    print(f"   Hyperparameters: {best_run['hyperparams']}")
    print(f"   Best Model: {best_run['best_model_path']}")
    print(f"   Global Best Saved: {global_best_path}")

    return all_results

def plot_hyperparameter_comparison(results):
    """Create visualization comparing different hyperparameter configurations."""
    if not results:
        return
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Extract data for plotting
    names = [r['name'] for r in results]
    best_rewards = [r['best_reward'] for r in results]
    final_avg100 = [r['final_avg100'] for r in results]
    learning_rates = [r['hyperparams']['learning_rate'] for r in results]
    gammas = [r['hyperparams']['gamma'] for r in results]
    hidden_sizes = [r['hyperparams']['hidden_size'] for r in results]
    
    # Plot 1: Best rewards by configuration
    bars = axes[0, 0].bar(names, best_rewards, color='skyblue', alpha=0.7)
    axes[0, 0].set_title('Best Rewards by Configuration')
    axes[0, 0].set_ylabel('Best Reward')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Highlight the best run
    best_idx = np.argmax(best_rewards)
    bars[best_idx].set_color('red')
    
    # Plot 2: Learning rate vs performance
    scatter = axes[0, 1].scatter(learning_rates, best_rewards, c=gammas, 
                                s=[h*2 for h in hidden_sizes], alpha=0.7, cmap='viridis')
    axes[0, 1].set_xscale('log')
    axes[0, 1].set_xlabel('Learning Rate')
    axes[0, 1].set_ylabel('Best Reward')
    axes[0, 1].set_title('Learning Rate vs Performance\n(Color=Gamma, Size=Hidden Size)')
    plt.colorbar(scatter, ax=axes[0, 1], label='Gamma')
    
    # Plot 3: Training curves for top 4 configurations
    axes[1, 0].set_title('Training Progress - Top 4 Configurations')
    top_results = sorted(results, key=lambda x: x['best_reward'], reverse=True)[:4]
    for res in top_results:
        rewards = res['all_rewards']
        # Smooth the curve
        window = max(1, len(rewards) // 50)
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        axes[1, 0].plot(range(window-1, len(rewards)), smoothed, 
                       label=f"{res['name']} (Best: {res['best_reward']:.1f})")
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Reward')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Gamma vs performance
    for lr in set(learning_rates):
        mask = [lr == l for l in learning_rates]
        axes[1, 1].scatter([g for i, g in enumerate(gammas) if mask[i]], 
                          [r for i, r in enumerate(best_rewards) if mask[i]],
                          label=f'LR={lr}', s=50)
    axes[1, 1].set_xlabel('Gamma')
    axes[1, 1].set_ylabel('Best Reward')
    axes[1, 1].set_title('Gamma vs Performance by Learning Rate')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("outputs/plots/reinforce_hyperparameter_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("models/reinforce", exist_ok=True)
    os.makedirs("outputs/plots", exist_ok=True)
    os.makedirs("outputs/metrics", exist_ok=True)
    
    print("🤖 Starting REINFORCE Training")
    print("=" * 50)
    
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
            'best_reward': res['best_reward'],
            'final_reward': res['final_reward'],
            'final_avg100': res['final_avg100'],
            'avg_episode_length': res['avg_episode_length'],
            'learning_rate': res['hyperparams']['learning_rate'],
            'gamma': res['hyperparams']['gamma'],
            'hidden_size': res['hyperparams']['hidden_size'],
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
    print("\n🎉 All REINFORCE training complete!")