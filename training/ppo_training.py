import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

def create_ppo_hyperparameter_combinations():
    """Generate 10 different PPO hyperparameter combinations"""
    return [
        # Learning Rate Variations
        {'learning_rate': 0.001, 'n_steps': 64, 'batch_size': 16, 'n_epochs': 3, 'clip_range': 0.2, 'group': 'High LR'},
        {'learning_rate': 0.0003, 'n_steps': 64, 'batch_size': 16, 'n_epochs': 3, 'clip_range': 0.2, 'group': 'Standard LR'},
        {'learning_rate': 0.0001, 'n_steps': 64, 'batch_size': 16, 'n_epochs': 3, 'clip_range': 0.2, 'group': 'Low LR'},
        
        # Step Variations
        {'learning_rate': 0.0003, 'n_steps': 32, 'batch_size': 16, 'n_epochs': 3, 'clip_range': 0.2, 'group': 'Short Steps'},
        {'learning_rate': 0.0003, 'n_steps': 128, 'batch_size': 16, 'n_epochs': 3, 'clip_range': 0.2, 'group': 'Long Steps'},
        
        # Batch Size Variations
        {'learning_rate': 0.0003, 'n_steps': 64, 'batch_size': 8, 'n_epochs': 3, 'clip_range': 0.2, 'group': 'Small Batch'},
        {'learning_rate': 0.0003, 'n_steps': 64, 'batch_size': 32, 'n_epochs': 3, 'clip_range': 0.2, 'group': 'Large Batch'},
        
        # Epoch Variations
        {'learning_rate': 0.0003, 'n_steps': 64, 'batch_size': 16, 'n_epochs': 2, 'clip_range': 0.2, 'group': 'Few Epochs'},
        {'learning_rate': 0.0003, 'n_steps': 64, 'batch_size': 16, 'n_epochs': 5, 'clip_range': 0.2, 'group': 'More Epochs'},
        
        # Clip Range Variations
        {'learning_rate': 0.0003, 'n_steps': 64, 'batch_size': 16, 'n_epochs': 3, 'clip_range': 0.1, 'group': 'Tight Clip'},
        {'learning_rate': 0.0003, 'n_steps': 64, 'batch_size': 16, 'n_epochs': 3, 'clip_range': 0.3, 'group': 'Loose Clip'},
    ]

def train_ppo_ultra_fast():
    print("⚡ Starting ULTRA-FAST PPO Hyperparameter Tuning...")
    
    # Create environment with small settings
    env = SMEEFEnv(grid_size=6, max_steps=60)
    env = Monitor(env, "outputs/logs/ppo/")
    
    # Create directories
    os.makedirs("models/ppo", exist_ok=True)
    os.makedirs("outputs/metrics/ppo", exist_ok=True)
    
    # Hyperparameter combinations (10 runs)
    hyperparams = create_ppo_hyperparameter_combinations()
    results = []
    
    best_reward = -float('inf')
    best_params = None
    
    for i, params in enumerate(hyperparams):
        print(f"\n🏃 PPO Run {i+1}/10 - {params['group']}")
        print(f"   LR: {params['learning_rate']}, Steps: {params['n_steps']}")
        print(f"   Batch: {params['batch_size']}, Epochs: {params['n_epochs']}")
        
        # Create PPO model with current hyperparameters
        model = PPO(
            "MultiInputPolicy",
            env,
            learning_rate=params['learning_rate'],
            n_steps=params['n_steps'],
            batch_size=params['batch_size'],
            n_epochs=params['n_epochs'],
            clip_range=params['clip_range'],
            verbose=0,
        )
        
        
        model.learn(total_timesteps=100000)
        
    
        mean_reward = evaluate_ppo_ultra_fast(model)
        
        
        result = {
            'run_id': i + 1,
            'group': params['group'],
            'learning_rate': params['learning_rate'],
            'n_steps': params['n_steps'],
            'batch_size': params['batch_size'],
            'n_epochs': params['n_epochs'],
            'clip_range': params['clip_range'],
            'mean_reward': mean_reward,
        }
        
        results.append(result)
        
        print(f"   ✅ Reward: {mean_reward:6.2f}")
        
        # Save if best model
        if mean_reward > best_reward:
            best_reward = mean_reward
            best_params = params
            model.save("models/ppo/ppo_best_model")
            print("   💫 NEW BEST MODEL!")
    
    # Quick analysis
    analyze_ppo_results_fast(results, best_params, best_reward)
    
    print(f"\n🎉 ULTRA-FAST PPO Complete! (~1-2 minutes)")
    print(f"🏆 Best Model: {best_params['group']}")
    print(f"📈 Best Mean Reward: {best_reward:.2f}")
    
    env.close()
    return results

def evaluate_ppo_ultra_fast(model, n_episodes=5):
    """Ultra-fast evaluation for PPO"""
    eval_env = SMEEFEnv(grid_size=6, max_steps=60)
    total_reward = 0
    
    for _ in range(n_episodes):
        obs, _ = eval_env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        total_reward += episode_reward
    
    eval_env.close()
    return total_reward / n_episodes

def analyze_ppo_results_fast(results, best_params, best_reward):
    """Quick analysis for PPO results"""
    
    # Simple bar plot
    plt.figure(figsize=(12, 6))
    
    groups = [r['group'] for r in results]
    rewards = [r['mean_reward'] for r in results]
    
    # Color by learning rate
    colors = []
    for result in results:
        lr = result['learning_rate']
        if lr >= 0.001:
            colors.append('#ff6b6b')  # Red for high LR
        elif lr >= 0.0003:
            colors.append('#4ecdc4')  # Teal for medium LR
        else:
            colors.append('#45b7d1')  # Blue for low LR
    
    bars = plt.bar(range(len(results)), rewards, color=colors, alpha=0.8)
    plt.xlabel('Configuration')
    plt.ylabel('Mean Reward')
    plt.title('PPO Ultra-Fast Hyperparameter Tuning\n(Red=High LR, Teal=Medium LR, Blue=Low LR)')
    plt.xticks(range(len(results)), groups, rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(rewards):
        plt.text(i, v + 0.5, f'{v:.1f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('outputs/plots/ppo_ultra_fast_results.png', dpi=200, bbox_inches='tight')
    plt.show()
    
    # Quick text analysis
    print(f"\n🧪 PPO QUICK ANALYSIS")
    print("=" * 40)
    
    # Top 3
    top_3 = sorted(results, key=lambda x: x['mean_reward'], reverse=True)[:3]
    print("🏅 TOP 3 CONFIGURATIONS:")
    for i, result in enumerate(top_3):
        print(f"  {i+1}. {result['group']}: {result['mean_reward']:.2f}")
    
    # Learning rate analysis
    lr_performance = {}
    for result in results:
        lr = result['learning_rate']
        if lr not in lr_performance:
            lr_performance[lr] = []
        lr_performance[lr].append(result['mean_reward'])
    
    print("\n📊 Learning Rate Performance:")
    for lr, rewards in sorted(lr_performance.items()):
        avg = np.mean(rewards)
        print(f"  LR {lr}: {avg:.2f}")

# Keep your original simple version for quick testing
def train_ppo_simple():
    """Simple PPO test - your original version"""
    print("🚀 Starting Simple PPO Training...")

    os.makedirs("models/ppo", exist_ok=True)
    os.makedirs("outputs/metrics/ppo", exist_ok=True)
    
    env = SMEEFEnv(grid_size=6, max_steps=50)
    
    model = PPO(
        "MultiInputPolicy",
        env,
        learning_rate=0.0003,
        n_steps=128,
        batch_size=32,
        verbose=0,
    )
    
    model.learn(total_timesteps=100000)  
    model.save("models/ppo/ppo_simple_model")
    print("✅ Simple PPO training complete!")
    
    env.close()

if __name__ == "__main__":
          
    train_ppo_ultra_fast() 