import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import shutil

# Import gymnasium instead of gym
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env

# Create a complete test environment using Gymnasium
class SMEEFEnv(gym.Env):
    """
    Simple Multi-Entity Environment Framework (SMEEF)
    A grid-based environment for testing RL algorithms
    """
    def __init__(self, grid_size=6, max_steps=60):
        super().__init__()
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.current_step = 0
        
        # Define observation space (grid observation)
        self.observation_space = spaces.Dict({
            'observation': spaces.Box(low=0, high=1, shape=(grid_size, grid_size), dtype=np.float32)
        })
        
        # Define action space (4 directions)
        self.action_space = spaces.Discrete(4)
        
        # Environment state
        self.agent_pos = None
        self.target_pos = None
        self.obstacles = None
        
    def reset(self, seed=None, options=None):
        # Gymnasium uses super().reset(seed=seed)
        super().reset(seed=seed)
        self.current_step = 0
        
        # Initialize agent at random position
        self.agent_pos = np.array([self.np_random.integers(0, self.grid_size), 
                                 self.np_random.integers(0, self.grid_size)])
        
        # Initialize target at random position (different from agent)
        self.target_pos = self.agent_pos.copy()
        while np.array_equal(self.target_pos, self.agent_pos):
            self.target_pos = np.array([self.np_random.integers(0, self.grid_size), 
                                      self.np_random.integers(0, self.grid_size)])
        
        # Create some random obstacles
        self.obstacles = []
        for _ in range(3):
            obs_pos = np.array([self.np_random.integers(0, self.grid_size), 
                              self.np_random.integers(0, self.grid_size)])
            if not (np.array_equal(obs_pos, self.agent_pos) or 
                    np.array_equal(obs_pos, self.target_pos)):
                self.obstacles.append(obs_pos)
        
        # Create observation grid
        obs_grid = self._create_observation_grid()
        
        return obs_grid, {}
    
    def _create_observation_grid(self):
        """Create the observation grid with agent, target, and obstacles"""
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        
        # Mark agent position (value 1.0)
        grid[self.agent_pos[0], self.agent_pos[1]] = 1.0
        
        # Mark target position (value 0.8)
        grid[self.target_pos[0], self.target_pos[1]] = 0.8
        
        # Mark obstacles (value 0.3)
        for obs in self.obstacles:
            grid[obs[0], obs[1]] = 0.3
            
        return {'observation': grid}
    
    def step(self, action):
        self.current_step += 1
        
        # Move agent based on action
        new_pos = self.agent_pos.copy()
        if action == 0:  # Up
            new_pos[0] = max(0, new_pos[0] - 1)
        elif action == 1:  # Right
            new_pos[1] = min(self.grid_size - 1, new_pos[1] + 1)
        elif action == 2:  # Down
            new_pos[0] = min(self.grid_size - 1, new_pos[0] + 1)
        elif action == 3:  # Left
            new_pos[1] = max(0, new_pos[1] - 1)
        
        # Check if new position is valid (not an obstacle)
        valid_move = True
        for obs in self.obstacles:
            if np.array_equal(new_pos, obs):
                valid_move = False
                break
        
        if valid_move:
            self.agent_pos = new_pos
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check termination conditions
        terminated = np.array_equal(self.agent_pos, self.target_pos)
        truncated = self.current_step >= self.max_steps
        
        # Create new observation
        obs = self._create_observation_grid()
        
        return obs, reward, terminated, truncated, {}
    
    def _calculate_reward(self):
        """Calculate reward based on agent's state"""
        # Distance to target (negative reward for being far)
        distance = np.linalg.norm(self.agent_pos - self.target_pos)
        distance_reward = -distance * 0.1
        
        # Step penalty
        step_penalty = -0.01
        
        # Success bonus
        success_bonus = 10.0 if np.array_equal(self.agent_pos, self.target_pos) else 0.0
        
        # Obstacle penalty
        obstacle_penalty = 0.0
        for obs in self.obstacles:
            if np.array_equal(self.agent_pos, obs):
                obstacle_penalty = -2.0
                break
        
        total_reward = distance_reward + step_penalty + success_bonus + obstacle_penalty
        return total_reward
    
    def render(self):
        """Simple text-based rendering"""
        grid = np.full((self.grid_size, self.grid_size), '.', dtype=str)
        
        # Mark target
        grid[self.target_pos[0], self.target_pos[1]] = 'T'
        
        # Mark obstacles
        for obs in self.obstacles:
            grid[obs[0], obs[1]] = 'X'
        
        # Mark agent
        grid[self.agent_pos[0], self.agent_pos[1]] = 'A'
        
        print(f"Step: {self.current_step}/{self.max_steps}")
        for row in grid:
            print(' '.join(row))
        print()
        
    def close(self):
        pass

def setup_directories():
    """Create all necessary directories"""
    directories = [
        "models/ppo",
        "outputs/logs/ppo", 
        "outputs/metrics/ppo",
        "outputs/plots"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"📁 Created directory: {directory}")

def create_ppo_hyperparameter_combinations():
    """Generate PPO hyperparameter combinations based on previous results"""
    return [
        # Learning Rate Variations (with analysis columns)
        {
            'learning_rate': 0.001, 'n_steps': 64, 'batch_size': 16, 'n_epochs': 3, 'clip_range': 0.2, 
            'group': 'High LR', 'expected_reward': -31.45, 'notes': 'Initial high LR run',
            'target_improvement': 'Reduce LR for better stability'
        },
        {
            'learning_rate': 0.0003, 'n_steps': 64, 'batch_size': 16, 'n_epochs': 3, 'clip_range': 0.2, 
            'group': 'Standard LR', 'expected_reward': -24.40, 'notes': 'New best model',
            'target_improvement': 'Baseline configuration'
        },
        {
            'learning_rate': 0.0001, 'n_steps': 64, 'batch_size': 16, 'n_epochs': 3, 'clip_range': 0.2, 
            'group': 'Low LR', 'expected_reward': -24.60, 'notes': 'Slightly worse than Run 2',
            'target_improvement': 'May need more timesteps to converge'
        },
        
        # Step Variations
        {
            'learning_rate': 0.0003, 'n_steps': 32, 'batch_size': 16, 'n_epochs': 3, 'clip_range': 0.2, 
            'group': 'Short Steps', 'expected_reward': -24.60, 'notes': 'Short steps, same reward as Run 3',
            'target_improvement': 'More frequent updates but less data per update'
        },
        {
            'learning_rate': 0.0003, 'n_steps': 128, 'batch_size': 16, 'n_epochs': 3, 'clip_range': 0.2, 
            'group': 'Long Steps', 'expected_reward': -24.40, 'notes': 'Matches Run 2',
            'target_improvement': 'More data per update but less frequent'
        },
        
        # Batch Size Variations
        {
            'learning_rate': 0.0003, 'n_steps': 64, 'batch_size': 8, 'n_epochs': 3, 'clip_range': 0.2, 
            'group': 'Small Batch', 'expected_reward': -24.40, 'notes': 'Small batch, same as Run 2',
            'target_improvement': 'More noisy updates but faster'
        },
        {
            'learning_rate': 0.0003, 'n_steps': 64, 'batch_size': 32, 'n_epochs': 3, 'clip_range': 0.2, 
            'group': 'Large Batch', 'expected_reward': -34.55, 'notes': 'Large batch has worse performance',
            'target_improvement': 'Reduce batch size for better performance'
        },
        
        # Epoch Variations
        {
            'learning_rate': 0.0003, 'n_steps': 64, 'batch_size': 16, 'n_epochs': 2, 'clip_range': 0.2, 
            'group': 'Few Epochs', 'expected_reward': -25.45, 'notes': 'Fewer epochs, slightly worse',
            'target_improvement': 'Increase epochs for better optimization'
        },
        {
            'learning_rate': 0.0003, 'n_steps': 64, 'batch_size': 16, 'n_epochs': 5, 'clip_range': 0.2, 
            'group': 'More Epochs', 'expected_reward': -24.20, 'notes': 'More epochs, good performance',
            'target_improvement': 'Good balance of computation and performance'
        },
        
        # Clip Range Variations
        {
            'learning_rate': 0.0003, 'n_steps': 64, 'batch_size': 16, 'n_epochs': 3, 'clip_range': 0.1, 
            'group': 'Tight Clip', 'expected_reward': -24.00, 'notes': 'New best overall',
            'target_improvement': 'Conservative updates work best'
        },
        {
            'learning_rate': 0.0003, 'n_steps': 64, 'batch_size': 16, 'n_epochs': 3, 'clip_range': 0.3, 
            'group': 'Loose Clip', 'expected_reward': -25.00, 'notes': 'Test loose clipping',
            'target_improvement': 'May allow too large policy changes'
        },
    ]

def train_ppo_ultra_fast():
    print("⚡ Starting ULTRA-FAST PPO Hyperparameter Tuning...")
    print("📊 Using previous results as baseline expectations")
    print("=" * 60)
    
    # Setup directories first
    setup_directories()
    
    # Create environment with small settings
    env = SMEEFEnv(grid_size=6, max_steps=60)
    
    # Check environment compatibility
    try:
        check_env(env)
        print("✅ Environment passed compatibility check")
    except Exception as e:
        print(f"⚠️  Environment check warning: {e}")
    
    # Use Monitor with the environment
    env = Monitor(env, "outputs/logs/ppo/")
    
    # Hyperparameter combinations
    hyperparams = create_ppo_hyperparameter_combinations()
    results = []
    
    best_reward = -float('inf')
    best_params = None
    best_model = None
    global_best_reward = -float('inf')
    
    print(f"\n🎯 Expected reward range: {-34.55:.2f} (Run 7) to {-24.00:.2f} (Run 10)")
    
    for i, params in enumerate(hyperparams):
        print(f"\n{'='*60}")
        print(f"🏃 PPO Run {i+1}/10 - {params['group']}")
        print(f"   Start: {pd.Timestamp.now().strftime('%H:%M:%S')}")
        print(f"   LR: {params['learning_rate']}, Steps: {params['n_steps']}")
        print(f"   Batch: {params['batch_size']}, Epochs: {params['n_epochs']}")
        print(f"   Clip: {params['clip_range']}")
        print(f"   📈 Expected: {params['expected_reward']:.2f}")
        print(f"   💡 Notes: {params['notes']}")
        print(f"{'='*60}")
        
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
            device='cpu'
        )
        
        # Train with progress updates
        total_timesteps = 5000
        model.learn(total_timesteps=total_timesteps)
        
        # Evaluate
        mean_reward, std_reward = evaluate_ppo_ultra_fast(model)
        
        # Calculate performance metrics
        expected_reward = params['expected_reward']
        reward_difference = mean_reward - expected_reward
        improvement_status = "✅ EXCEEDED" if mean_reward > expected_reward else "⚠️ BELOW" if mean_reward < expected_reward else "🎯 MATCHED"
        
        result = {
            'run_id': i + 1,
            'group': params['group'],
            'learning_rate': params['learning_rate'],
            'n_steps': params['n_steps'],
            'batch_size': params['batch_size'],
            'n_epochs': params['n_epochs'],
            'clip_range': params['clip_range'],
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            'expected_reward': expected_reward,
            'reward_difference': reward_difference,
            'improvement_status': improvement_status,
            'notes': params['notes'],
            'target_improvement': params['target_improvement'],
        }
        
        results.append(result)
        
        print(f"   ✅ Actual Reward: {mean_reward:6.2f} ± {std_reward:.2f}")
        print(f"   🎯 Expected: {expected_reward:6.2f}")
        print(f"   📊 Difference: {reward_difference:+.2f}")
        print(f"   📈 Status: {improvement_status}")
        
        # Save individual model with full path
        model_path = f"models/ppo/ppo_run_{i+1}"
        model.save(model_path)
        print(f"   💾 Model saved: {model_path}.zip")
        
        # Save if best model in this session
        if mean_reward > best_reward:
            best_reward = mean_reward
            best_params = params
            best_model = model
            best_model_path = "models/ppo/ppo_best_model"
            best_model.save(best_model_path)
            print("   💫 NEW BEST MODEL IN THIS SESSION!")
            print(f"   💾 Best model saved: {best_model_path}.zip")
        
        # Check for global best (across all sessions)
        if mean_reward > global_best_reward:
            global_best_reward = mean_reward
            global_best_path = "models/ppo/ppo_global_best_model"
            model.save(global_best_path)
            print("   🌟 NEW GLOBAL BEST MODEL!")
            print(f"   💾 Global best model saved: {global_best_path}.zip")
    
    # Verify best model was saved and can be loaded
    if best_model is not None:
        print(f"\n🔍 Verifying best model can be loaded...")
        try:
            # Try to load the best model
            loaded_model = PPO.load("models/ppo/ppo_best_model")
            verification_reward, _ = evaluate_ppo_ultra_fast(loaded_model, n_episodes=2)
            print(f"   ✅ Best model verified! Loaded reward: {verification_reward:.2f}")
        except Exception as e:
            print(f"   ❌ Failed to load best model: {e}")
    
    # Verify global best model
    if global_best_reward > -float('inf'):
        print(f"\n🔍 Verifying global best model can be loaded...")
        try:
            # Try to load the global best model
            loaded_global_model = PPO.load("models/ppo/ppo_global_best_model")
            global_verification_reward, _ = evaluate_ppo_ultra_fast(loaded_global_model, n_episodes=2)
            print(f"   ✅ Global best model verified! Loaded reward: {global_verification_reward:.2f}")
        except Exception as e:
            print(f"   ❌ Failed to load global best model: {e}")
    
    # Comprehensive analysis
    analyze_ppo_results_comprehensive(results, best_params, best_reward, global_best_reward)
    
    print(f"\n🎉 ULTRA-FAST PPO Complete!")
    if best_params:
        print(f"🏆 Best Session Model: {best_params['group']}")
        print(f"📈 Best Session Reward: {best_reward:.2f}")
        print(f"🌍 Global Best Reward: {global_best_reward:.2f}")
    
    # List saved models
    print(f"\n📁 Saved models in models/ppo/:")
    try:
        model_files = os.listdir("models/ppo")
        ppo_files = [f for f in model_files if f.startswith('ppo_') and f.endswith('.zip')]
        for file in sorted(ppo_files):
            file_path = os.path.join("models/ppo", file)
            file_size = os.path.getsize(file_path) / 1024  # Size in KB
            print(f"   📄 {file} ({file_size:.1f} KB)")
            
            # Highlight important models
            if file == "ppo_global_best_model.zip":
                print(f"        ⭐ GLOBAL BEST MODEL")
            elif file == "ppo_best_model.zip":
                print(f"        ✅ SESSION BEST MODEL")
                
    except FileNotFoundError:
        print("   ❌ No models directory found")
    except Exception as e:
        print(f"   ⚠️  Error listing models: {e}")
    
    env.close()
    return results

def evaluate_ppo_ultra_fast(model, n_episodes=5):
    """Ultra-fast evaluation for PPO with statistics"""
    eval_env = SMEEFEnv(grid_size=6, max_steps=60)
    rewards = []
    
    for _ in range(n_episodes):
        obs, _ = eval_env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        rewards.append(episode_reward)
    
    eval_env.close()
    return np.mean(rewards), np.std(rewards)

def analyze_ppo_results_comprehensive(results, best_params, best_reward, global_best_reward):
    """Comprehensive analysis for PPO results with comparison to expectations"""
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(results)
    
    # Save detailed results
    results_path = 'outputs/metrics/ppo/ppo_detailed_results.csv'
    df.to_csv(results_path, index=False)
    
    # Create comprehensive visualization
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Actual vs Expected Rewards
    groups = [r['group'] for r in results]
    actual_rewards = [r['mean_reward'] for r in results]
    expected_rewards = [r['expected_reward'] for r in results]
    
    x = np.arange(len(groups))
    width = 0.35
    
    plt.subplot(2, 2, 1)
    bars1 = plt.bar(x - width/2, actual_rewards, width, label='Actual', color='skyblue', alpha=0.8)
    bars2 = plt.bar(x + width/2, expected_rewards, width, label='Expected', color='lightcoral', alpha=0.8)
    
    plt.xlabel('Configuration')
    plt.ylabel('Mean Reward')
    plt.title('PPO: Actual vs Expected Performance\n(Higher is better)')
    plt.xticks(x, groups, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Performance Differences
    plt.subplot(2, 2, 2)
    differences = [r['reward_difference'] for r in results]
    colors = ['green' if diff > 0 else 'red' for diff in differences]
    
    plt.bar(groups, differences, color=colors, alpha=0.7)
    plt.xlabel('Configuration')
    plt.ylabel('Actual - Expected Reward')
    plt.title('Performance vs Expectations\n(Green = Exceeded, Red = Below)')
    plt.xticks(rotation=45, ha='right')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Learning Rate Analysis
    plt.subplot(2, 2, 3)
    lr_groups = {}
    for result in results:
        lr = result['learning_rate']
        if lr not in lr_groups:
            lr_groups[lr] = []
        lr_groups[lr].append(result['mean_reward'])
    
    lr_means = [np.mean(lr_groups[lr]) for lr in sorted(lr_groups.keys())]
    lr_stds = [np.std(lr_groups[lr]) for lr in sorted(lr_groups.keys())]
    
    plt.bar([f'LR {lr}' for lr in sorted(lr_groups.keys())], lr_means, 
            yerr=lr_stds, capsize=5, alpha=0.7, color=['#ff6b6b', '#4ecdc4', '#45b7d1'])
    plt.xlabel('Learning Rate')
    plt.ylabel('Mean Reward')
    plt.title('Performance by Learning Rate')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Clip Range Analysis
    plt.subplot(2, 2, 4)
    clip_groups = {}
    for result in results:
        clip = result['clip_range']
        if clip not in clip_groups:
            clip_groups[clip] = []
        clip_groups[clip].append(result['mean_reward'])
    
    clip_means = [np.mean(clip_groups[clip]) for clip in sorted(clip_groups.keys())]
    
    plt.bar([f'Clip {clip}' for clip in sorted(clip_groups.keys())], clip_means,
            alpha=0.7, color=['#ff9ff3', '#f368e0', '#a29bfe'])
    plt.xlabel('Clip Range')
    plt.ylabel('Mean Reward')
    plt.title('Performance by Clip Range')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = 'outputs/plots/ppo_comprehensive_analysis.png'
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.show()
    
    # Print comprehensive summary
    print(f"\n{'='*80}")
    print("📊 PPO COMPREHENSIVE RESULTS SUMMARY")
    print(f"{'='*80}")
    
    summary_df = df[['run_id', 'group', 'learning_rate', 'n_steps', 'batch_size', 
                    'mean_reward', 'expected_reward', 'improvement_status']].copy()
    
    print(summary_df.to_string(index=False))
    
    # Key insights
    print(f"\n🔍 KEY INSIGHTS:")
    if not df.empty:
        best_run = df.loc[df['mean_reward'].idxmax()]
        print(f"   🏆 Best Configuration: Run {best_run['run_id']} - {best_run['group']}")
        print(f"   📈 Best Session Reward: {best_reward:.2f}")
        print(f"   🌍 Global Best Reward: {global_best_reward:.2f}")
        print(f"   ✅ Configurations that exceeded expectations: {len(df[df['improvement_status'] == '✅ EXCEEDED'])}/10")
        print(f"   ⚠️  Configurations below expectations: {len(df[df['improvement_status'] == '⚠️ BELOW'])}/10")
        
        # Learning rate recommendation
        print(f"   🎯 Recommended Learning Rate: {best_run['learning_rate']}")
        
        # Clip range recommendation
        print(f"   🎯 Recommended Clip Range: {best_run['clip_range']}")
    
    print(f"\n💾 Results saved to:")
    print(f"   📄 CSV: {results_path}")
    print(f"   📊 Plot: {plot_path}")
    print(f"   🤖 Models: models/ppo/")
    print(f"        ⭐ ppo_global_best_model.zip - Best across all sessions")
    print(f"        ✅ ppo_best_model.zip - Best in this session")
    print(f"        📁 ppo_run_*.zip - Individual run models")

if __name__ == "__main__":
    train_ppo_ultra_fast()