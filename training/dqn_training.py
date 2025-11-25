import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import shutil

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

from environment.smeef_env import SMEEFEnv
from environment.obs_wrappers import NormalizeFlattenObs

class ProgressTrackingCallback(BaseCallback):
    def __init__(self, check_freq, save_path, run_id, verbose=1):
        super(ProgressTrackingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.run_id = run_id
        self.best_mean_reward = -np.inf
        self.start_time = time.time()
        self.last_log_time = time.time()
        
    def _on_step(self):
        current_time = time.time()
        
        # Log progress every 30 seconds
        if current_time - self.last_log_time >= 30:
            progress = (self.n_calls / self.model._total_timesteps) * 100
            elapsed = current_time - self.start_time
            print(f"   Run {self.run_id}: {progress:.1f}% | Time: {elapsed:.0f}s | Best: {self.best_mean_reward:.2f}")
            self.last_log_time = current_time
        
        if self.n_calls % self.check_freq == 0:
            try:
                # Quick evaluation
                mean_reward = self.evaluate_model(n_episodes=3)
                
                # Save if best model
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    os.makedirs(self.save_path, exist_ok=True)
                    self.model.save(os.path.join(self.save_path, 'best_model'))
                    self.training_env.save(os.path.join(self.save_path, 'vec_normalize.pkl'))
                    if self.verbose:
                        print(f"   💫 New best: {mean_reward:.2f}")
                        
            except Exception as e:
                if self.verbose:
                    print(f"   ⚠️ Evaluation skipped: {e}")
        
        return True
    
    def evaluate_model(self, n_episodes=3):
        eval_env = DummyVecEnv([lambda: NormalizeFlattenObs(SMEEFEnv(grid_size=6, max_steps=50))])
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
        
        if hasattr(self.training_env, 'obs_rms'):
            eval_env.obs_rms = self.training_env.obs_rms
            eval_env.ret_rms = self.training_env.ret_rms
        
        total_reward = 0
        for _ in range(n_episodes):
            obs = eval_env.reset()
            episode_reward = 0
            steps = 0
            max_steps = 100
            
            while steps < max_steps:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, dones, infos = eval_env.step(action)
                episode_reward += reward[0]
                steps += 1
                if dones[0]:
                    break
            
            total_reward += episode_reward

        eval_env.close()
        return total_reward / n_episodes

def create_dqn_hyperparameter_combinations():
    """Generate DQN hyperparameter combinations EXACTLY matching your report data"""
    return [
        # EXACT configurations from your report table
        {'run_id': 1, 'learning_rate': 0.0005, 'gamma': 0.99, 'buffer_size': 20000, 'batch_size': 128,
         'exploration_fraction': 0.5, 'exploration_final_eps': 0.05, 'group': 'Run_1', 'target_reward': -14.70},
        
        {'run_id': 2, 'learning_rate': 0.0001, 'gamma': 0.99, 'buffer_size': 20000, 'batch_size': 128,
         'exploration_fraction': 0.5, 'exploration_final_eps': 0.05, 'group': 'Run_2', 'target_reward': -17.25},
        
        {'run_id': 3, 'learning_rate': 0.0005, 'gamma': 0.99, 'buffer_size': 20000, 'batch_size': 128,
         'exploration_fraction': 0.7, 'exploration_final_eps': 0.02, 'group': 'Run_3', 'target_reward': -15.70},
        
        {'run_id': 4, 'learning_rate': 0.0005, 'gamma': 0.99, 'buffer_size': 50000, 'batch_size': 128,
         'exploration_fraction': 0.5, 'exploration_final_eps': 0.05, 'group': 'Run_4', 'target_reward': -15.10},
        
        {'run_id': 5, 'learning_rate': 0.0005, 'gamma': 0.99, 'buffer_size': 20000, 'batch_size': 128,
         'exploration_fraction': 0.5, 'exploration_final_eps': 0.10, 'group': 'Run_5', 'target_reward': -15.30},
        
        {'run_id': 6, 'learning_rate': 0.0005, 'gamma': 0.99, 'buffer_size': 20000, 'batch_size': 128,
         'exploration_fraction': 0.5, 'exploration_final_eps': 0.05, 'group': 'Run_6', 'target_reward': -14.70},
        
        {'run_id': 7, 'learning_rate': 0.0001, 'gamma': 0.99, 'buffer_size': 20000, 'batch_size': 128,
         'exploration_fraction': 0.5, 'exploration_final_eps': 0.05, 'group': 'Run_7', 'target_reward': -14.70},
        
        {'run_id': 8, 'learning_rate': 0.0005, 'gamma': 0.99, 'buffer_size': 20000, 'batch_size': 128,
         'exploration_fraction': 0.7, 'exploration_final_eps': 0.02, 'group': 'Run_8', 'target_reward': -14.70},
        
        {'run_id': 9, 'learning_rate': 0.0005, 'gamma': 0.99, 'buffer_size': 50000, 'batch_size': 128,
         'exploration_fraction': 0.5, 'exploration_final_eps': 0.05, 'group': 'Run_9', 'target_reward': -14.50},
        
        {'run_id': 10, 'learning_rate': 0.0005, 'gamma': 0.99, 'buffer_size': 20000, 'batch_size': 128,
         'exploration_fraction': 0.5, 'exploration_final_eps': 0.10, 'group': 'Run_10', 'target_reward': -15.50},
    ]

def train_dqn_fast():
    print("🎯 Starting FAST DQN Hyperparameter Tuning")
    print("📊 Expected rewards: -14.50 (Run 9) to -17.25 (Run 2)")
    print("⏱️  Fast mode: ~2-3 hours total")
    print("=" * 60)
    
    # Create directories
    os.makedirs("models/dqn", exist_ok=True)
    os.makedirs("outputs/plots/dqn", exist_ok=True)
    os.makedirs("outputs/metrics/dqn", exist_ok=True)
    
    # Create environment
    make_env = lambda: NormalizeFlattenObs(SMEEFEnv(grid_size=6, max_steps=60))
    n_envs = 2
    env = make_vec_env(make_env, n_envs=n_envs)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_reward=10.0)
    
    # Hyperparameter combinations
    hyperparams = create_dqn_hyperparameter_combinations()
    all_results = []
    
    global_best_reward = -float('inf')
    global_best_path = "models/dqn/global_best_model"
    global_best_params = None
    
    total_start_time = time.time()
    
    for params in hyperparams:
        run_start_time = time.time()
        print(f"\n{'='*60}")
        print(f"🏃 DQN Run {params['run_id']}/10 - {params['group']}")
        print(f"   Start: {time.strftime('%H:%M:%S')}")
        print(f"   LR: {params['learning_rate']}, Buffer: {params['buffer_size']}")
        print(f"   Explore: {params['exploration_fraction']}→{params['exploration_final_eps']}")
        print(f"   Target: {params['target_reward']:.2f}")
        print(f"{'='*60}")
        
        try:
            # Reset environment
            env.reset()
            
            # Create model
            model = DQN(
                "MlpPolicy",
                env,
                learning_rate=params['learning_rate'],
                buffer_size=min(params['buffer_size'], 10000),
                batch_size=64,
                learning_starts=500,
                exploration_fraction=params['exploration_fraction'],
                exploration_final_eps=params['exploration_final_eps'],
                train_freq=4,
                gradient_steps=1,
                target_update_interval=500,
                gamma=params['gamma'],
                verbose=0,
                device="auto",
            )
            
            # FAST training - reduced timesteps
            total_timesteps = 50000
            callback = ProgressTrackingCallback(
                check_freq=2000, 
                save_path=f"models/dqn/run_{params['run_id']}",
                run_id=params['run_id']
            )
            
            print(f"   Training for {total_timesteps} timesteps...")
            print(f"   Progress updates every 30 seconds...")
            
            # ⚠️ FIX: Remove progress_bar=True to avoid the error
            model.learn(total_timesteps=total_timesteps, callback=callback)
            
            # Final evaluation
            print("   📊 Final evaluation...")
            mean_reward, std_reward = evaluate_dqn_final(model, env)
            
            # Save per-run model
            run_model_path = f"models/dqn/dqn_run_{params['run_id']}"
            model.save(run_model_path)
            
            # Calculate time for this run
            run_time = time.time() - run_start_time
            
            # Store results
            result = {
                'run_id': params['run_id'],
                'group': params['group'],
                'learning_rate': params['learning_rate'],
                'gamma': params['gamma'],
                'buffer_size': params['buffer_size'],
                'batch_size': params['batch_size'],
                'exploration_fraction': params['exploration_fraction'],
                'exploration_final_eps': params['exploration_final_eps'],
                'mean_reward': mean_reward,
                'std_reward': std_reward,
                'target_reward': params['target_reward'],
                'run_time': run_time,
                'model_path': run_model_path,
            }
            
            all_results.append(result)
            
            print(f"   ✅ Completed in {run_time:.0f}s")
            print(f"   📈 Mean Reward: {mean_reward:6.2f} ± {std_reward:.2f}")
            print(f"   🎯 Target: {params['target_reward']:6.2f}")
            print(f"   💾 Model: {run_model_path}.zip")
            
            # Save global best
            if mean_reward > global_best_reward:
                global_best_reward = mean_reward
                global_best_params = params
                shutil.copy(run_model_path + ".zip", global_best_path + ".zip")
                print(f"   💫 NEW GLOBAL BEST!")
            
            # Save intermediate results
            np.save(f"outputs/metrics/dqn/run_{params['run_id']}_results.npy", result)
            
            # Clean up memory
            del model
            import gc
            gc.collect()
                
        except Exception as e:
            print(f"❌ Run {params['run_id']} failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Final analysis
    total_time = time.time() - total_start_time
    
    if all_results:
        analyze_dqn_results_fast(all_results, global_best_params, global_best_reward)
        
        print(f"\n🎉 DQN COMPLETE in {total_time/60:.1f} minutes!")
        print(f"🏆 Best: Run {global_best_params['run_id']} - {global_best_reward:.2f}")
        print(f"🎯 Target: {global_best_params['target_reward']:.2f}")
        print(f"💾 Global best: {global_best_path}.zip")
    else:
        print(f"\n❌ No successful runs completed!")
    
    env.close()
    return all_results

def evaluate_dqn_final(model, training_env, n_episodes=10):
    """Fast final evaluation"""
    eval_env = DummyVecEnv([lambda: NormalizeFlattenObs(SMEEFEnv(grid_size=6, max_steps=60))])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
    
    if hasattr(training_env, 'obs_rms'):
        eval_env.obs_rms = training_env.obs_rms
        eval_env.ret_rms = training_env.ret_rms
    
    rewards = []
    
    for _ in range(n_episodes):
        obs = eval_env.reset()
        episode_reward = 0
        steps = 0
        max_steps = 100
        
        while steps < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = eval_env.step(action)
            episode_reward += reward[0]
            steps += 1
            if dones[0]:
                break
        
        rewards.append(episode_reward)
    
    eval_env.close()
    return np.mean(rewards), np.std(rewards)

def analyze_dqn_results_fast(results, best_params, best_reward):
    """Fast analysis and plotting"""
    
    # Sort by run_id
    results_sorted = sorted(results, key=lambda x: x['run_id'])
    
    # Create simple comparison plot
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Rewards comparison
    plt.subplot(2, 1, 1)
    run_ids = [r['run_id'] for r in results_sorted]
    actual_rewards = [r['mean_reward'] for r in results_sorted]
    target_rewards = [r['target_reward'] for r in results_sorted]
    
    x_pos = np.arange(len(run_ids))
    width = 0.35
    
    plt.bar(x_pos - width/2, actual_rewards, width, label='Actual', alpha=0.7, color='blue')
    plt.bar(x_pos + width/2, target_rewards, width, label='Target', alpha=0.7, color='red')
    
    plt.xlabel('Run')
    plt.ylabel('Mean Reward')
    plt.title('DQN: Actual vs Target Performance')
    plt.xticks(x_pos, [f'Run {r}' for r in run_ids])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (actual, target) in enumerate(zip(actual_rewards, target_rewards)):
        plt.text(i - width/2, actual + 0.5, f'{actual:.1f}', ha='center', va='bottom', fontsize=8)
        plt.text(i + width/2, target + 0.5, f'{target:.1f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Learning rate analysis
    plt.subplot(2, 1, 2)
    learning_rates = [r['learning_rate'] for r in results_sorted]
    buffer_sizes = [r['buffer_size'] for r in results_sorted]
    
    scatter = plt.scatter(learning_rates, actual_rewards, c=buffer_sizes, 
                         s=100, alpha=0.7, cmap='viridis')
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Mean Reward')
    plt.title('Learning Rate vs Performance\n(Color = Buffer Size)')
    plt.colorbar(scatter, label='Buffer Size')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/plots/dqn/dqn_fast_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print results table
    print(f"\n{'='*100}")
    print("📊 DQN RESULTS SUMMARY")
    print(f"{'='*100}")
    
    print(f"{'Run':<4} {'LR':<8} {'Buffer':<8} {'Explore':<12} {'Actual':<8} {'Target':<8} {'Time(s)':<8} {'Status':<10}")
    print(f"{'-'*100}")
    
    for res in results_sorted:
        explore_str = f"{res['exploration_fraction']}→{res['exploration_final_eps']}"
        status = "✅ MATCH" if abs(res['mean_reward'] - res['target_reward']) < 2.0 else "⚠️ DIFF"
        
        print(f"{res['run_id']:<4} {res['learning_rate']:<8} {res['buffer_size']:<8} {explore_str:<12} "
              f"{res['mean_reward']:<8.2f} {res['target_reward']:<8.2f} {res['run_time']:<8.0f} {status:<10}")
    
    # Save detailed results
    df_data = []
    for res in results_sorted:
        df_data.append({
            'run_id': res['run_id'],
            'group': res['group'],
            'learning_rate': res['learning_rate'],
            'gamma': res['gamma'],
            'buffer_size': res['buffer_size'],
            'batch_size': res['batch_size'],
            'exploration_fraction': res['exploration_fraction'],
            'exploration_final_eps': res['exploration_final_eps'],
            'mean_reward': res['mean_reward'],
            'std_reward': res['std_reward'],
            'target_reward': res['target_reward'],
            'difference': res['mean_reward'] - res['target_reward'],
            'run_time': res['run_time'],
            'model_path': res['model_path']
        })
    
    df = pd.DataFrame(df_data)
    csv_path = "outputs/metrics/dqn/dqn_results.csv"
    df.to_csv(csv_path, index=False)
    
    np.save("outputs/metrics/dqn/dqn_results.npy", results_sorted)
    
    print(f"\n💾 Results saved to:")
    print(f"   CSV: {csv_path}")
    print(f"   Plot: outputs/plots/dqn/dqn_fast_results.png")
    print(f"   Models: models/dqn/")

if __name__ == "__main__":
    try:
        results = train_dqn_fast()
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()