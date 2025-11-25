import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import shutil
from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from environment.smeef_env import SMEEFEnv
from environment.obs_wrappers import NormalizeFlattenObs

def create_a2c_hyperparameter_combinations():
    """Generate A2C hyperparameter combinations EXACTLY matching your report data"""
    return [
        # EXACT configurations from your report table
        {'run_id': 1, 'learning_rate': 0.001, 'gamma': 0.99, 'n_steps': 64, 'ent_coef': 0.01, 'vf_coef': 0.5, 'group': 'High_LR', 'target_reward': -33.45},
        {'run_id': 2, 'learning_rate': 0.0007, 'gamma': 0.99, 'n_steps': 64, 'ent_coef': 0.01, 'vf_coef': 0.5, 'group': 'Medium_LR', 'target_reward': -27.35},
        {'run_id': 3, 'learning_rate': 0.0003, 'gamma': 0.99, 'n_steps': 64, 'ent_coef': 0.01, 'vf_coef': 0.5, 'group': 'Low_LR', 'target_reward': -34.55},
        {'run_id': 4, 'learning_rate': 0.0007, 'gamma': 0.99, 'n_steps': 32, 'ent_coef': 0.01, 'vf_coef': 0.5, 'group': 'Short_Steps', 'target_reward': -25.80},
        {'run_id': 5, 'learning_rate': 0.0007, 'gamma': 0.99, 'n_steps': 128, 'ent_coef': 0.01, 'vf_coef': 0.5, 'group': 'Long_Steps', 'target_reward': -32.10},
        {'run_id': 6, 'learning_rate': 0.0007, 'gamma': 0.99, 'n_steps': 64, 'ent_coef': 0.001, 'vf_coef': 0.5, 'group': 'Low_Entropy', 'target_reward': -34.55},
        {'run_id': 7, 'learning_rate': 0.0007, 'gamma': 0.99, 'n_steps': 64, 'ent_coef': 0.1, 'vf_coef': 0.5, 'group': 'High_Entropy', 'target_reward': -25.80},
        {'run_id': 8, 'learning_rate': 0.0007, 'gamma': 0.99, 'n_steps': 64, 'ent_coef': 0.01, 'vf_coef': 0.25, 'group': 'Best_Model', 'target_reward': -24.40},
        {'run_id': 9, 'learning_rate': 0.0007, 'gamma': 0.99, 'n_steps': 64, 'ent_coef': 0.01, 'vf_coef': 0.75, 'group': 'High_VF_Coef', 'target_reward': -30.35},
        {'run_id': 10, 'learning_rate': 0.001, 'gamma': 0.99, 'n_steps': 128, 'ent_coef': 0.01, 'vf_coef': 0.5, 'group': 'Aggressive', 'target_reward': -34.35},
    ]

def train_a2c_with_tracking():
    print("🎯 Starting A2C Hyperparameter Tuning - Matching Report Data Exactly...")
    
    # Create directories
    os.makedirs("models/a2c", exist_ok=True)
    os.makedirs("outputs/plots/a2c", exist_ok=True)
    os.makedirs("outputs/metrics/a2c", exist_ok=True)
    
    # Create environment - use MlpPolicy instead of MultiInputPolicy for Box spaces
    env = NormalizeFlattenObs(SMEEFEnv(grid_size=6, max_steps=60))
    env = Monitor(env, "outputs/logs/a2c/")
    
    # Hyperparameter combinations EXACTLY matching your report
    hyperparams = create_a2c_hyperparameter_combinations()
    all_results = []
    
    global_best_reward = -float('inf')
    global_best_path = "models/a2c/global_best_model"
    global_best_params = None
    global_best_result = None
    
    for params in hyperparams:
        print(f"\n{'='*60}")
        print(f"🏃 A2C Run {params['run_id']}/10 - {params['group']}")
        print(f"   LR: {params['learning_rate']}, Gamma: {params['gamma']}, Steps: {params['n_steps']}")
        print(f"   Entropy: {params['ent_coef']}, VF Coef: {params['vf_coef']}")
        print(f"   Target from report: {params['target_reward']:.2f}")
        print(f"{'='*60}")
        
        try:
            # Create A2C model with EXACT hyperparameters from your report
            # Use "MlpPolicy" instead of "MultiInputPolicy" for Box observation spaces
            model = A2C(
                "MlpPolicy",  # CHANGED: Use MlpPolicy for flattened Box spaces
                env,
                learning_rate=params['learning_rate'],
                gamma=params['gamma'],
                n_steps=params['n_steps'],
                ent_coef=params['ent_coef'],
                vf_coef=params['vf_coef'],
                verbose=0,
            )
            
            # Train with sufficient timesteps for stable results
            model.learn(total_timesteps=20000)
            
            # Comprehensive evaluation
            mean_reward, std_reward = evaluate_a2c_comprehensive(model)
            
            # Save per-run model
            run_model_path = f"models/a2c/a2c_run_{params['run_id']}_{params['group']}"
            model.save(run_model_path)
            
            # Store detailed results
            result = {
                'run_id': params['run_id'],
                'group': params['group'],
                'learning_rate': params['learning_rate'],
                'gamma': params['gamma'],
                'n_steps': params['n_steps'],
                'ent_coef': params['ent_coef'],
                'vf_coef': params['vf_coef'],
                'mean_reward': mean_reward,
                'std_reward': std_reward,
                'target_reward': params['target_reward'],
                'model_path': run_model_path,
            }
            
            all_results.append(result)
            
            print(f"   ✅ Mean Reward: {mean_reward:6.2f} ± {std_reward:.2f}")
            print(f"   🎯 Target: {params['target_reward']:6.2f}")
            print(f"   📁 Model saved: {run_model_path}.zip")
            
            # Save global best model (like REINFORCE)
            if mean_reward > global_best_reward:
                global_best_reward = mean_reward
                global_best_params = params
                global_best_result = result
                # Copy the best model to global best path
                shutil.copy(run_model_path + ".zip", global_best_path + ".zip")
                print(f"   💫 NEW GLOBAL BEST MODEL!")
                print(f"      Saved to: {global_best_path}.zip")
                print(f"      Configuration: {params['group']}")
                
        except Exception as e:
            print(f"❌ Error in run {params['group']}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Only run analysis if we have successful results
    if all_results and global_best_params is not None:
        analyze_a2c_results_comprehensive(all_results, global_best_params, global_best_reward, global_best_result)
        
        print(f"\n🎉 A2C Hyperparameter Tuning Complete!")
        print(f"🏆 Global Best Model: Run {global_best_params['run_id']} - {global_best_params['group']}")
        print(f"📈 Best Mean Reward: {global_best_reward:.2f}")
        print(f"🎯 Target from report: {global_best_params['target_reward']:.2f}")
        print(f"💾 Global best saved to: {global_best_path}.zip")
    else:
        print(f"\n❌ No successful A2C runs completed!")
    
    env.close()
    return all_results

def evaluate_a2c_comprehensive(model, n_episodes=15):
    """Comprehensive evaluation for A2C"""
    eval_env = NormalizeFlattenObs(SMEEFEnv(grid_size=6, max_steps=60))
    
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

def analyze_a2c_results_comprehensive(results, best_params, best_reward, best_result):
    """Comprehensive analysis for A2C results matching your report format"""
    
    # Sort by run_id for consistent ordering
    results_sorted = sorted(results, key=lambda x: x['run_id'])
    
    # Create comprehensive plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Extract data
    run_ids = [f"Run {r['run_id']}" for r in results_sorted]
    groups = [r['group'] for r in results_sorted]
    mean_rewards = [r['mean_reward'] for r in results_sorted]
    target_rewards = [r['target_reward'] for r in results_sorted]
    learning_rates = [r['learning_rate'] for r in results_sorted]
    n_steps_list = [r['n_steps'] for r in results_sorted]
    ent_coefs = [r['ent_coef'] for r in results_sorted]
    vf_coefs = [r['vf_coef'] for r in results_sorted]
    
    # Plot 1: Actual vs Target rewards (bar chart)
    x_pos = np.arange(len(run_ids))
    width = 0.35
    
    bars1 = ax1.bar(x_pos - width/2, mean_rewards, width, label='Actual', alpha=0.7, color='blue')
    bars2 = ax1.bar(x_pos + width/2, target_rewards, width, label='Target', alpha=0.7, color='red')
    
    ax1.set_xlabel('Run')
    ax1.set_ylabel('Mean Reward')
    ax1.set_title('A2C: Actual vs Target Mean Rewards\n(Matches Your Report Data)')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(run_ids, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (actual, target) in enumerate(zip(mean_rewards, target_rewards)):
        ax1.text(i - width/2, actual + 0.5, f'{actual:.1f}', ha='center', va='bottom', fontsize=8)
        ax1.text(i + width/2, target + 0.5, f'{target:.1f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Learning rate vs performance (scatter)
    scatter = ax2.scatter(learning_rates, mean_rewards, c=n_steps_list, 
                         s=[ec*500 for ec in ent_coefs], alpha=0.7, cmap='viridis')
    ax2.set_xscale('log')
    ax2.set_xlabel('Learning Rate')
    ax2.set_ylabel('Mean Reward')
    ax2.set_title('A2C: Learning Rate vs Performance\n(Color=N Steps, Size=Entropy Coef)')
    plt.colorbar(scatter, ax=ax2, label='N Steps')
    ax2.grid(True, alpha=0.3)
    
    # Highlight Run 8 (your best from report)
    best_idx = next(i for i, r in enumerate(results_sorted) if r['run_id'] == 8)
    ax2.scatter(learning_rates[best_idx], mean_rewards[best_idx], 
               s=200, facecolors='none', edgecolors='red', linewidth=3, label='Run 8 (Report Best)')
    
    # Highlight actual best run
    actual_best_idx = np.argmax(mean_rewards)
    ax2.scatter(learning_rates[actual_best_idx], mean_rewards[actual_best_idx], 
               s=200, facecolors='none', edgecolors='green', linewidth=2, label='Actual Best')
    ax2.legend()
    
    # Plot 3: Configuration groups comparison
    colors = plt.cm.Set3(np.linspace(0, 1, len(groups)))
    bars = ax3.bar(groups, mean_rewards, color=colors, alpha=0.7)
    ax3.set_xlabel('Configuration Group')
    ax3.set_ylabel('Mean Reward')
    ax3.set_title('A2C: Performance by Configuration Group')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(mean_rewards):
        ax3.text(i, v + 0.5, f'{v:.1f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 4: Performance difference from target
    differences = [actual - target for actual, target in zip(mean_rewards, target_rewards)]
    bars = ax4.bar(run_ids, differences, color=['green' if d >= 0 else 'red' for d in differences], alpha=0.7)
    ax4.set_xlabel('Run')
    ax4.set_ylabel('Difference (Actual - Target)')
    ax4.set_title('A2C: Performance vs Target Expectations\n(Green=Better, Red=Worse)')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)
    
    # Add value labels to difference plot
    for i, v in enumerate(differences):
        ax4.text(i, v + (0.3 if v >= 0 else -0.5), f'{v:+.1f}', 
                ha='center', va='bottom' if v >= 0 else 'top', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('outputs/plots/a2c/a2c_comprehensive_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print comprehensive summary in EXACT table format from your report
    print(f"\n{'='*100}")
    print("📊 A2C HYPERPARAMETER RESULTS - MATCHING YOUR REPORT FORMAT")
    print(f"{'='*100}")
    
    print(f"{'Run':<4} {'Learning Rate':<14} {'Gamma':<8} {'N Steps':<10} {'Ent Coef':<10} {'VF Coef':<10} {'Mean Reward':<12} {'Target':<10} {'Group':<15}")
    print(f"{'-'*100}")
    
    for res in results_sorted:
        print(f"{res['run_id']:<4} {res['learning_rate']:<14} {res['gamma']:<8} {res['n_steps']:<10} "
              f"{res['ent_coef']:<10} {res['vf_coef']:<10} {res['mean_reward']:<12.2f} "
              f"{res['target_reward']:<10.2f} {res['group']:<15}")
    
    # Find Run 8 (your reported best)
    run_8_result = next((r for r in results_sorted if r['run_id'] == 8), None)
    
    print(f"\n{'='*100}")
    print("🏆 PERFORMANCE SUMMARY")
    print(f"{'='*100}")
    
    if run_8_result:
        print(f"📋 YOUR REPORT BEST: Run 8")
        print(f"   Mean Reward: {run_8_result['mean_reward']:.2f} (Target: {run_8_result['target_reward']:.2f})")
        print(f"   Hyperparameters: LR={run_8_result['learning_rate']}, γ={run_8_result['gamma']}, "
              f"Steps={run_8_result['n_steps']}, Entropy={run_8_result['ent_coef']}, VF={run_8_result['vf_coef']}")
        print(f"   Group: {run_8_result['group']}")
    
    print(f"\n🎯 ACTUAL BEST: Run {best_result['run_id']} - {best_result['group']}")
    print(f"   Mean Reward: {best_reward:.2f} (Target: {best_result['target_reward']:.2f})")
    print(f"   Hyperparameters: LR={best_result['learning_rate']}, γ={best_result['gamma']}, "
          f"Steps={best_result['n_steps']}, Entropy={best_result['ent_coef']}, VF={best_result['vf_coef']}")
    
    # Save detailed results to CSV
    df_data = []
    for res in results_sorted:
        df_data.append({
            'run_id': res['run_id'],
            'group': res['group'],
            'learning_rate': res['learning_rate'],
            'gamma': res['gamma'],
            'n_steps': res['n_steps'],
            'ent_coef': res['ent_coef'],
            'vf_coef': res['vf_coef'],
            'mean_reward': res['mean_reward'],
            'std_reward': res['std_reward'],
            'target_reward': res['target_reward'],
            'difference': res['mean_reward'] - res['target_reward'],
            'model_path': res['model_path']
        })
    
    df = pd.DataFrame(df_data)
    csv_path = "outputs/metrics/a2c/a2c_results.csv"
    df.to_csv(csv_path, index=False)
    
    # Save numpy format
    np.save("outputs/metrics/a2c/a2c_results.npy", results_sorted)
    
    print(f"\n💾 Results saved to:")
    print(f"   CSV: {csv_path}")
    print(f"   NPY: outputs/metrics/a2c/a2c_results.npy")
    print(f"   Plots: outputs/plots/a2c/")
    print(f"   Models: models/a2c/")

if __name__ == "__main__":
    # Run comprehensive A2C training with exact report data
    results = train_a2c_with_tracking()