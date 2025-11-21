import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Import environment and REINFORCE agent
from environment.smeef_env import SMEEFEnv
from agents.reinforce_agent import REINFORCEAgent

# -----------------------------
# Helper function to flatten dict observations
# -----------------------------
def flatten_obs(obs):
    if isinstance(obs, dict):
        return np.concatenate([v.flatten() for v in obs.values()])
    else:
        return np.array(obs)

# -----------------------------
# Evaluate REINFORCE model
# -----------------------------
def evaluate_reinforce_model(model_path, n_episodes=5):
    env = SMEEFEnv(grid_size=6, max_steps=80)
    obs, _ = env.reset()
    obs_dim = flatten_obs(obs).shape[0]
    act_dim = env.action_space.n if env.action_space is not None else 6

    policy = REINFORCEAgent(obs_dim, act_dim)
    policy.load_state_dict(torch.load(model_path, map_location="cpu"))
    policy.eval()

    total_reward = 0
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0
        while not done:
            obs_tensor = torch.tensor(flatten_obs(obs), dtype=torch.float32)
            action_probs = policy(obs_tensor)
            action = torch.argmax(action_probs).item()
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        total_reward += ep_reward
    env.close()
    return total_reward / n_episodes

# -----------------------------
# Evaluate SB3 models (PPO, A2C, DQN)
# -----------------------------
def evaluate_sb3_model(model_path, algorithm_name, n_episodes=5):
    try:
        if 'dqn' in algorithm_name.lower():
            from stable_baselines3 import DQN
            model = DQN.load(model_path)
        elif 'ppo' in algorithm_name.lower():
            from stable_baselines3 import PPO
            model = PPO.load(model_path)
        elif 'a2c' in algorithm_name.lower():
            from stable_baselines3 import A2C
            model = A2C.load(model_path)
        else:
            return 0

        env = SMEEFEnv(grid_size=6, max_steps=80)
        total_reward = 0

        for _ in range(n_episodes):
            obs, _ = env.reset()
            done = False
            ep_reward = 0
            while not done:
                obs_array = flatten_obs(obs)
                action, _ = model.predict(obs_array, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                ep_reward += reward
                done = terminated or truncated
            total_reward += ep_reward

        env.close()
        return total_reward / n_episodes
    except Exception as e:
        print(f"❌ Error evaluating {algorithm_name}: {e}")
        return 0

# -----------------------------
# Compare all algorithms
# -----------------------------
def compare_all_algorithms():
    print("🧪 COMPARING ALL ALGORITHMS")
    print("="*50)

    algorithms = [
        ("DQN", "models/dqn/dqn_best_model.zip"),
        ("A2C", "models/a2c/a2c_best_model.zip"),
        ("PPO", "models/ppo/ppo_best_model.zip"),
        ("REINFORCE BEST", "models/reinforce/default/policy_best.pth"),
        ("REINFORCE FINAL", "models/reinforce/default/policy_final.pth"),
    ]

    results = {}

    for algo_name, model_path in algorithms:
        if os.path.exists(model_path):
            if "REINFORCE" in algo_name:
                reward = evaluate_reinforce_model(model_path)
            else:
                reward = evaluate_sb3_model(model_path, algo_name)
            results[algo_name] = reward
            print(f"📊 {algo_name:15}: {reward:8.2f}")
        else:
            print(f"⚠️  {algo_name}: Model not found -> {model_path}")
            results[algo_name] = 0

    plot_comparison(results)

# -----------------------------
# Plot results
# -----------------------------
def plot_comparison(results):
    algorithms = list(results.keys())
    rewards = list(results.values())

    plt.figure(figsize=(10, 6))
    bars = plt.bar(algorithms, rewards, alpha=0.8,
                   color=['#ff6b6b', '#4ecdc4', '#45b7d1', '#ffa600', '#bb8fce'])

    plt.ylabel('Mean Reward')
    plt.title('RL Algorithm Performance Comparison')
    plt.grid(True, alpha=0.3)

    for i, v in enumerate(rewards):
        plt.text(i, v + 0.5, f'{v:.1f}', ha='center', fontweight='bold')

    plt.tight_layout()
    os.makedirs("outputs/plots", exist_ok=True)
    plt.savefig('outputs/plots/algorithm_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    compare_all_algorithms()
