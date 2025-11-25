# compare_best_model.py
import os
import sys
import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import FlattenObservation

# Make sure your environment folder is in the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment.smeef_env import SMEEFEnv

# ================================
# ENVIRONMENT CREATION
# ================================
def make_env():
    env = SMEEFEnv()
    env = FlattenObservation(env)
    env = Monitor(env)  # Proper monitoring for evaluation
    return env

# ================================
# LOAD MODELS
# ================================
def load_models():
    models = {}
    
    model_paths = {
        "DQN": "models/dqn/global_best_model.zip",
        "PPO": "models/ppo/ppo_global_best_model.zip",
        "A2C": "models/a2c/global_best_model.zip"
    }
    
    for name, path in model_paths.items():
        if os.path.exists(path):
            if name == "DQN":
                models[name] = DQN.load(path)
            elif name == "PPO":
                models[name] = PPO.load(path)
            elif name == "A2C":
                models[name] = A2C.load(path)
            print(f"✅ Loaded {name}")
        else:
            print(f"⚠️ {name} model not found at {path}")
    
    return models

# ================================
# EVALUATE MODELS
# ================================
def evaluate_model(model, env, n_eval_episodes=5):
    print(f"🔹 Evaluating {model.__class__.__name__} for {n_eval_episodes} episodes...")
    mean_reward, std_reward = evaluate_policy(
        model,
        env,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        return_episode_rewards=False,
        warn=False
    )
    return mean_reward, std_reward

# ================================
# MAIN
# ================================
if __name__ == "__main__":
    env = make_env()
    models = load_models()

    if not models:
        print("❌ No models found.")
        exit(1)

    eval_results = {}
    for name, model in models.items():
        mean, std = evaluate_model(model, env, n_eval_episodes=5)
        eval_results[name] = mean
        print(f"{name}: mean reward = {mean:.2f}, std = {std:.2f}")

    # Determine best model
    best_model = max(eval_results, key=eval_results.get)
    print(f"\n🏆 Best model: {best_model} with mean reward = {eval_results[best_model]:.2f}")

    env.close()
