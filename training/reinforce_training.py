import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
import shutil

# Uncomment these lines when running in your repo
# from environment.smeef_env import SMEEFEnv
# from environment.obs_wrappers import NormalizeFlattenObs
# from agents.reinforce_agent import REINFORCEAgent

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

    env = NormalizeFlattenObs(SMEEFEnv(grid_size=6, max_steps=50))
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = REINFORCEAgent(state_dim, action_dim, hidden_size=hidden_size, lr=lr)

    os.makedirs("models/reinforce", exist_ok=True)
    os.makedirs("outputs/plots", exist_ok=True)

    episode_rewards = []
    best_reward = -float('inf')

    for ep in range(total_episodes):
        state = safe_reset(env)
        log_probs = []
        rewards = []
        done = False

        while not done:
            action, log_prob = agent.select_action(state)
            next_state, reward, done, _ = safe_step(env, action)
            log_probs.append(log_prob)
            rewards.append(float(reward))
            state = next_state

        returns = calculate_returns(rewards, gamma)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        policy_loss = [-log * G for log, G in zip(log_probs, returns)]
        loss = torch.stack(policy_loss).sum()

        agent.optimizer.zero_grad()
        loss.backward()
        agent.optimizer.step()

        total_reward = sum(rewards)
        episode_rewards.append(total_reward)

        # Save per-run best model
        if total_reward > best_reward:
            best_reward = total_reward
            best_model_path = f"models/reinforce/best_model_{name}.pt"
            torch.save({
                'policy_state_dict': agent.policy.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'episode': ep + 1,
                'reward': total_reward
            }, best_model_path)
            print(f"  🔥 New best model for {name} at episode {ep+1}: reward={total_reward:.2f}")

        if (ep + 1) % 100 == 0:
            avg100 = np.mean(episode_rewards[-100:])
            print(f"Episode {ep+1}/{total_episodes} | Avg100 Reward: {avg100:.2f}")

    # Save final per-run model
    final_model_path = f"models/reinforce/final_model_{name}.pt"
    torch.save({
        'policy_state_dict': agent.policy.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict()
    }, final_model_path)
    print(f"Saved final model for {name} to {final_model_path}")

    # Plot rewards
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards, label="Episode Reward")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title(f"REINFORCE Training Rewards: {name}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"outputs/plots/rewards_{name}.png", dpi=200)
    plt.show()

    env.close()

    return {
        'name': name,
        'best_reward': best_reward,
        'final_avg100': float(np.mean(episode_rewards[-100:])),
        'best_model_path': best_model_path,
        'final_model_path': final_model_path
    }

def create_hyperparam_grid() -> List[Dict]:
    """Return at least 10 hyperparameter combinations."""
    grid = []
    names = [f"run_{i+1}" for i in range(12)]
    lrs = [5e-4, 1e-3, 2e-3]
    gammas = [0.9, 0.95, 0.99]
    hidden_sizes = [64, 128]

    i = 0
    for lr in lrs:
        for gamma in gammas:
            for h in hidden_sizes:
                if i >= len(names):
                    return grid
                grid.append({
                    'name': names[i],
                    'learning_rate': lr,
                    'gamma': gamma,
                    'hidden_size': h,
                    'total_episodes': 1000,
                    'group': f'LR{lr}_G{gamma}_H{h}'
                })
                i += 1
    return grid

def train_reinforce_all():
    grid = create_hyperparam_grid()
    all_results = []
    global_best = -float('inf')
    global_best_path = "models/reinforce/global_best_model.pt"

    for i, cfg in enumerate(grid):
        print(f"\n🏃 REINFORCE Run {i+1}/{len(grid)} - {cfg['group']}")
        res = run_one(cfg)
        all_results.append(res)

        # Save global best model
        if res['best_reward'] > global_best:
            global_best = res['best_reward']
            shutil.copy(res['best_model_path'], global_best_path)
            print(f"💫 NEW GLOBAL BEST MODEL saved to {global_best_path} | Reward: {global_best:.2f}")

    return all_results

if __name__ == "__main__":
    results = train_reinforce_all()
    print("\n🎉 All runs complete!")
