# training/reinforce_training.py
import os
import torch
from environment.smeef_env import SMEEFEnv
from agents.reinforce_agent import REINFORCEAgent

# -------------------------------
# Hyperparameters
# -------------------------------
episodes = 500
gamma = 0.99
lr = 1e-3

# -------------------------------
# Environment and agent
# -------------------------------
env = SMEEFEnv()
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = REINFORCEAgent(state_dim, action_dim, lr=lr)

# Create folder for saving models
os.makedirs("models/reinforce", exist_ok=True)

# -------------------------------
# Training loop
# -------------------------------
best_reward = float('-inf')

for episode in range(1, episodes + 1):
    state, _ = env.reset()
    log_probs = []
    rewards = []
    done = False

    while not done:
        action, log_prob = agent.select_action(state)
        next_state, reward, terminated, _, _ = env.step(action)
        log_probs.append(log_prob)
        rewards.append(reward)
        state = next_state
        done = terminated

    total_reward = sum(rewards)

    # Compute returns
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    # Update policy
    loss = 0
    for log_prob, Gt in zip(log_probs, returns):
        loss -= log_prob * Gt
    agent.optimizer.zero_grad()
    loss.backward()
    agent.optimizer.step()

    # Save best model in .zip
    if total_reward > best_reward:
        best_reward = total_reward
        save_path = "models/reinforce/best_model.zip"
        torch.save(agent, save_path)
        print(f"Episode {episode}: New best reward {best_reward:.2f}, model saved to {save_path}")

    if episode % 50 == 0:
        print(f"Episode {episode}/{episodes}, Total Reward: {total_reward:.2f}")

env.close()
print("REINFORCE training completed!")

# -------------------------------
# Save model
# -------------------------------
save_path = "models/reinforce/best_model.zip"
torch.save(agent, save_path)
print(f"Model saved to {save_path}")