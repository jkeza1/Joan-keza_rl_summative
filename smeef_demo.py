import os
import yaml
import torch
import cv2
import numpy as np
from environment.smeef_env import SMEEFEnv
from agents.reinforce_agent import REINFORCEAgent
from stable_baselines3 import DQN, PPO, A2C

# -------------------------------
# Load environment configuration
# -------------------------------
with open("config/env_config.yaml", "r") as f:
    env_config = yaml.safe_load(f)

# Initialize environment
render_mode = "human" 
env = SMEEFEnv(config=env_config, render_mode=render_mode)  

# -------------------------------
# Select algorithm to load
# -------------------------------
algorithm = "dqn"  # Options: 'dqn', 'ppo', 'a2c', 'reinforce'

# -------------------------------
# Load trained model
# -------------------------------
if algorithm in ["dqn", "ppo", "a2c"]:
    model_path = f"models/{algorithm}/best_model.zip"  # SB3 expects .zip
elif algorithm == "reinforce":
    model_path = f"models/{algorithm}/best_model.pt"  # PyTorch model
else:
    raise ValueError("Invalid algorithm selected!")

# Load agent
if algorithm == "dqn":
    agent = DQN.load(model_path)
elif algorithm == "ppo":
    agent = PPO.load(model_path, env=env)
elif algorithm == "a2c":
    agent = A2C.load(model_path, env=env)
elif algorithm == "reinforce":
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = REINFORCEAgent(state_dim, action_dim)
    agent.load_state_dict(torch.load(model_path))
    agent.eval()

# -------------------------------
# Prepare video writer
# -------------------------------
os.makedirs("outputs/videos", exist_ok=True)
frame = env.render()
frame_width, frame_height = frame.shape[1], frame.shape[0]

video_path = f"outputs/videos/{algorithm}_demo.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(video_path, fourcc, 10, (frame_width, frame_height))

# -------------------------------
# Run simulation and record video
# -------------------------------
num_episodes = 1

for ep in range(num_episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        frame = env.render()  # get RGB frame
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

        # Select action
        if algorithm in ["dqn", "ppo", "a2c"]:
            action, _ = agent.predict(state, deterministic=True)
        elif algorithm == "reinforce":
            action, _ = agent.select_action(state)

        # Step environment
        next_state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        state = next_state
        done = terminated or truncated

    print(f"Episode {ep+1} Total Reward: {total_reward}")

# Release video writer and close environment
out.release()
env.close()
print(f"Demo video saved at {video_path}")
