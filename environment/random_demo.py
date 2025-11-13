# environment/random_demo.py
import gymnasium as gym
import numpy as np
import time
from environment.smeef_env import SMEEFEnv

def random_demo(episodes=3, steps_per_episode=20, delay=0.4):
    env = SMEEFEnv(render_mode="human")
    for episode in range(episodes):
        obs, _ = env.reset()
        total_reward = 0
        for step in range(steps_per_episode):
            action = env.action_space.sample()
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward
            env.render()
            time.sleep(delay)
            if done:
                break
        print(f"Episode {episode + 1}: Total Reward = {total_reward:.2f}")
    env.close()

if __name__ == "__main__":
    random_demo()
