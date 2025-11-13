# training/dqn_training.py
import os
from stable_baselines3 import DQN
from environment.smeef_env import SMEEFEnv
from agents.dqn_agent import create_dqn_agent

# Create environment
env = SMEEFEnv(render_mode=None)

# Create DQN agent
model = create_dqn_agent(env, learning_rate=1e-3, buffer_size=50000, batch_size=32)

# Train the agent
total_timesteps = 50000
model.learn(total_timesteps=total_timesteps)

# Create models directory if not exists
os.makedirs("models/dqn", exist_ok=True)

# Save the trained model
model.save("models/dqn/best_model")

print("DQN training completed and model saved!")
env.close()
