# training/ppo_training.py
import os
from environment.smeef_env import SMEEFEnv
from agents.ppo_agent import create_ppo_agent

# Create environment
env = SMEEFEnv(render_mode=None)

# Create PPO agent
model = create_ppo_agent(env, learning_rate=3e-4, n_steps=2048, batch_size=64)

# Train the agent
total_timesteps = 50000
model.learn(total_timesteps=total_timesteps)

# Save model
os.makedirs("models/ppo", exist_ok=True)
model.save("models/ppo/best_model")

print("PPO training completed and model saved!")
env.close()
