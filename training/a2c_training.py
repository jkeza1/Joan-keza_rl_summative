# training/a2c_training.py
import os
from environment.smeef_env import SMEEFEnv
from agents.a2c_agent import create_a2c_agent

# Create environment
env = SMEEFEnv(render_mode=None)

# Create A2C agent
model = create_a2c_agent(env, learning_rate=7e-4, n_steps=5)

# Train the agent
total_timesteps = 50000
model.learn(total_timesteps=total_timesteps)

# Save model
os.makedirs("models/a2c", exist_ok=True)
model.save("models/a2c/best_model")

print("A2C training completed and model saved!")
env.close()
