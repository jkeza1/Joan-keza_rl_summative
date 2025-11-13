# agents/ppo_agent.py
from stable_baselines3 import PPO

def create_ppo_agent(env, **kwargs):
    """
    Create a PPO agent for the given environment.

    Parameters:
        env (gym.Env): Gymnasium environment
        kwargs: Additional hyperparameters like learning_rate, n_steps, gamma, etc.

    Returns:
        model: Stable Baselines3 PPO model
    """
    default_params = dict(
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.0,
        verbose=1,
    )
    default_params.update(kwargs)

    model = PPO("MlpPolicy", env, **default_params)
    return model
