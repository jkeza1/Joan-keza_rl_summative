# agents/a2c_agent.py
from stable_baselines3 import A2C

def create_a2c_agent(env, **kwargs):
    """
    Create an Advantage Actor-Critic (A2C) agent.

    Parameters:
        env (gym.Env): Gymnasium environment
        kwargs: Hyperparameters like learning_rate, gamma, n_steps, etc.

    Returns:
        model: Stable Baselines3 A2C model
    """
    default_params = dict(
        learning_rate=7e-4,
        n_steps=5,
        gamma=0.99,
        gae_lambda=1.0,
        ent_coef=0.01,
        verbose=1,
    )
    default_params.update(kwargs)

    model = A2C("MlpPolicy", env, **default_params)
    return model
