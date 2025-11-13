# agents/dqn_agent.py
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env

def create_dqn_agent(env, **kwargs):
    """
    Create a DQN agent for the given environment.

    Parameters:
        env (gym.Env): Gymnasium environment
        kwargs: Additional hyperparameters like learning_rate, buffer_size, etc.

    Returns:
        model: Stable Baselines3 DQN model
    """
    default_params = dict(
        learning_rate=1e-3,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=32,
        gamma=0.99,
        target_update_interval=500,
        verbose=1,
    )
    default_params.update(kwargs)

    model = DQN("MlpPolicy", env, **default_params)
    return model
