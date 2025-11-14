# environment/smeef_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

class SMEEFEnv(gym.Env):
    """
    Custom Gymnasium environment simulating empowerment for single mothers.
    The agent (a single mother) moves around a grid environment to collect skills,
    gain resources, and avoid burnout while striving to reach economic empowerment.
    """

    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(self, render_mode=None, grid_size=5, config=None):
       super(SMEEFEnv, self).__init__()

    # Override grid_size from config if provided
       if config is not None:
           grid_size = config.get("grid_size", grid_size)

       self.grid_size = grid_size
       self.window_size = 500
       self.render_mode = render_mode

    # Define action space and observation space
       self.action_space = spaces.Discrete(6)
       low = np.array([0, 0, 0, 0], dtype=np.float32)
       high = np.array([self.grid_size - 1, self.grid_size - 1, 10, 10], dtype=np.float32)
       self.observation_space = spaces.Box(low, high, dtype=np.float32)

    # Initialize state variables
       self.agent_pos = np.array([0, 0])
       self.skill_level = 0
       self.energy_level = 10
       self.goal_pos = np.array([self.grid_size - 1, self.grid_size - 1])

    # Rendering variables
       self.window = None
       self.clock = None


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = np.array([0, 0])
        self.skill_level = 0
        self.energy_level = 10

        observation = np.array([
            *self.agent_pos, self.skill_level, self.energy_level
        ], dtype=np.float32)
        info = {}
        return observation, info

    def step(self, action):
        reward = -0.1  # small penalty to encourage efficiency
        terminated = False

        # Movement Actions
        if action == 0 and self.agent_pos[1] > 0:  # UP
            self.agent_pos[1] -= 1
        elif action == 1 and self.agent_pos[1] < self.grid_size - 1:  # DOWN
            self.agent_pos[1] += 1
        elif action == 2 and self.agent_pos[0] > 0:  # LEFT
            self.agent_pos[0] -= 1
        elif action == 3 and self.agent_pos[0] < self.grid_size - 1:  # RIGHT
            self.agent_pos[0] += 1
        elif action == 4:  # LEARN
            self.skill_level += 1
            reward += 2
        elif action == 5:  # REST
            self.energy_level = min(self.energy_level + 2, 10)
            reward += 1

        # Decrease energy with every move
        self.energy_level -= 1

        # Terminal condition: out of energy or reached goal
        if np.array_equal(self.agent_pos, self.goal_pos) and self.skill_level >= 5:
            reward += 10
            terminated = True
        elif self.energy_level <= 0:
            reward -= 5
            terminated = True

        observation = np.array([
            *self.agent_pos, self.skill_level, self.energy_level
        ], dtype=np.float32)

        info = {}
        return observation, reward, terminated, False, info

    def render(self):
        from environment.rendering import render_environment
        return render_environment(self)

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
