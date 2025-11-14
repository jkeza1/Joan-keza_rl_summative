# environment/rendering.py
import pygame
import numpy as np

# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 200, 0)
RED = (255, 0, 0)
BLUE = (0, 100, 255)
YELLOW = (255, 255, 0)

def render_environment(env):
    if env.window is None:
        pygame.init()
        env.window = pygame.display.set_mode((env.window_size, env.window_size))
        pygame.display.set_caption("SMEEF Empowerment Simulation")
    if env.clock is None:
        env.clock = pygame.time.Clock()

    env.window.fill(WHITE)

    grid_size = env.grid_size
    cell_size = env.window_size // grid_size

    # Draw grid
    for x in range(grid_size):
        for y in range(grid_size):
            rect = pygame.Rect(x * cell_size, y * cell_size, cell_size, cell_size)
            pygame.draw.rect(env.window, BLACK, rect, 1)

    # Draw goal (empowerment area)
    goal_rect = pygame.Rect(
        env.goal_pos[0] * cell_size, env.goal_pos[1] * cell_size, cell_size, cell_size
    )
    pygame.draw.rect(env.window, GREEN, goal_rect)

    # Draw agent
    agent_rect = pygame.Rect(
        env.agent_pos[0] * cell_size + 5,
        env.agent_pos[1] * cell_size + 5,
        cell_size - 10,
        cell_size - 10,
    )
    pygame.draw.rect(env.window, BLUE, agent_rect)

    # Display stats
    font = pygame.font.Font(None, 26)
    text = font.render(
        f"Skill: {env.skill_level} | Energy: {env.energy_level}", True, BLACK
    )
    env.window.blit(text, (10, 10))

    pygame.display.flip()
    env.clock.tick(env.metadata["render_fps"])
    frame = pygame.surfarray.array3d(env.window)  # shape: (width, height, 3)
    frame = np.transpose(frame, (1, 0, 2))        # transpose to (height, width, 3)
    return frame