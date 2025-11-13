# agents/reinforce_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class REINFORCEAgent(nn.Module):
    """
    Vanilla Policy Gradient (REINFORCE) Agent.
    """

    def __init__(self, state_dim, action_dim, hidden_size=128, lr=1e-3):
        super(REINFORCEAgent, self).__init__()
        self.policy = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
            nn.Softmax(dim=-1)
        )
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

    def forward(self, state):
        return self.policy(state)

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        probs = self.forward(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)
