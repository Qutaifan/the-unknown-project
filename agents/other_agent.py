# agents/other_agent.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from utils.config import load_config

logger = logging.getLogger(__name__)

class GNNPolicy(nn.Module):
    """
    Standard feed-forward neural network policy for multi-agent systems.
    """

    def __init__(self, obs_dim, act_dim, num_agents, hidden_dim=128):
        """
        Initializes the GNNPolicy.

        Args:
            obs_dim (int): Dimension of the observation space.
            act_dim (int): Dimension of the action space.
            num_agents (int): Number of agents in the environment.
            hidden_dim (int, optional): Number of neurons in hidden layers. Defaults to 128.
        """
        super(GNNPolicy, self).__init__()
        self.num_agents = num_agents

        # Hidden layers
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # Output layer for actions
        self.action_layer = nn.Linear(hidden_dim, act_dim)

    def forward(self, x):
        """
        Forward pass for feed-forward policy.

        Args:
            x (torch.Tensor): Input features (observations) of shape (num_agents, obs_dim).

        Returns:
            torch.Tensor: Action outputs of shape (num_agents, act_dim).
        """
        try:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            actions = self.action_layer(x)
            logger.debug(f"Feed-forward Policy forward pass successful. Actions: {actions}")
            return actions
        except Exception as e:
            logger.error(f"Error during feed-forward policy forward pass: {e}")
            raise e