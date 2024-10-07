# agents/role_based_agent.py

import torch.nn as nn
import torch
import numpy as np

class RoleBasedAgent(nn.Module):
    def __init__(self, role, obs_dim, act_dim, hidden_dim, lr, device):
        super(RoleBasedAgent, self).__init__()
        self.role = role
        self.device = device
        self.to(self.device)
        
        # Define a simple neural network
        self.model = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
    def select_action(self, observation):
        """
        Selects an action based on the current observation.
        """
        self.eval()
        with torch.no_grad():
            observation = torch.tensor(observation, dtype=torch.float32).to(self.device)
            logits = self.model(observation)
            action = torch.argmax(logits).item()
        return action

    def load(self, path):
        """
        Loads the model state from a file.
        """
        self.load_state_dict(torch.load(path, map_location=self.device))
        self.eval()

    def eval(self):
        """
        Sets the model to evaluation mode.
        """
        self.model.eval()
        self.critic.eval()