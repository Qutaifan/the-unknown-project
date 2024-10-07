# tests/test_role_based_agent.py

import unittest
from unittest.mock import MagicMock
import sys
import os
import numpy as np

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.role_based_agent import RoleBasedAgent
from MultiAgentRLSystem import MultiAgentRLSystem

class TestRoleBasedAgent(unittest.TestCase):
    def setUp(self):
        # Initialize RoleBasedAgent
        self.agent = RoleBasedAgent(
            role='leader',
            obs_dim=4,
            act_dim=7,
            hidden_dim=64,
            lr=1e-3,
            device='cpu'
        )

    def test_select_action(self):
        observation = np.random.randn(4)
        action = self.agent.select_action(observation)
        self.assertIsInstance(action, int)
        self.assertTrue(0 <= action < 7)

    def test_update(self):
        experience = (np.random.randn(4), 3, 1.0, np.random.randn(4), False)
        critic_loss, actor_loss = self.agent.update(experience)
        self.assertIsInstance(critic_loss, float)
        self.assertIsInstance(actor_loss, float)

    def test_save_and_load(self):
        filename = 'test_agent.pth'
        self.agent.save(filename)
        self.agent.load(filename)
        # If no exception is raised, assume success
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
