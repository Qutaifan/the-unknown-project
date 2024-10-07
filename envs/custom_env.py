# envs/custom_env.py

import numpy as np
import gym
from gym import spaces
import logging
from pettingzoo.utils.env import AECEnv
from pettingzoo.utils import agent_selector
from utils.config import load_config

logger = logging.getLogger(__name__)

class CustomMultiAgentEnv(AECEnv):
    """
    Custom Multi-Agent Environment where multiple agents interact with each other and the environment.
    """

    metadata = {'render.modes': ['human']}

    def __init__(self, num_agents=2, config_path='utils/config.yaml'):
        """
        Initializes the CustomMultiAgentEnv.

        Args:
            num_agents (int, optional): Number of agents in the environment. Defaults to 2.
            config_path (str, optional): Path to the configuration file. Defaults to 'utils/config.yaml'.
        """
        super(CustomMultiAgentEnv, self).__init__()
        self.num_agents = num_agents

        # Load configuration
        self.config = load_config(config_path)

        # Define agents
        self.agents = [f"agent_{i}" for i in range(self.num_agents)]
        self.possible_agents = self.agents[:]
        self.agent_selector = agent_selector(self.possible_agents)
        self.agent_selection = self.agent_selector.next()

        # Define action and observation space for each agent
        # Assuming continuous action space for each agent
        self.action_space = {
            agent: spaces.Box(
                low=self.config['action_space']['low'],
                high=self.config['action_space']['high'],
                shape=(self.config['action_space']['shape'],),
                dtype=np.float32
            ) for agent in self.agents
        }

        # Assuming each agent has a continuous observation space
        self.observation_space = {
            agent: spaces.Box(
                low=self.config['observation_space']['low'],
                high=self.config['observation_space']['high'],
                shape=(self.config['observation_space']['shape'],),
                dtype=np.float32
            ) for agent in self.agents
        }

        # Initialize the state of the environment
        self.state = np.zeros((self.num_agents, self.observation_space[self.agents[0]].shape[0]), dtype=np.float32)

        # Initialize rewards, dones, infos
        self.rewards = {agent: 0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        # Debugging: Print the structure of the observation space
        logger.debug(f"Observation space structure: {self.env.observation_space}")

        logger.info("Custom Multi-Agent Environment initialized successfully.")

    def reset(self):
        """
        Resets the environment to an initial state.

        Returns:
            None
        """
        try:
            self.state = np.random.randn(self.num_agents, self.observation_space[self.agents[0]].shape[0]).astype(np.float32)
            self.rewards = {agent: 0 for agent in self.agents}
            self.dones = {agent: False for agent in self.agents}
            self.infos = {agent: {} for agent in self.agents}
            self.agent_selector = agent_selector(self.possible_agents)
            self.agent_selection = self.agent_selector.next()
            self._clear_rewards()
            self._clear_dones()
            self._clear_infos()
            logger.info("Environment reset successfully.")
        except Exception as e:
            logger.error(f"Error during environment reset: {e}")
            raise e

    def step(self, action):
        """
        Executes a step in the environment.

        Args:
            action (np.ndarray): Action taken by the current agent.

        Returns:
            None
        """
        try:
            agent = self.agent_selection
            agent_idx = self.agents.index(agent)

            # Apply action to the environment state
            noise = np.random.randn(*action.shape).astype(np.float32) * self.config['noise_multiplier']
            self.state[agent_idx] += noise + action

            # Compute rewards: negative Euclidean distance from origin
            self.rewards[agent] = -np.linalg.norm(self.state[agent_idx])

            # Determine termination: if any agent's state norm exceeds the threshold
            self.dones[agent] = bool(np.linalg.norm(self.state[agent_idx]) > self.config['termination_threshold'])

            # Inform if all agents are done
            all_done = all(self.dones.values())
            if all_done:
                self.dones = {agent: True for agent in self.agents}

            # Advance to the next agent
            self.agent_selection = self.agent_selector.next()

            logger.debug(f"Agent {agent} took action {action}. Reward: {self.rewards[agent]}, Done: {self.dones[agent]}")

        except Exception as e:
            logger.error(f"Error during environment step: {e}")
            raise e

    def render(self, mode='human'):
        """
        Renders the environment.

        Args:
            mode (str): The mode to render with.
        """
        try:
            if mode == 'human':
                print(f"Current state:\n{self.state}")
        except Exception as e:
            logger.error(f"Error during render: {e}")
            raise e

    def close(self):
        """
        Closes the environment.
        """
        try:
            logger.info("Environment closed.")
        except Exception as e:
            logger.error(f"Error during environment close: {e}")
            raise e