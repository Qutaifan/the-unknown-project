# agents/continuous_multiagent_rl.py

import numpy as np
import torch as th
from collections import defaultdict
import logging
import yaml
from utils.config import load_config

logger = logging.getLogger(__name__)

class ContinuousMultiAgentRLSystem:
    """
    Manages continuous training and interaction loops for multiple agents in the environment.
    """

    def __init__(self, env, agents, total_iterations=1000000, log_interval=100):
        """
        Initializes the ContinuousMultiAgentRLSystem.

        Args:
            env (gym.Env): The environment in which agents interact.
            agents (list): A list of agent instances participating in the environment.
            total_iterations (int, optional): Total number of training iterations. Defaults to 1000000.
            log_interval (int, optional): Interval at which to log metrics. Defaults to 100.
        """
        self.env = env
        self.agents = agents
        self.total_iterations = total_iterations
        self.log_interval = log_interval
        self.iteration = 0
        self.metrics = defaultdict(list)

    def run(self):
        """
        Executes the continuous training loop.
        """
        logger.info("Starting continuous multi-agent training...")
        try:
            for self.iteration in range(1, self.total_iterations + 1):
                state = self.env.reset()
                done = False
                episode_reward = {agent: 0 for agent in self.agents}

                while not done:
                    try:
                        actions = [agent.select_action(state[agent]) for agent in self.agents]
                        next_state, rewards, done, _ = self.env.step(actions)

                        for agent, action, reward in zip(self.agents, actions, rewards):
                            agent.store_transition(state[agent], action, reward, next_state[agent], done)
                            episode_reward[agent] += reward

                        state = next_state

                        # Update agents after each step
                        for agent in self.agents:
                            agent.update()

                    except Exception as step_error:
                        logger.error(f"Error during step execution: {step_error}")
                        done = True  # Terminate the episode on error

                # Logging
                if self.iteration % self.log_interval == 0:
                    self.log_scoreboard()

        except KeyboardInterrupt:
            logger.info("Training interrupted by user.")
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
        finally:
            logger.info("Training loop has ended.")

    def log_scoreboard(self):
        """
        Logs the average rewards of agents at specified intervals.
        """
        avg_rewards = {agent: agent.get_average_reward() for agent in self.agents}
        logger.info(f"Iteration {self.iteration}: Average Rewards: {avg_rewards}")
        for agent, reward in avg_rewards.items():
            self.metrics[agent].append(reward)

    def get_metrics(self):
        """
        Retrieves the logged metrics.

        Returns:
            dict: A dictionary containing average rewards per agent over iterations.
        """
        return self.metrics