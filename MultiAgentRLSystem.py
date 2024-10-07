# MultiAgentRLSystem.py

import numpy as np
import matplotlib.pyplot as plt
from pettingzoo.classic import connect_four_v3
from supersuit import black_death_v3
import torch as th
import os
import logging
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
from agents.role_based_agent import RoleBasedAgent
from agents.other_agent import GNNPolicy
from collections import deque
import random
from utils.config import load_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MultiAgentRLSystem:
    """
    Core system managing multi-agent interactions, training loops, and system dynamics.
    """

    def __init__(self, total_timesteps=500000, model_dir='models/', log_dir='logs/', config_path='utils/config.yaml'):
        """
        Initializes the MultiAgentRLSystem.

        Args:
            total_timesteps (int, optional): Total number of training episodes. Defaults to 500000.
            model_dir (str, optional): Directory to save models. Defaults to 'models/'.
            log_dir (str, optional): Directory to save logs. Defaults to 'logs/'.
            config_path (str, optional): Path to the configuration file. Defaults to 'utils/config.yaml'.
        """
        self.config = load_config(config_path)
        self.total_timesteps = total_timesteps
        self.model_dir = model_dir
        self.log_dir = log_dir
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        # Initialize the Connect Four environment
        try:
            env = connect_four_v3.env()
            env.metadata['is_parallelizable'] = True  # Indicate that the environment can be parallelized
            env = black_death_v3(env)  # Handles agent removal upon termination
            self.env = env
            logger.info("Environment initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing environment: {e}")
            raise e

        # Initialize TensorBoard writer
        try:
            self.writer = SummaryWriter(log_dir=self.log_dir)
            logger.info("TensorBoard writer initialized.")
        except Exception as e:
            logger.error(f"Error initializing TensorBoard writer: {e}")
            raise e

        # Environment parameters
        self.num_agents = len(self.env.possible_agents)
        # Assuming discrete action space from Connect Four
        self.obs_dim = self.env.observation_space().spaces['observation'].shape[0]
        
        self.act_dim = self.env.action_space.n  # Number of discrete actions

        # Initialize agents with RoleBasedAgent and GNNPolicy
        try:
            self.agent_policies = {
                agent: GNNPolicy(
                    obs_dim=self.obs_dim,
                    act_dim=self.act_dim,
                    num_agents=self.num_agents,
                    hidden_dim=self.config['agent']['hidden_dim']
                )
                for agent in self.env.possible_agents
            }
            logger.info("Agents initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing agents: {e}")
            raise e

        # Initialize optimizers
        try:
            self.optimizers = {
                agent: th.optim.Adam(self.agent_policies[agent].parameters(), lr=self.config['agent']['learning_rate'])
                for agent in self.env.possible_agents
            }
            logger.info("Optimizers initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing optimizers: {e}")
            raise e

        # Replay buffer
        self.memory = deque(maxlen=self.config['training']['replay_buffer_size'])
        self.batch_size = self.config['training']['batch_size']

        # Metrics
        self.win_counts = {agent: 0 for agent in self.agent_policies.keys()}
        self.loss_logs = {agent: [] for agent in self.agent_policies.keys()}
        self.scoreboard = {agent: {"wins": 0, "losses": 0, "draws": 0} for agent in self.agent_policies.keys()}

    def train(self):
        """
        Executes the training loop for the specified number of episodes.
        """
        logger.info("Starting training...")

        try:
            for episode in range(1, self.total_timesteps + 1):
                obs_n = self.env.reset()
                done_n = {agent: False for agent in self.env.possible_agents}
                episode_reward = {agent: 0 for agent in self.env.possible_agents}
                steps = 0

                while not all(done_n.values()):
                    actions = {}
                    roles = {}
                    for agent in self.env.agent_iter():
                        observation, reward, done, info = self.env.last()
                        if done:
                            action = None
                            actions[agent] = action
                            self.env.step(action)
                            continue

                        # Select action based on policy
                        action = self.agent_policies[agent](observation['observation']).argmax().item()
                        actions[agent] = action
                        episode_reward[agent] += reward

                        # Step the environment
                        self.env.step(action)

                    steps += 1

                    # Placeholder: Store transitions if using experience replay
                    # Implement experience replay storage based on your RL algorithm
                    # Example:
                    # for agent in self.env.possible_agents:
                    #     self.memory.append((obs_n[agent], actions[agent], episode_reward[agent], next_obs_n[agent], done_n[agent], roles.get(agent)))

                    # Optionally, update agents with sampled experiences
                    if len(self.memory) > self.batch_size:
                        samples = random.sample(self.memory, self.batch_size)
                        self._update_agents(samples, episode)

                # Update scoreboard based on episode outcomes
                self._update_scoreboard(episode_reward)

                # Logging
                if episode % self.config['training']['log_interval'] == 0:
                    self._log_metrics(episode)

                # Periodically save models
                if episode % self.config['training']['save_interval'] == 0:
                    self.save_models()

        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise e
        finally:
            logger.info("Training loop has ended.")
            self.writer.close()

    def _update_agents(self, samples, episode):
        """
        Updates the agents based on sampled experiences.

        Args:
            samples (list): A list of sampled experiences.
            episode (int): Current episode number.
        """
        try:
            for agent in self.agent_policies.keys():
                # Prepare batch data
                obs_batch = th.tensor([sample[0][agent] for sample in samples], dtype=th.float32).to(self.agent_policies[agent].device)
                actions_batch = th.tensor([sample[1][agent] for sample in samples], dtype=th.long).to(self.agent_policies[agent].device)
                rewards_batch = th.tensor([sample[2][agent] for sample in samples], dtype=th.float32).to(self.agent_policies[agent].device)
                next_obs_batch = th.tensor([sample[3][agent] for sample in samples], dtype=th.float32).to(self.agent_policies[agent].device)
                done_batch = th.tensor([sample[4][agent] for sample in samples], dtype=th.float32).to(self.agent_policies[agent].device)

                # Forward pass through current policy's critic
                q_values = self.agent_policies[agent](next_obs_batch).squeeze()

                # Compute target Q-values
                with th.no_grad():
                    target_q_values = rewards_batch + (self.config['training']['gamma'] * q_values) * (1 - done_batch)

                # Compute loss
                loss = F.mse_loss(self.agent_policies[agent](obs_batch).squeeze(), target_q_values)

                # Optimize the critic
                self.optimizers[agent].zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.agent_policies[agent].parameters(), max_norm=1.0)
                self.optimizers[agent].step()

                # Log loss
                self.loss_logs[agent].append(loss.item())

        except Exception as e:
            logger.error(f"Error during agent update: {e}")
            raise e

    def _update_scoreboard(self, episode_reward):
        """
        Updates the scoreboard based on episode rewards.

        Args:
            episode_reward (dict): Rewards obtained by each agent in the episode.
        """
        try:
            for agent, reward in episode_reward.items():
                if reward > 0:
                    self.scoreboard[agent]['wins'] += 1
                elif reward < 0:
                    self.scoreboard[agent]['losses'] += 1
                else:
                    self.scoreboard[agent]['draws'] += 1
        except Exception as e:
            logger.error(f"Error updating scoreboard: {e}")
            raise e

    def _log_metrics(self, episode):
        """
        Logs the metrics to TensorBoard and console.

        Args:
            episode (int): Current episode number.
        """
        try:
            # Log average loss
            for agent, losses in self.loss_logs.items():
                if len(losses) >= 100:
                    avg_loss = np.mean(losses[-100:])
                    self.writer.add_scalar(f'Loss/{agent}', avg_loss, episode)
                    logger.info(f"Agent: {agent}, Avg Loss: {avg_loss}")

            # Log the scoreboard
            for agent, scores in self.scoreboard.items():
                logger.info(f"Agent: {agent}, Wins: {scores['wins']}, Losses: {scores['losses']}, Draws: {scores['draws']}")
        except Exception as e:
            logger.error(f"Error during metric logging: {e}")
            raise e

    def play(self, num_games=10):
        """
        Allows trained agents to play against each other.

        Args:
            num_games (int, optional): Number of games to play. Defaults to 10.
        """
        try:
            logger.info("Agents are playing against each other...")

            for game in range(1, num_games + 1):
                obs_n = self.env.reset()
                done_n = {agent: False for agent in self.env.possible_agents}
                episode_reward = {agent: 0 for agent in self.env.possible_agents}

                while not all(done_n.values()):
                    for agent in self.env.agent_iter():
                        observation, reward, done, info = self.env.last()
                        if done:
                            action = None
                            self.env.step(action)
                            continue

                        # Select action based on policy
                        action = self.agent_policies[agent](observation['observation']).argmax().item()
                        self.env.step(action)
                        episode_reward[agent] += reward

                # Determine the winner
                result = self.env.get_outcome()
                winner = result.get('winner', None)
                if winner is not None:
                    self.win_counts[winner] += 1
                    logger.info(f"Game {game}: Winner - {winner}")
                else:
                    logger.info(f"Game {game}: Draw")

            logger.info("Play session completed.")
            logger.info(f"Win counts: {self.win_counts}")
            self._plot_win_counts()

        except Exception as e:
            logger.error(f"Error during play session: {e}")
            raise e

    def _plot_win_counts(self):
        """
        Plots the number of wins per agent.
        """
        try:
            agents = list(self.win_counts.keys())
            wins = list(self.win_counts.values())

            plt.figure(figsize=(10, 6))
            plt.bar(agents, wins, color=['blue', 'green', 'red', 'orange', 'purple'])
            plt.xlabel('Agents')
            plt.ylabel('Number of Wins')
            plt.title('Win Counts after Games')
            plt.show()
        except Exception as e:
            logger.error(f"Error during plotting win counts: {e}")
            raise e

    def save_models(self):
        """
        Saves the model weights for all agents.
        """
        try:
            for agent, policy in self.agent_policies.items():
                path = os.path.join(self.model_dir, f"{agent}_gnn_policy.pth")
                th.save(policy.state_dict(), path)
                logger.info(f"Saved model for {agent} at {path}")
        except Exception as e:
            logger.error(f"Error saving models: {e}")
            raise e

    def load_models(self):
        """
        Loads the model weights for all agents.
        """
        try:
            for agent, policy in self.agent_policies.items():
                path = os.path.join(self.model_dir, f"{agent}_gnn_policy.pth")
                if os.path.exists(path):
                    policy.load_state_dict(th.load(path))
                    policy.eval()
                    logger.info(f"Loaded model for {agent} from {path}")
                else:
                    logger.warning(f"Model file for {agent} not found at {path}")
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise e

    def store_transition(self, obs, action, reward, next_obs, done, role=None):
        transition = (obs, action, reward, next_obs, done, role)
        self.memory.append(transition)

    def sample_experiences(self, batch_size):
        return random.sample(self.memory, batch_size)