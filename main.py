# main.py

import logging
from MultiAgentRLSystem import MultiAgentRLSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    """
    Main entry point for initializing and running the multi-agent RL system.
    """
    # Initialize the multi-agent RL system
    rl_system = MultiAgentRLSystem(
        total_timesteps=50000,        # Adjust timesteps as needed
        model_dir='models/',          # Directory to save models
        log_dir='logs/',              # Directory to save logs
        config_path='utils/config.yaml'  # Path to config file
    )

    try:
        # Train the agents
        rl_system.train()

        # Optionally, load pre-trained models
        # rl_system.load_models()

        # Let the trained agents play against each other
        rl_system.play(num_games=10)
    except KeyboardInterrupt:
        logger.info("Interrupted by user. Saving models before exiting.")
        rl_system.save_models()
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        rl_system.save_models()

if __name__ == "__main__":
    main()