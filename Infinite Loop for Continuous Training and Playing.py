# infinite_loop_for_continuous_training_and_playing.py

import logging
import time
from utils.config import load_config
from utils.logging_setup import setup_logging
from MultiAgentRLSystem import MultiAgentRLSystem

# Configure logging
setup_logging()
logger = logging.getLogger(__name__)

def main():
    """
    Main function to continuously train and play with the multi-agent RL system.
    """
    # Load configuration
    config = load_config('utils/config.yaml')

    # Initialize the multi-agent system with configurations
    multi_agent_rl = MultiAgentRLSystem(
        total_timesteps=config['training']['total_timesteps'],
        model_dir=config['training']['model_dir'],
        log_dir=config['training']['log_dir']
    )

    try:
        while True:
            logger.info("Starting a new training session.")
            multi_agent_rl.train()

            # Optionally, load pre-trained models
            # multi_agent_rl.load_models()

            logger.info("Starting a play session.")
            multi_agent_rl.play(num_games=config['play']['num_games'])

            # Sleep for a short duration to prevent resource exhaustion
            time.sleep(config['training']['sleep_duration'])

    except KeyboardInterrupt:
        logger.info("Training interrupted by user. Saving models and exiting...")
        multi_agent_rl.save_models()
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        multi_agent_rl.save_models()

if __name__ == "__main__":
    main()