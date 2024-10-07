# utils/logging_setup.py

import logging
import os

def setup_logging(log_dir='logs/', log_file='log_file.txt'):
    """
    Sets up logging to file and console.

    Args:
        log_dir (str, optional): Directory to store log files. Defaults to 'logs/'.
        log_file (str, optional): Log file name. Defaults to 'log_file.txt'.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)

    # Clear previous handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
