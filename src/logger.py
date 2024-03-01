"""
src/logger.py

Module for setting up logging configuration in the project.
ref: https://docs.python.org/3/library/logging.html
"""
import logging
import os
from datetime import datetime

# logfile format: mm_dd_YYYY_HH_MM_SS.log
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_path = os.path.join(os.getcwd(), "logs", LOG_FILE)

os.makedirs(logs_path, exist_ok=True)

# Full logfile path
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# Basic logging configurations
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format='[ %(asctime)s ] %(lineno)d - %(name)s %(levelname)s - %(message)s',
    level=logging.INFO
)

