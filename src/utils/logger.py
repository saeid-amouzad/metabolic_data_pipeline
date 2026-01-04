# src/utils/logger.py

"""
Creates a consistent logger with timestamps and log levels.
Ensures logs look the same everywhere
"""

import logging
import sys


def setup_logger(name="metabolic_pipeline", level=logging.INFO):
    """
    Configure structured logging for the application.
    Logs include timestamp, level, module, and message.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )

        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
