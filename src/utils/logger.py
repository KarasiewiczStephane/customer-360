"""Structured logging utilities for the Customer 360 platform.

Provides a consistent logging interface across all modules with
configurable format and level.
"""

import logging
import sys


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create or retrieve a named logger with standard formatting.

    Args:
        name: Logger name, typically ``__name__`` of the calling module.
        level: Logging level (default ``INFO``).

    Returns:
        Configured :class:`logging.Logger` instance.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)

    return logger
