"""
Setup logger
"""

import logging
import os
import sys

from loguru import logger


def get_rank():
    """Get the current process rank."""
    return int(os.environ.get("LOCAL_RANK", 0))


# Create a custom handler that routes standard logging to loguru
class InterceptHandler(logging.Handler):
    def emit(self, record):
        # Skip logging if not rank 0
        if get_rank() != 0:
            return

        # Skip debug and lower level logs
        if record.levelno < logging.INFO:
            return

        # Print message in bold
        print(f"\033[1m{record.getMessage()}\033[0m")


# Configure loguru with minimal format (just the message)
logger_format = "{message}"
logger.remove()
logger.configure(
    handlers=[{"sink": sys.stderr, "level": "INFO", "format": logger_format}]
)

# Only add handler for rank 0
if get_rank() == 0:
    logger.add(sys.stderr, format=logger_format)

# Intercept everything from the default logging system at INFO level and above
logging.basicConfig(handlers=[InterceptHandler()], level=logging.INFO, force=True)

# Set higher levels for noisy loggers
logging.getLogger("hydra").setLevel(logging.WARNING)
logging.getLogger("filelock").setLevel(logging.WARNING)
logging.getLogger("omegaconf").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)


def log(message: str, color: str = ""):
    """Log message only for rank 0 process."""
    if get_rank() != 0:
        return

    if color:
        print(f"\033[1;{_get_color_code(color)}m{message}\033[0m")  # Bold and colored
    else:
        print(f"\033[1m{message}\033[0m")  # Just bold


def _get_color_code(color: str) -> str:
    """Convert color name to ANSI color code."""
    color_map = {
        "red": "31",
        "green": "32",
        "yellow": "33",
        "blue": "34",
        "magenta": "35",
        "cyan": "36",
        "white": "37",
    }
    return color_map.get(color, "0")
