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

        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame, depth = sys._getframe(6), 6
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


# Configure loguru
logger_format = "<level>{message}</level>"
logger.remove()

# Only add handler for rank 0
if get_rank() == 0:
    logger.add(sys.stderr, format=logger_format)

# Intercept everything from the default logging system
logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

# Optional: Set specific levels for some loggers
logging.getLogger("hydra").setLevel(logging.INFO)
logging.getLogger("filelock").setLevel(logging.INFO)
logging.getLogger("omegaconf").setLevel(logging.INFO)


def log(message: str, color: str = ""):
    """Log message only for rank 0 process."""
    if get_rank() != 0:
        return

    if color:
        logger.opt(colors=True).info(f"<{color}>{message}</{color}>")
    else:
        logger.info(message)
