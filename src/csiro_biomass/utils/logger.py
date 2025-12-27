"""Logging management."""

import logging
import os

import wandb


class WandBHandler(logging.Handler):
    """Custom logging handler to send logs to wandb."""

    def __init__(self, level=logging.DEBUG):
        super().__init__(level)
        self.log_buffer = []

    def emit(self, record: logging.LogRecord):
        """Emit a log record to wandb."""
        try:
            log_message = self.format(record)
            # Log to wandb as a message
            wandb.log({"logs": log_message}, commit=False)
        except Exception:
            self.handleError(record)


def setup_logger(name: str = "kaggle", level: int = logging.INFO) -> logging.Logger:
    """
    Sets up a logger with the specified name and logging level.

    Args:
        name (str): The name of the logger.
        level (int): The logging level (default is logging.INFO).

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    if not logger.hasHandlers():
        # add console handler
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # add file handler
        if os.path.exists("./logs") is False:
            os.makedirs("./logs")
        fh = logging.FileHandler("./logs/app.log", mode="w")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    if os.getenv("USE_WANDB", False):
        # add wandb handler
        wh = WandBHandler(level=logging.DEBUG)
        wh.setFormatter(formatter)
        logger.addHandler(wh)

    return logger
