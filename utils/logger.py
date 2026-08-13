"""Application-wide structured logging system."""

import logging
from config import LOG_DIR

def get_logger(name: str) -> logging.Logger:
    """Retrieve or create a structured logger writing to system.log."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        handler = logging.FileHandler(LOG_DIR / "system.log", encoding="utf-8")
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        # Console handler for development debugging
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        logger.setLevel(logging.INFO)
        logger.propagate = False
        
    return logger
