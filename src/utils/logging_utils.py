"""Structured logging setup for the lidar-moe-detector package."""

from __future__ import annotations

import logging
import sys


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a logger with a consistent format.

    Call once per module:
        log = get_logger(__name__)
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        # Guard against duplicate handlers: get_logger(__name__) may be
        # called more than once for the same module (e.g. reimported in a
        # notebook), and logging.getLogger() returns the same instance each
        # time by name.
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    # Don't propagate to the root logger, otherwise messages would be
    # emitted twice if the root logger also has a handler configured.
    logger.propagate = False
    return logger
