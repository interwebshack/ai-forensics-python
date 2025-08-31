# ai_forensics/logging.py
"""
Logging setup using Loguru, friendly for concurrent environments.

- Debug toggle
- Human-readable console formatting
"""
from __future__ import annotations

import sys

from loguru import logger


def configure_logging(*, debug: bool = False) -> None:
    """Configure loguru logging sinks.

    Args:
        debug: Enable verbose debug logging.
    """
    logger.remove()
    level = "DEBUG" if debug else "INFO"
    fmt = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> "
        "| <level>{level: <8}</level> "
        "| pid={process} tid={thread} "
        "<cyan>{module}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> "
        "- <level>{message}</level>"
    )
    logger.add(sys.stderr, level=level, format=fmt, enqueue=True, backtrace=debug, diagnose=debug)
