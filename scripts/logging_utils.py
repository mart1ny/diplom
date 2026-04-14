from __future__ import annotations

import logging
import os

DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def configure_logging(level: str | None = None) -> None:
    """
    Configure root logging once for CLI/API entrypoints.
    """
    resolved_level = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    root_logger = logging.getLogger()
    if root_logger.handlers:
        root_logger.setLevel(resolved_level)
        return
    logging.basicConfig(level=resolved_level, format=DEFAULT_LOG_FORMAT)
