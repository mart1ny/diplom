from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
        }
        event_data = getattr(record, "event_data", None)
        if isinstance(event_data, dict) and event_data:
            payload["context"] = event_data
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def log_event(
    logger: logging.Logger,
    level: int,
    message: str,
    **fields: object,
) -> None:
    logger.log(level, message, extra={"event_data": fields or None})


def configure_logging(level: str | None = None) -> None:
    """
    Configure root logging once for CLI/API entrypoints.
    """
    resolved_level = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    resolved_format = os.getenv("LOG_FORMAT", "json").lower()
    root_logger = logging.getLogger()
    if root_logger.handlers:
        root_logger.setLevel(resolved_level)
        for handler in root_logger.handlers:
            if resolved_format == "json":
                handler.setFormatter(JsonFormatter())
            else:
                handler.setFormatter(logging.Formatter(DEFAULT_LOG_FORMAT))
        return
    handler = logging.StreamHandler()
    if resolved_format == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter(DEFAULT_LOG_FORMAT))
    root_logger.setLevel(resolved_level)
    root_logger.addHandler(handler)
