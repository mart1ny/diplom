from __future__ import annotations

import logging

from scripts.logging_utils import DEFAULT_LOG_FORMAT, JsonFormatter, configure_logging, log_event


def test_log_event_attaches_structured_context(caplog) -> None:
    logger = logging.getLogger("test_logger")
    with caplog.at_level(logging.INFO):
        log_event(logger, logging.INFO, "hello", job_id="123", status="ok")

    assert caplog.records[0].event_data == {"job_id": "123", "status": "ok"}


def test_configure_logging_reuses_existing_handler(monkeypatch) -> None:
    root = logging.getLogger()
    original_handlers = list(root.handlers)
    original_level = root.level
    handler = logging.StreamHandler()
    root.handlers = [handler]
    try:
        monkeypatch.setenv("LOG_FORMAT", "plain")
        configure_logging("debug")
        assert root.level == logging.DEBUG
        assert isinstance(root.handlers[0].formatter, logging.Formatter)
        assert root.handlers[0].formatter._fmt == DEFAULT_LOG_FORMAT  # type: ignore[attr-defined]
    finally:
        root.handlers = original_handlers
        root.setLevel(original_level)


def test_configure_logging_creates_json_handler(monkeypatch) -> None:
    root = logging.getLogger()
    original_handlers = list(root.handlers)
    original_level = root.level
    root.handlers = []
    try:
        monkeypatch.setenv("LOG_FORMAT", "json")
        configure_logging("info")
        assert len(root.handlers) == 1
        assert isinstance(root.handlers[0].formatter, JsonFormatter)
    finally:
        for handler in list(root.handlers):
            root.removeHandler(handler)
        root.handlers = original_handlers
        root.setLevel(original_level)
