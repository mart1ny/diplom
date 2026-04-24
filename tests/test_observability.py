from __future__ import annotations

import json
import logging

from scripts.logging_utils import JsonFormatter
from scripts.observability import MetricsRegistry


def test_json_formatter_renders_structured_payload() -> None:
    record = logging.LogRecord(
        name="traffic_api",
        level=logging.INFO,
        pathname=__file__,
        lineno=10,
        msg="Structured event",
        args=(),
        exc_info=None,
    )
    record.event_data = {"job_id": "abc123", "status": "queued"}

    payload = json.loads(JsonFormatter().format(record))

    assert payload["logger"] == "traffic_api"
    assert payload["message"] == "Structured event"
    assert payload["context"]["job_id"] == "abc123"


def test_metrics_registry_renders_prometheus_text() -> None:
    registry = MetricsRegistry()
    registry.counter("requests_total", "Request count.", labels={"path": "/health"})
    registry.gauge("pipeline_ready", "Pipeline readiness.", 1.0)
    registry.histogram(
        "request_duration_seconds",
        "Request duration.",
        0.12,
        buckets=(0.1, 0.5, 1.0),
        labels={"path": "/health"},
    )

    rendered = registry.render_prometheus()

    assert "# TYPE requests_total counter" in rendered
    assert 'requests_total{path="/health"} 1.0' in rendered
    assert "# TYPE pipeline_ready gauge" in rendered
    assert 'request_duration_seconds_bucket{path="/health",le="+Inf"} 1' in rendered
