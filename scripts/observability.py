from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, Optional, Tuple


def _normalize_labels(labels: Optional[Mapping[str, object]]) -> Tuple[Tuple[str, str], ...]:
    if not labels:
        return ()
    return tuple(sorted((str(key), str(value)) for key, value in labels.items()))


def _render_labels(labels: Tuple[Tuple[str, str], ...]) -> str:
    if not labels:
        return ""
    rendered = ",".join(f'{key}="{value}"' for key, value in labels)
    return f"{{{rendered}}}"


@dataclass
class CounterMetric:
    description: str
    values: Dict[Tuple[Tuple[str, str], ...], float] = field(default_factory=dict)


@dataclass
class GaugeMetric:
    description: str
    values: Dict[Tuple[Tuple[str, str], ...], float] = field(default_factory=dict)


@dataclass
class HistogramValue:
    count: int = 0
    total: float = 0.0
    buckets: Dict[float, int] = field(default_factory=dict)


@dataclass
class HistogramMetric:
    description: str
    bucket_bounds: Tuple[float, ...]
    values: Dict[Tuple[Tuple[str, str], ...], HistogramValue] = field(default_factory=dict)


class MetricsRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: Dict[str, CounterMetric] = {}
        self._gauges: Dict[str, GaugeMetric] = {}
        self._histograms: Dict[str, HistogramMetric] = {}

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()

    def counter(
        self,
        name: str,
        description: str,
        *,
        amount: float = 1.0,
        labels: Optional[Mapping[str, object]] = None,
    ) -> None:
        with self._lock:
            metric = self._counters.setdefault(name, CounterMetric(description=description))
            key = _normalize_labels(labels)
            metric.values[key] = metric.values.get(key, 0.0) + float(amount)

    def gauge(
        self,
        name: str,
        description: str,
        value: float,
        *,
        labels: Optional[Mapping[str, object]] = None,
    ) -> None:
        with self._lock:
            metric = self._gauges.setdefault(name, GaugeMetric(description=description))
            metric.values[_normalize_labels(labels)] = float(value)

    def histogram(
        self,
        name: str,
        description: str,
        value: float,
        *,
        buckets: Iterable[float],
        labels: Optional[Mapping[str, object]] = None,
    ) -> None:
        bucket_bounds = tuple(sorted(float(bound) for bound in buckets))
        with self._lock:
            metric = self._histograms.setdefault(
                name,
                HistogramMetric(description=description, bucket_bounds=bucket_bounds),
            )
            key = _normalize_labels(labels)
            sample = metric.values.setdefault(
                key,
                HistogramValue(
                    buckets={bound: 0 for bound in metric.bucket_bounds},
                ),
            )
            observed = float(value)
            sample.count += 1
            sample.total += observed
            for bound in metric.bucket_bounds:
                if observed <= bound:
                    sample.buckets[bound] += 1

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "counters": {
                    name: {
                        "description": metric.description,
                        "values": {
                            dict(labels).__repr__(): value
                            for labels, value in metric.values.items()
                        },
                    }
                    for name, metric in self._counters.items()
                },
                "gauges": {
                    name: {
                        "description": metric.description,
                        "values": {
                            dict(labels).__repr__(): value
                            for labels, value in metric.values.items()
                        },
                    }
                    for name, metric in self._gauges.items()
                },
                "histograms": {
                    name: {
                        "description": metric.description,
                        "values": {
                            dict(labels).__repr__(): {
                                "count": sample.count,
                                "sum": sample.total,
                                "buckets": sample.buckets,
                            }
                            for labels, sample in metric.values.items()
                        },
                    }
                    for name, metric in self._histograms.items()
                },
            }

    def render_prometheus(self) -> str:
        with self._lock:
            lines: list[str] = []
            for name, metric in self._counters.items():
                lines.append(f"# HELP {name} {metric.description}")
                lines.append(f"# TYPE {name} counter")
                for labels, value in metric.values.items():
                    lines.append(f"{name}{_render_labels(labels)} {value}")

            for name, metric in self._gauges.items():
                lines.append(f"# HELP {name} {metric.description}")
                lines.append(f"# TYPE {name} gauge")
                for labels, value in metric.values.items():
                    lines.append(f"{name}{_render_labels(labels)} {value}")

            for name, metric in self._histograms.items():
                lines.append(f"# HELP {name} {metric.description}")
                lines.append(f"# TYPE {name} histogram")
                for labels, sample in metric.values.items():
                    running = 0
                    for bound in metric.bucket_bounds:
                        running += sample.buckets.get(bound, 0)
                        bucket_labels = labels + (("le", str(bound)),)
                        lines.append(f"{name}_bucket{_render_labels(bucket_labels)} {running}")
                    inf_labels = labels + (("le", "+Inf"),)
                    lines.append(f"{name}_bucket{_render_labels(inf_labels)} {sample.count}")
                    lines.append(f"{name}_count{_render_labels(labels)} {sample.count}")
                    lines.append(f"{name}_sum{_render_labels(labels)} {sample.total}")

            return "\n".join(lines) + "\n"


DEFAULT_REGISTRY = MetricsRegistry()


def metric_counter(
    name: str,
    description: str,
    *,
    amount: float = 1.0,
    labels: Optional[Mapping[str, object]] = None,
) -> None:
    DEFAULT_REGISTRY.counter(name, description, amount=amount, labels=labels)


def metric_gauge(
    name: str,
    description: str,
    value: float,
    *,
    labels: Optional[Mapping[str, object]] = None,
) -> None:
    DEFAULT_REGISTRY.gauge(name, description, value, labels=labels)


def metric_histogram(
    name: str,
    description: str,
    value: float,
    *,
    buckets: Iterable[float],
    labels: Optional[Mapping[str, object]] = None,
) -> None:
    DEFAULT_REGISTRY.histogram(name, description, value, buckets=buckets, labels=labels)
