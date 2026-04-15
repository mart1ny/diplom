from __future__ import annotations

from enum import Enum
from typing import Any


class TrackerBackend(str, Enum):
    SIMPLE = "simple"
    BYTETRACK = "bytetrack"


def normalize_tracker_backend(value: str | TrackerBackend) -> TrackerBackend:
    if isinstance(value, TrackerBackend):
        return value
    return TrackerBackend(str(value).lower())


def _scalar(value: Any) -> float:
    if hasattr(value, "item"):
        return float(value.item())
    if isinstance(value, (list, tuple)):
        return float(value[0])
    return float(value)


def _xyxy_list(box: Any) -> list[float]:
    raw = box.xyxy[0]
    if hasattr(raw, "tolist"):
        return [float(item) for item in raw.tolist()]
    return [float(item) for item in raw]


def detection_centers(results: Any, class_id: int) -> list[tuple[float, float]]:
    detections: list[tuple[float, float]] = []
    for box in getattr(results, "boxes", []):
        if int(_scalar(box.cls)) != class_id:
            continue
        x1, y1, x2, y2 = _xyxy_list(box)
        detections.append(((x1 + x2) / 2.0, (y1 + y2) / 2.0))
    return detections


def tracked_centers(results: Any, class_id: int) -> dict[int, tuple[float, float]]:
    positions: dict[int, tuple[float, float]] = {}
    for box in getattr(results, "boxes", []):
        if int(_scalar(box.cls)) != class_id:
            continue
        track_id = getattr(box, "id", None)
        if track_id is None:
            continue
        x1, y1, x2, y2 = _xyxy_list(box)
        positions[int(_scalar(track_id))] = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    return positions
