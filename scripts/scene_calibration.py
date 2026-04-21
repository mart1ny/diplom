from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np


def _normalize_homography(matrix: Any) -> Optional[np.ndarray]:
    if matrix is None:
        return None
    homography = np.asarray(matrix, dtype=np.float32)
    if homography.shape != (3, 3):
        raise ValueError("homography must be a 3x3 matrix")
    return homography


@dataclass(frozen=True)
class SceneCalibration:
    name: str = "uncalibrated"
    meters_per_pixel: Optional[float] = None
    homography: Optional[np.ndarray] = None
    distance_threshold_meters: Optional[float] = None

    def __post_init__(self) -> None:
        if self.meters_per_pixel is not None and self.meters_per_pixel <= 0:
            raise ValueError("meters_per_pixel must be positive")
        if self.distance_threshold_meters is not None and self.distance_threshold_meters <= 0:
            raise ValueError("distance_threshold_meters must be positive")
        object.__setattr__(self, "homography", _normalize_homography(self.homography))

    @property
    def is_calibrated(self) -> bool:
        return self.meters_per_pixel is not None or self.homography is not None

    def project_point(self, point: np.ndarray) -> np.ndarray:
        p = np.asarray(point, dtype=np.float32).reshape(2)
        if self.homography is None:
            if self.meters_per_pixel is None:
                return p.copy()
            return p * float(self.meters_per_pixel)

        x, y = float(p[0]), float(p[1])
        homogeneous = self.homography @ np.array([x, y, 1.0], dtype=np.float32)
        scale = float(homogeneous[2])
        if abs(scale) < 1e-6:
            raise ValueError("homography projected point to infinity")
        return np.array(
            [float(homogeneous[0] / scale), float(homogeneous[1] / scale)],
            dtype=np.float32,
        )

    def project_displacement(self, origin: np.ndarray, displacement: np.ndarray) -> np.ndarray:
        start = np.asarray(origin, dtype=np.float32).reshape(2)
        delta = np.asarray(displacement, dtype=np.float32).reshape(2)
        end = start + delta
        return self.project_point(end) - self.project_point(start)

    def distance_between(self, point_a: np.ndarray, point_b: np.ndarray) -> Optional[float]:
        if not self.is_calibrated:
            return None
        projected_a = self.project_point(point_a)
        projected_b = self.project_point(point_b)
        return float(np.linalg.norm(projected_b - projected_a))

    def as_metadata(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "name": self.name,
            "is_calibrated": self.is_calibrated,
            "distance_threshold_meters": self.distance_threshold_meters,
        }
        if self.meters_per_pixel is not None:
            metadata["meters_per_pixel"] = float(self.meters_per_pixel)
        if self.homography is not None:
            metadata["homography"] = self.homography.tolist()
        return metadata

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SceneCalibration":
        return cls(
            name=str(payload.get("name") or "scene"),
            meters_per_pixel=(
                float(payload["meters_per_pixel"])
                if payload.get("meters_per_pixel") is not None
                else None
            ),
            homography=payload.get("homography"),
            distance_threshold_meters=(
                float(payload["distance_threshold_meters"])
                if payload.get("distance_threshold_meters") is not None
                else None
            ),
        )


def load_scene_calibration(path: str | Path | None) -> Optional[SceneCalibration]:
    if not path:
        return None
    calibration_path = Path(path)
    with calibration_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError("scene calibration file must contain a JSON object")
    calibration = SceneCalibration.from_dict(payload)
    if not calibration.is_calibrated:
        raise ValueError("scene calibration must define meters_per_pixel or homography")
    return calibration
