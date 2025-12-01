import json
import os
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np


class QueueCounter:
    """
    Подсчёт очередей автомобилей внутри заданных ROI (полигоны подходов).
    Считаем машину частью очереди, если она находится внутри ROI несколько последовательных кадров.
    """

    def __init__(
        self,
        roi_polygons: Dict[str, np.ndarray],
        min_frames_inside: int = 2,
        ttl_frames: int = 15,
    ):
        self.roi_polygons = roi_polygons
        self.min_frames_inside = min_frames_inside
        self.ttl_frames = ttl_frames
        # track_id -> {"approach": str|None, "frames_inside": int, "last_seen": int}
        self.track_state: Dict[int, Dict[str, Optional[float]]] = {}

    def _point_to_approach(self, point: Tuple[float, float]) -> Optional[str]:
        x, y = point
        for approach, polygon in self.roi_polygons.items():
            if cv2.pointPolygonTest(polygon, (x, y), False) >= 0:
                return approach
        return None

    def _cleanup(self, current_frame: int) -> None:
        dead = [
            track_id
            for track_id, state in self.track_state.items()
            if current_frame - state["last_seen"] > self.ttl_frames
        ]
        for track_id in dead:
            del self.track_state[track_id]

    def update(self, tracks: Dict[int, Tuple[float, float]], frame_idx: int) -> Dict[str, int]:
        """
        Возвращает {approach: queue_length} для текущего кадра.
        """
        # Обновляем состояния треков
        for track_id, position in tracks.items():
            approach = self._point_to_approach(position)
            state = self.track_state.get(track_id, {"approach": None, "frames_inside": 0, "last_seen": frame_idx})
            if approach is None:
                state["approach"] = None
                state["frames_inside"] = 0
            else:
                if state["approach"] == approach:
                    state["frames_inside"] += 1
                else:
                    state["approach"] = approach
                    state["frames_inside"] = 1
            state["last_seen"] = frame_idx
            self.track_state[track_id] = state

        counts = {approach: 0 for approach in self.roi_polygons.keys()}
        for state in self.track_state.values():
            approach = state["approach"]
            if approach and state["frames_inside"] >= self.min_frames_inside and frame_idx - state["last_seen"] <= 1:
                counts[approach] += 1

        self._cleanup(frame_idx)
        return counts

    def get_track_approach(self, track_id: int) -> Optional[str]:
        state = self.track_state.get(track_id)
        if state:
            return state["approach"]
        return None


DEFAULT_NORMALIZED_ROI = {
    # Координаты указаны в долях от ширины/высоты кадра
    "north": [(0.4, 0.0), (0.6, 0.0), (0.6, 0.25), (0.4, 0.25)],
    "south": [(0.4, 1.0), (0.6, 1.0), (0.6, 0.75), (0.4, 0.75)],
    "east": [(1.0, 0.4), (1.0, 0.6), (0.75, 0.6), (0.75, 0.4)],
    "west": [(0.0, 0.4), (0.0, 0.6), (0.25, 0.6), (0.25, 0.4)],
}


def _scale_polygon(poly: List[Tuple[float, float]], width: int, height: int) -> np.ndarray:
    pts = [(int(x * width), int(y * height)) for x, y in poly]
    return np.array(pts, dtype=np.float32)


def _load_roi_from_file(path: str) -> Dict[str, np.ndarray]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    roi_polygons = {}
    for name, pts in data.items():
        roi_polygons[name] = np.array(pts, dtype=np.float32)
    return roi_polygons


def load_roi_config(roi_path: Optional[str], frame_width: int, frame_height: int) -> Dict[str, np.ndarray]:
    """
    Загружает полигоны ROI из JSON (если указан) или возвращает дефолтные.
    Формат JSON: {"north": [[x1, y1], [x2, y2], ...], ...} в пикселях.
    """
    if roi_path and os.path.isfile(roi_path):
        return _load_roi_from_file(roi_path)
    return {
        name: _scale_polygon(poly, frame_width, frame_height)
        for name, poly in DEFAULT_NORMALIZED_ROI.items()
    }

