import numpy as np
import cv2
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class TrackState:
    """
    Контейнер для хранения метаданных по отдельному треку.
    """

    kf: cv2.KalmanFilter
    track_id: int
    hits: int = 1
    age: int = 1
    missed: int = 0
    last_position: Tuple[float, float] = None


class SimpleKalmanTracker:
    """
    Трекинг машин с помощью Kalman фильтра (OpenCV) и элементарной ассоциации.
    Добавлены:
        • удаление "мертвых" треков по таймауту,
        • удержание статистики попаданий,
        • жадное сопоставление по матрице расстояний,
        • обнуление пропусков при успешной коррекции.
    """

    def __init__(self, distance_threshold: float = 60.0, max_age: int = 30):
        self.distance_threshold = distance_threshold
        self.max_age = max_age
        self.tracks: Dict[int, TrackState] = {}
        self.next_id = 0

    @staticmethod
    def _create_kalman(x: float, y: float) -> cv2.KalmanFilter:
        kf = cv2.KalmanFilter(4, 2)
        kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                         [0, 1, 0, 0]], np.float32)
        kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                        [0, 1, 0, 1],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]], np.float32)
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
        kf.statePre = np.array([[x], [y], [0], [0]], np.float32)
        kf.statePost = np.array([[x], [y], [0], [0]], np.float32)
        return kf

    def _predict_tracks(self) -> Dict[int, Tuple[float, float]]:
        preds = {}
        for track in self.tracks.values():
            state = track.kf.predict()
            track.age += 1
            x, y = float(state[0][0]), float(state[1][0])
            track.last_position = (x, y)
            preds[track.track_id] = track.last_position
        return preds

    def _greedy_assignment(self, predictions: Dict[int, Tuple[float, float]], detections: List[Tuple[float, float]]):
        if not predictions or not detections:
            return [], list(predictions.keys()), list(range(len(detections)))

        track_ids = list(predictions.keys())
        cost = np.zeros((len(track_ids), len(detections)), dtype=np.float32)
        for i, tid in enumerate(track_ids):
            px, py = predictions[tid]
            for j, det in enumerate(detections):
                dx, dy = det
                cost[i, j] = np.hypot(dx - px, dy - py)

        assigned_tracks = set()
        assigned_dets = set()
        matches = []

        while True:
            min_idx = np.unravel_index(np.argmin(cost), cost.shape)
            min_val = cost[min_idx]
            if min_val > self.distance_threshold:
                break
            row, col = min_idx
            if row in assigned_tracks or col in assigned_dets:
                cost[row, col] = np.inf
                continue
            assigned_tracks.add(row)
            assigned_dets.add(col)
            matches.append((track_ids[row], col))
            cost[row, :] = np.inf
            cost[:, col] = np.inf

        unmatched_tracks = [track_ids[i] for i in range(len(track_ids)) if i not in assigned_tracks]
        unmatched_dets = [j for j in range(len(detections)) if j not in assigned_dets]
        return matches, unmatched_tracks, unmatched_dets

    def _remove_dead_tracks(self):
        dead = [tid for tid, track in self.tracks.items() if track.missed > self.max_age]
        for tid in dead:
            del self.tracks[tid]

    def update(self, detections: List[Tuple[float, float]]):
        """
        detections: список [(x_center, y_center), ...] текущих наблюдений.
        """
        if not self.tracks and detections:
            for det in detections:
                self._start_new_track(det)
            return

        predictions = self._predict_tracks()
        matches, unmatched_tracks, unmatched_dets = self._greedy_assignment(predictions, detections)

        # Корректируем сопоставленные треки
        for track_id, det_idx in matches:
            dx, dy = detections[det_idx]
            measurement = np.array([[np.float32(dx)], [np.float32(dy)]])
            track = self.tracks[track_id]
            track.kf.correct(measurement)
            track.last_position = (dx, dy)
            track.hits += 1
            track.missed = 0

        # Обновляем непривязанные треки (увеличиваем счётчик пропусков)
        for track_id in unmatched_tracks:
            track = self.tracks[track_id]
            track.missed += 1

        # Создаём новые треки для непривязанных детекций
        for det_idx in unmatched_dets:
            self._start_new_track(detections[det_idx])

        self._remove_dead_tracks()

    def _start_new_track(self, detection: Tuple[float, float]):
        x, y = detection
        kf = self._create_kalman(x, y)
        track = TrackState(kf=kf, track_id=self.next_id, last_position=(x, y))
        self.tracks[self.next_id] = track
        self.next_id += 1

    def step(self, detections: List[Tuple[float, float]]):
        """
        Выполняет один кадр: предсказание, ассоциация и коррекция.
        Возвращает dict: track_id -> (x, y) текущей позиции.
        """
        self.update(detections)
        positions = {}
        for track_id, track in self.tracks.items():
            positions[track_id] = track.last_position
        return positions

    def predict(self):
        """
        Возвращает предсказанные позиции всех треков без коррекции.
        """
        return self._predict_tracks()

    def correct(self, track_id, x, y):
        """
        Корректирует трек по новым измерениям.
        """
        if track_id in self.tracks:
            measurement = np.array([[np.float32(x)], [np.float32(y)]])
            track = self.tracks[track_id]
            track.kf.correct(measurement)
            track.last_position = (x, y)

# --- Пример использования ---

if __name__ == "__main__":
    # Пример: получаем координаты машин с inference.py
    # detections = [(x_center1, y_center1), (x_center2, y_center2), ...]
    detections = [(100, 200), (300, 400)]
    tracker = SimpleKalmanTracker()
    tracker.update(detections)

    # Предсказание позиций на следующем кадре
    predictions = tracker.predict()
    print("Predicted positions:", predictions)

    # Коррекция по новым данным
    tracker.correct(0, 105, 205)
    tracker.correct(1, 305, 405)
    print("Corrected positions:", tracker.predict())



"""
Объяснения, что происходит в tracking.py
 1. SimpleKalmanTracker — класс для трекинга машин с помощью Kalman фильтра.
Каждый трек — отдельный KalmanFilter, который хранит положение и скорость объекта.
 2. create_kalman(x, y) — инициализация Kalman фильтра для нового объекта (машины) по центру bbox.
 3. update(detections) — обновляет список треков: предсказывает позиции, жадно сопоставляет детекции с существующими треками по расстоянию и удаляет "мертвые" треки.
 4. predict() — возвращает предсказанные позиции всех треков на следующий кадр (по модели движения).
 5. correct(track_id, x, y) — корректирует трек по новым измерениям (например, если YOLO на следующем кадре снова обнаружил машину).
 6. Пример использования — показано, как инициализировать трекер, обновить его по детекциям, получить предсказания и скорректировать по новым данным.

Как это работает в твоём пайплайне
 • После инференса YOLO (inference.py) получаешь координаты центров bbox машин.
 • Передаёшь их в SimpleKalmanTracker.
• На каждом кадре трекер предсказывает новые позиции машин, сопоставляет их с детекциями и корректирует состояние.
• В дальнейшем можно расширять ассоциацию (например, Hungarian/IoU), хранить траектории, считать скорости, анализировать конфликты.
 
 
 
 Выход трекинга на кадр:
 • tracked_object:
 ▫ track_id: int
 ▫ center: (x, y)
 ▫ speed_pred: опционально (из состояния KF)
 ▫ bbox_xyxy
 ▫ t: frame_idx
 
 """