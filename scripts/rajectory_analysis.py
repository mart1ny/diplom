import os
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


class TrajectoryAnalyzer:
    """
    Утилита для накопления траекторий и оценки базовых характеристик движения машин.
    """

    def __init__(self):
        # track_id -> list[(frame_idx, x, y)]
        self.trajectories: dict[int, List[Tuple[int, float, float]]] = {}

    def add_position(self, track_id: int, frame_idx: int, x: float, y: float) -> None:
        """
        Добавляет новую точку траектории для track_id.
        """
        if track_id not in self.trajectories:
            self.trajectories[track_id] = []
        self.trajectories[track_id].append((frame_idx, x, y))

    def get_speed(self, track_id: int) -> List[float]:
        """
        Возвращает список скоростей (пикселей/кадр) для машины.
        """
        traj = self.trajectories.get(track_id, [])
        speeds: List[float] = []
        for i in range(1, len(traj)):
            f0, x0, y0 = traj[i - 1]
            f1, x1, y1 = traj[i]
            dt = f1 - f0
            if dt == 0:
                continue
            dist = np.hypot(x1 - x0, y1 - y0)
            speeds.append(dist / dt)
        return speeds

    def get_direction(self, track_id: int) -> List[float]:
        """
        Возвращает список направлений (радианы) для машины.
        """
        traj = self.trajectories.get(track_id, [])
        directions: List[float] = []
        for i in range(1, len(traj)):
            _, x0, y0 = traj[i - 1]
            _, x1, y1 = traj[i]
            directions.append(float(np.arctan2(y1 - y0, x1 - x0)))
        return directions

    def get_last_position(self, track_id: int) -> Optional[Tuple[float, float]]:
        traj = self.trajectories.get(track_id, [])
        if traj:
            _, x, y = traj[-1]
            return x, y
        return None

    def get_conflict_candidates(self, threshold: float = 50.0):
        """
        Возвращает пары машин, находящихся ближе заданного порога.
        """
        last_positions = {tid: self.get_last_position(tid) for tid in self.trajectories}
        tids = list(last_positions.keys())
        candidates = []
        for i in range(len(tids)):
            for j in range(i + 1, len(tids)):
                p1 = last_positions[tids[i]]
                p2 = last_positions[tids[j]]
                if not p1 or not p2:
                    continue
                dist = np.hypot(p1[0] - p2[0], p1[1] - p2[1])
                if dist < threshold:
                    candidates.append((tids[i], tids[j], dist))
        return candidates

    def estimate_velocity(
        self, track_id: int, window: int = 3
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[int]]:
        """
        Возвращает усреднённый вектор скорости и последнюю позицию за последнее окно точек.
        """
        traj = self.trajectories.get(track_id, [])
        if len(traj) < 2:
            return None, None, None

        start_idx = max(1, len(traj) - window)
        velocity_sum = np.zeros(2, dtype=np.float32)
        samples = 0
        for idx in range(start_idx, len(traj)):
            f0, x0, y0 = traj[idx - 1]
            f1, x1, y1 = traj[idx]
            dt = max(1, f1 - f0)
            velocity_sum += np.array([(x1 - x0) / dt, (y1 - y0) / dt], dtype=np.float32)
            samples += 1
        if samples == 0:
            return None, None, None
        velocity = velocity_sum / samples
        last_frame, x_last, y_last = traj[-1]
        position = np.array([x_last, y_last], dtype=np.float32)
        return velocity, position, last_frame


class RiskLSTMModel(nn.Module):
    """
    Простейшая LSTM-модель для дообучения оценки риска по временным рядам.
    """

    def __init__(self, input_size: int = 3, hidden_size: int = 16, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        risk = self.sigmoid(self.fc(last))
        return risk.squeeze(-1)


class RiskAnalyzer:
    """
    Сборная near-miss логика:
    - усреднённые скорости для устойчивости;
    - оценка сближения (closing speed), минимальной дистанции и разницы направлений;
    - комбинированный скоринг + опциональный LSTM.
    """

    def __init__(
        self,
        trajectory_analyzer: TrajectoryAnalyzer,
        ttc_threshold: float = 2.0,
        pet_threshold: float = 2.0,
        lambda_ttc: float = 0.6,
        lambda_pet: float = 0.4,
        lstm_weight: float = 0.3,
        use_lstm: bool = False,
        model_path: Optional[str] = None,
        device: str = "cpu",
        time_horizon: float = 4.0,
        min_conflict_speed: float = 0.25,
        max_closing_speed: float = 15.0,
    ):
        self.ta = trajectory_analyzer
        self.ttc_threshold = ttc_threshold
        self.pet_threshold = pet_threshold
        self.lambda_ttc = lambda_ttc
        self.lambda_pet = lambda_pet
        self.lstm_weight = lstm_weight
        self.use_lstm = use_lstm
        self.device = device
        self.time_horizon = time_horizon
        self.min_conflict_speed = min_conflict_speed
        self.max_closing_speed = max_closing_speed
        self.lstm_model: Optional[RiskLSTMModel] = None

        if self.use_lstm:
            self.lstm_model = RiskLSTMModel().to(self.device)
            self.lstm_model.eval()
            if model_path and os.path.isfile(model_path):
                try:
                    self.lstm_model.load_state_dict(torch.load(model_path, map_location=self.device))
                except Exception:
                    # Если веса не загрузились, продолжаем с дефолтными
                    pass

    def compute_ttc_pet(self, id1: int, id2: int):
        """
        Для обратной совместимости считаем TTC/PET через новые оценки скорости.
        """
        v1, p1, f1 = self.ta.estimate_velocity(id1)
        v2, p2, f2 = self.ta.estimate_velocity(id2)
        if v1 is None or v2 is None or p1 is None or p2 is None:
            return float("inf"), float("inf"), max(f1 or 0, f2 or 0)
        rel_pos = p2 - p1
        rel_vel = v2 - v1
        dist = float(np.linalg.norm(rel_pos))
        rel_speed = float(np.dot(rel_pos, rel_vel) / (np.linalg.norm(rel_pos) + 1e-6))
        ttc = dist / rel_speed if rel_speed > 1e-3 else float("inf")
        rel_vel_norm_sq = float(np.dot(rel_vel, rel_vel))
        if rel_vel_norm_sq > 1e-6:
            t_close = -float(np.dot(rel_pos, rel_vel)) / rel_vel_norm_sq
            pet = t_close if t_close > 0 else float("inf")
        else:
            pet = float("inf")
        return ttc, pet, max(f1 or 0, f2 or 0)

    def _build_sequence(self, id1: int, id2: int, max_len: int = 10):
        seq: List[List[float]] = []
        traj1 = self.ta.trajectories.get(id1, [])
        traj2 = self.ta.trajectories.get(id2, [])
        L = min(len(traj1), len(traj2))
        if L < 2:
            return None
        for i in range(max(1, L - max_len), L):
            f1, x1, y1 = traj1[i]
            f0_1, x0_1, y0_1 = traj1[i - 1]
            s1 = np.hypot(x1 - x0_1, y1 - y0_1) / max(1, f1 - f0_1)

            f2, x2, y2 = traj2[i]
            f0_2, x0_2, y0_2 = traj2[i - 1]
            s2 = np.hypot(x2 - x0_2, y2 - y0_2) / max(1, f2 - f0_2)

            dist = np.hypot(x2 - x1, y2 - y1)
            seq.append([s1, s2, dist])
        if not seq:
            return None
        return torch.tensor([seq], dtype=torch.float32, device=self.device)

    def _relative_metrics(self, id1: int, id2: int):
        v1, p1, f1 = self.ta.estimate_velocity(id1)
        v2, p2, f2 = self.ta.estimate_velocity(id2)
        if v1 is None or v2 is None or p1 is None or p2 is None:
            return None

        rel_pos = p2 - p1
        rel_vel = v2 - v1
        dist = float(np.linalg.norm(rel_pos))
        if dist < 1e-3:
            return None

        rel_pos_unit = rel_pos / (dist + 1e-6)
        closing_speed = -float(np.dot(rel_vel, rel_pos_unit))
        if closing_speed <= self.min_conflict_speed:
            return None

        rel_speed_sq = float(np.dot(rel_vel, rel_vel))
        if rel_speed_sq < self.min_conflict_speed ** 2:
            return None

        ttc = dist / closing_speed if closing_speed > 1e-3 else float("inf")
        t_min = max(0.0, min(self.time_horizon, -float(np.dot(rel_pos, rel_vel)) / (rel_speed_sq + 1e-6)))
        min_distance = float(np.linalg.norm(rel_pos + rel_vel * t_min))

        angle_diff = 0.0
        if np.linalg.norm(v1) > 1e-3 and np.linalg.norm(v2) > 1e-3:
            heading1 = float(np.arctan2(v1[1], v1[0]))
            heading2 = float(np.arctan2(v2[1], v2[0]))
            raw_diff = abs(heading1 - heading2)
            angle_diff = float(min(raw_diff, 2 * np.pi - raw_diff))

        return {
            "ttc": ttc,
            "time_to_min_distance": t_min,
            "min_distance": min_distance,
            "closing_speed": closing_speed,
            "distance": dist,
            "angle_diff": angle_diff,
            "last_frame": max(f1 or 0, f2 or 0),
        }

    def risk_score(self, id1: int, id2: int):
        metrics = self._relative_metrics(id1, id2)
        if metrics is None:
            return 0.0, None

        ttc = metrics["ttc"]
        min_distance = metrics["min_distance"]
        time_to_min = metrics["time_to_min_distance"]
        closing_speed = metrics["closing_speed"]
        angle_diff = metrics["angle_diff"]

        time_factor = 0.0
        if ttc != float("inf"):
            time_factor = max(0.0, (self.time_horizon - ttc) / self.time_horizon)

        # Чем ближе ожидаемая дистанция, тем выше риск
        distance_factor = max(0.0, 1.0 - min_distance / (self.pet_threshold * 30.0 + 1e-6))
        closing_factor = min(1.0, closing_speed / self.max_closing_speed)
        angle_factor = 1.0 - min(angle_diff, np.pi) / np.pi

        base = 0.4 * distance_factor + 0.3 * time_factor + 0.2 * closing_factor + 0.1 * angle_factor
        if time_to_min > self.time_horizon:
            base *= 0.3

        base = float(np.clip(base, 0.0, 1.0))
        if not self.use_lstm or self.lstm_model is None:
            return base, metrics

        seq = self._build_sequence(id1, id2)
        if seq is None:
            return base, metrics
        with torch.no_grad():
            lstm_risk = float(self.lstm_model(seq).cpu().numpy().item())
        risk = float(np.clip((1.0 - self.lstm_weight) * base + self.lstm_weight * lstm_risk, 0.0, 1.0))
        return risk, metrics

    def analyze_and_get_events(self, distance_threshold: float = 50.0, risk_threshold: float = 0.6):
        """
        Возвращает события near-miss с расширенной телеметрией.
        """
        events = []
        candidates = self.ta.get_conflict_candidates(threshold=distance_threshold)
        for id1, id2, dist in candidates:
            risk, metrics = self.risk_score(id1, id2)
            if risk < risk_threshold or metrics is None:
                continue
            ttc, pet, frame = self.compute_ttc_pet(id1, id2)
            severity = "high" if risk >= 0.75 else "medium" if risk >= 0.5 else "low"
            events.append(
                {
                    "frame": int(frame),
                    "id1": int(id1),
                    "id2": int(id2),
                    "distance": float(dist),
                    "ttc": float(ttc) if ttc < float("inf") else None,
                    "pet": float(pet) if pet < float("inf") else None,
                    "risk_score": float(risk),
                    "closing_speed": float(metrics["closing_speed"]),
                    "min_distance": float(metrics["min_distance"]),
                    "time_to_min_distance": float(metrics["time_to_min_distance"]),
                    "angle_diff": float(metrics["angle_diff"]),
                    "severity": severity,
                }
            )
        return events


if __name__ == "__main__":
    analyzer = TrajectoryAnalyzer()
    analyzer.add_position(0, 0, 100, 200)
    analyzer.add_position(0, 1, 110, 210)
    analyzer.add_position(0, 2, 120, 220)
    analyzer.add_position(1, 0, 130, 200)
    analyzer.add_position(1, 1, 120, 205)
    analyzer.add_position(1, 2, 110, 210)

    ra = RiskAnalyzer(analyzer, use_lstm=False)
    events = ra.analyze_and_get_events(distance_threshold=80.0, risk_threshold=0.3)
    print("Detected events:", events)
