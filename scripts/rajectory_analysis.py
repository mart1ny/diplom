import numpy as np
import torch
import torch.nn as nn
import os
from typing import Optional

class TrajectoryAnalyzer:
    """
    Класс для анализа траекторий и скоростей машин.
    Хранит историю координат, вычисляет мгновенные скорости, направления, и может оценивать потенциальные конфликты.
    """

    def __init__(self):
        # track_id -> list of (frame_idx, x, y)
        self.trajectories = {}

    def add_position(self, track_id, frame_idx, x, y):
        """
        Добавляет новую точку траектории для машины с track_id.
        """
        if track_id not in self.trajectories:
            self.trajectories[track_id] = []
        self.trajectories[track_id].append((frame_idx, x, y))

    def get_speed(self, track_id):
        """
        Возвращает список скоростей (пикселей/кадр) для машины.
        """
        traj = self.trajectories.get(track_id, [])
        speeds = []
        for i in range(1, len(traj)):
            f0, x0, y0 = traj[i-1]
            f1, x1, y1 = traj[i]
            dt = f1 - f0
            if dt == 0:
                continue
            dist = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)
            speed = dist / dt
            speeds.append(speed)
        return speeds

    def get_direction(self, track_id):
        """
        Возвращает список направлений движения (угол в радианах) для машины.
        """
        traj = self.trajectories.get(track_id, [])
        directions = []
        for i in range(1, len(traj)):
            _, x0, y0 = traj[i-1]
            _, x1, y1 = traj[i]
            dx = x1 - x0
            dy = y1 - y0
            angle = np.arctan2(dy, dx)
            directions.append(angle)
        return directions

    def get_last_position(self, track_id):
        """
        Возвращает последнюю позицию машины.
        """
        traj = self.trajectories.get(track_id, [])
        if traj:
            return traj[-1][1], traj[-1][2]
        return None

    def get_conflict_candidates(self, threshold=50):
        """
        Находит пары машин, которые находятся ближе друг к другу, чем threshold (пиксели).
        Возвращает список пар (track_id1, track_id2, distance).
        """
        last_positions = {tid: self.get_last_position(tid) for tid in self.trajectories}
        candidates = []
        tids = list(last_positions.keys())
        for i in range(len(tids)):
            for j in range(i+1, len(tids)):
                pos1 = last_positions[tids[i]]
                pos2 = last_positions[tids[j]]
                if pos1 and pos2:
                    dist = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                    if dist < threshold:
                        candidates.append((tids[i], tids[j], dist))
        return candidates

class RiskLSTMModel(nn.Module):
    """
    Простейшая LSTM-модель для оценки риска по временным рядам (скорости и дистанции).
    Это заглушка: веса случайные/инициализируются по умолчанию. Для продакшена загрузите обученные веса.
    Вход: последовательность длины T, у каждого шага 3 признака: [speed1, speed2, distance].
    Выход: скалярный риск в диапазоне [0, 1] (через сигмоиду).
    """
    def __init__(self, input_size: int = 3, hidden_size: int = 16, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        # Берём выход последнего шага
        last = out[:, -1, :]
        risk = self.sigmoid(self.fc(last))
        return risk.squeeze(-1)

class RiskAnalyzer:
    """
    Анализ риска: TTC, PET и интеграция простой LSTM-модели.
    Использует TrajectoryAnalyzer для доступа к траекториям.
    """
    def __init__(self, trajectory_analyzer, ttc_threshold: float = 2.0, pet_threshold: float = 2.0,
                 lambda_ttc: float = 0.6, lambda_pet: float = 0.4,
                 lstm_weight: float = 0.3, use_lstm: bool = False, model_path: Optional[str] = None, device: str = "cpu"):
        self.ta = trajectory_analyzer
        self.ttc_threshold = ttc_threshold
        self.pet_threshold = pet_threshold
        self.lambda_ttc = lambda_ttc
        self.lambda_pet = lambda_pet
        self.lstm_weight = lstm_weight
        self.use_lstm = use_lstm
        self.device = device
        self.lstm_model = None
        if self.use_lstm:
            self.lstm_model = RiskLSTMModel().to(self.device)
            self.lstm_model.eval()
            if model_path and os.path.isfile(model_path):
                try:
                    self.lstm_model.load_state_dict(torch.load(model_path, map_location=self.device))
                except Exception:
                    # Если веса не загрузились, продолжаем с дефолтными
                    pass

    def _get_last_two_points(self, track_id):
        traj = self.ta.trajectories.get(track_id, [])
        if len(traj) >= 2:
            return traj[-2], traj[-1]  # (f0,x0,y0), (f1,x1,y1)
        return None, None

    def _estimate_velocity(self, track_id):
        p0, p1 = self._get_last_two_points(track_id)
        if p0 and p1:
            f0, x0, y0 = p0
            f1, x1, y1 = p1
            dt = max(1, (f1 - f0))
            vx = (x1 - x0) / dt
            vy = (y1 - y0) / dt
            return np.array([vx, vy], dtype=np.float32), np.array([x1, y1], dtype=np.float32), f1
        return None, None, None

    def compute_ttc_pet(self, id1: int, id2: int):
        v1, p1, f1 = self._estimate_velocity(id1)
        v2, p2, f2 = self._estimate_velocity(id2)
        if v1 is None or v2 is None or p1 is None or p2 is None:
            return float('inf'), float('inf'), max(f1 or 0, f2 or 0)
        rel_pos = p2 - p1
        rel_vel = v2 - v1
        dist = float(np.linalg.norm(rel_pos))
        # Скорость сближения (проекция относительной скорости на линию соединения)
        rel_speed = float(np.dot(rel_pos, rel_vel) / (np.linalg.norm(rel_pos) + 1e-6))
        # TTC: только если они действительно сближаются
        ttc = dist / rel_speed if rel_speed > 1e-3 else float('inf')
        # PET: приближённо время до точки наим.сближения (relative motion)
        rel_vel_norm_sq = float(np.dot(rel_vel, rel_vel))
        if rel_vel_norm_sq > 1e-6:
            t_close = - float(np.dot(rel_pos, rel_vel)) / rel_vel_norm_sq
            pet = t_close if t_close > 0 else float('inf')
        else:
            pet = float('inf')
        return ttc, pet, max(f1, f2)

    def _build_sequence(self, id1: int, id2: int, max_len: int = 10):
        # Формируем последовательность из последних max_len точек: [speed1, speed2, distance]
        seq = []
        traj1 = self.ta.trajectories.get(id1, [])
        traj2 = self.ta.trajectories.get(id2, [])
        L = min(len(traj1), len(traj2))
        if L < 2:
            return None
        # Берём последние max_len точек синхронно (по индексу, для простоты)
        for i in range(max(1, L - max_len), L):
            f1, x1, y1 = traj1[i]
            f0_1, x0_1, y0_1 = traj1[i - 1]
            s1 = np.sqrt((x1 - x0_1) ** 2 + (y1 - y0_1) ** 2) / max(1, f1 - f0_1)

            f2, x2, y2 = traj2[i]
            f0_2, x0_2, y0_2 = traj2[i - 1]
            s2 = np.sqrt((x2 - x0_2) ** 2 + (y2 - y0_2) ** 2) / max(1, f2 - f0_2)

            dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            seq.append([s1, s2, dist])
        if not seq:
            return None
        x = torch.tensor([seq], dtype=torch.float32, device=self.device)
        return x

    def risk_score(self, id1: int, id2: int):
        ttc, pet, _ = self.compute_ttc_pet(id1, id2)
        # Базовая метрика риска на основе порогов TTC/PET
        base_risk_ttc = (self.ttc_threshold / max(ttc, 1e-3)) if ttc < float('inf') else 0.0
        base_risk_pet = (self.pet_threshold / max(pet, 1e-3)) if pet < float('inf') else 0.0
        base = self.lambda_ttc * base_risk_ttc + self.lambda_pet * base_risk_pet
        base = float(np.clip(base, 0.0, 1.0))
        if not self.use_lstm:
            return base
        # Если LSTM включён, строим последовательность и объединяем с базовой метрикой
        seq = self._build_sequence(id1, id2)
        if seq is None or self.lstm_model is None:
            return base
        with torch.no_grad():
            lstm_risk = float(self.lstm_model(seq).cpu().numpy().item())
        risk = float(np.clip((1.0 - self.lstm_weight) * base + self.lstm_weight * lstm_risk, 0.0, 1.0))
        return risk

    def analyze_and_get_events(self, distance_threshold: float = 50.0, risk_threshold: float = 0.6):
        events = []
        candidates = self.ta.get_conflict_candidates(threshold=distance_threshold)
        for id1, id2, dist in candidates:
            risk = self.risk_score(id1, id2)
            if risk >= risk_threshold:
                ttc, pet, f = self.compute_ttc_pet(id1, id2)
                events.append({
                    "frame": int(f),
                    "id1": int(id1),
                    "id2": int(id2),
                    "distance": float(dist),
                    "ttc": float(ttc) if ttc < float('inf') else None,
                    "pet": float(pet) if pet < float('inf') else None,
                    "risk_score": float(risk),
                })
        return events

if __name__ == "__main__":
    # Пример использования
    analyzer = TrajectoryAnalyzer()
    analyzer.add_position(0, 0, 100, 200)
    analyzer.add_position(0, 1, 110, 210)
    analyzer.add_position(0, 2, 120, 220)
    analyzer.add_position(1, 0, 300, 400)
    analyzer.add_position(1, 1, 305, 405)
    analyzer.add_position(1, 2, 310, 410)

    print("Скорости машины 0:", analyzer.get_speed(0))
    print("Направления машины 0:", analyzer.get_direction(0))
    print("Пары кандидатов на конфликт:", analyzer.get_conflict_candidates(threshold=60))

    risk_analyzer = RiskAnalyzer(analyzer, ttc_threshold=2.0, pet_threshold=2.0, use_lstm=False)
    events = risk_analyzer.analyze_and_get_events(distance_threshold=60.0, risk_threshold=0.6)
    print("Near-miss события:", events)
    print("Последняя позиция машины 1:", analyzer.get_last_position(1))
    print("Пары кандидатов на конфликт:", analyzer.get_conflict_candidates(threshold=30))