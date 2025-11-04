import numpy as np

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

# --- Пример использования ---

if __name__ == "__main__":
    analyzer = TrajectoryAnalyzer()
    # Добавляем позиции для двух машин
    analyzer.add_position(0, 0, 100, 200)
    analyzer.add_position(0, 1, 110, 210)
    analyzer.add_position(0, 2, 120, 220)
    analyzer.add_position(1, 0, 300, 400)
    analyzer.add_position(1, 1, 305, 405)
    analyzer.add_position(1, 2, 310, 410)

    print("Скорости машины 0:", analyzer.get_speed(0))
    print("Направления машины 0:", analyzer.get_direction(0))
    print("Последняя позиция машины 1:", analyzer.get_last_position(1))
    print("Пары кандидатов на конфликт:", analyzer.get_conflict_candidates(threshold=30))


"""
Объяснения, что происходит в trajectory_analysis.py
 1. TrajectoryAnalyzer — класс для хранения и анализа траекторий машин.
 ▫ Для каждого track_id хранится список точек (frame_idx, x, y).
 2. add_position(track_id, frame_idx, x, y) — добавляет новую точку траектории для машины.
 3. get_speed(track_id) — вычисляет мгновенные скорости между точками (расстояние/разница кадров).
 4. get_direction(track_id) — вычисляет направление движения (угол в радианах) между точками.
 5. get_last_position(track_id) — возвращает последнюю позицию машины.
 6. get_conflict_candidates(threshold) — ищет пары машин, которые находятся ближе друг к другу, чем заданный порог (например, 50 пикселей). Это простая эвристика для поиска потенциальных конфликтов.
 7. Пример использования — показано, как добавить позиции, получить скорости, направления, последнюю позицию и найти пары машин, которые могут быть в конфликте.

Как это работает в твоём пайплайне
 • После трекинга (tracking.py) для каждого кадра добавляешь позиции машин в TrajectoryAnalyzer.
 • Анализируешь скорости, направления, ищешь потенциальные конфликты (near-miss).
 • Можно расширить: добавить расчёт TTC/PET, хранить траектории в БД, строить тепловые карты.
 
 
 
 Действие: для каждого track_id добавляется точка (frame_idx, x, y); считает мгновенные скорости, направления, даёт кандидатов на конфликты по расстоянию.
 """