import numpy as np

class SimpleKalmanTracker:
    """
    Трекинг машин с помощью Kalman фильтра (OpenCV).
    Для каждого обнаруженного объекта (car) создаётся трек, который обновляется по мере поступления новых координат.
    """

    def __init__(self):
        self.trackers = {}  # id -> KalmanFilter
        self.next_id = 0

    def create_kalman(self, x, y):
        kf = cv2.KalmanFilter(4, 2)
        kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                         [0, 1, 0, 0]], np.float32)
        kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                        [0, 1, 0, 1],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]], np.float32)
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        kf.statePre = np.array([[x], [y], [0], [0]], np.float32)
        return kf

    def update(self, detections):
        """
        detections: список [(x_center, y_center), ...] для машин на кадре
        """
        updated_tracks = {}
        for det in detections:
            # Простой assign: каждый detection — новый трек (для демо)
            kf = self.create_kalman(det[0], det[1])
            updated_tracks[self.next_id] = kf
            self.next_id += 1
        self.trackers = updated_tracks

    def predict(self):
        """
        Возвращает предсказанные позиции всех треков.
        """
        predictions = {}
        for track_id, kf in self.trackers.items():
            pred = kf.predict()
            x, y = pred[0][0], pred[1][0]
            predictions[track_id] = (x, y)
        return predictions

    def correct(self, track_id, x, y):
        """
        Корректирует трек по новым измерениям.
        """
        if track_id in self.trackers:
            measurement = np.array([[np.float32(x)], [np.float32(y)]])
            self.trackers[track_id].correct(measurement)

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
 3. update(detections) — обновляет список треков. В демо-версии каждый detection становится новым треком (для реального пайплайна нужно добавить ассоциацию треков и объектов, например, через Hungarian Algorithm или IoU matching).
 4. predict() — возвращает предсказанные позиции всех треков на следующий кадр (по модели движения).
 5. correct(track_id, x, y) — корректирует трек по новым измерениям (например, если YOLO на следующем кадре снова обнаружил машину).
 6. Пример использования — показано, как инициализировать трекер, обновить его по детекциям, получить предсказания и скорректировать по новым данным.

Как это работает в твоём пайплайне
 • После инференса YOLO (inference.py) получаешь координаты центров bbox машин.
 • Передаёшь их в SimpleKalmanTracker.
 • На каждом кадре трекер предсказывает новые позиции машин и корректирует их по новым детекциям.
 • В дальнейшем можно добавить ассоциацию треков и объектов, хранить траектории, считать скорости, анализировать конфликты.
 
 
 
 Выход трекинга на кадр:
 • tracked_object:
 ▫ track_id: int
 ▫ center: (x, y)
 ▫ speed_pred: опционально (из состояния KF)
 ▫ bbox_xyxy
 ▫ t: frame_idx
 
 """