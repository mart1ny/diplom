import cv2
from ultralytics import YOLO
import numpy as np

# Загрузка модели YOLOv8 (предобученная на COCO, класс 'car' = 2)
model = YOLO('yolov8n.pt')  # Скачает автоматически при первом запуске

# Параметры
source = 'path/to/your/video.mp4'  # Или RTSP: 'rtsp://your_camera_ip:554/stream'
save_path = 'output_video.mp4'     # Куда сохранить аннотированное видео
conf_threshold = 0.5               # Порог уверенности
track_history = 30                 # Длина траектории для визуализации

# Инициализация видео-капчура
cap = cv2.VideoCapture(source)
if not cap.isOpened():
    print("Ошибка: Не удалось открыть видео/стрим!")
    exit()

# Получаем свойства видео для сохранения
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Настройка видео-райтера
out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# Словарь для хранения траекторий (track_id -> список точек)
tracks = {}

print("Запуск детекции... Нажми 'q' для выхода.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Инференс YOLO (детекция + трекинг)
    results = model.track(frame, persist=True, conf=conf_threshold, classes=[2])  # Только 'car'

    # Обработка результатов
    if results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()  # Координаты боксов
        track_ids = results[0].boxes.id.int().cpu().numpy() if results[0].boxes.id is not None else []

        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = map(int, box)
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

            # Рисуем бокс
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Траектория (простая: добавляем центр)
            if track_id not in tracks:
                tracks[track_id] = []
            tracks[track_id].append((center_x, center_y))
            if len(tracks[track_id]) > track_history:
                tracks[track_id].pop(0)

            # Рисуем траекторию
            if len(tracks[track_id]) > 1:
                for i in range(1, len(tracks[track_id])):
                    cv2.line(frame, tracks[track_id][i-1], tracks[track_id][i], (255, 0, 0), 2)

    # Метрики на кадре (пример: кол-во машин)
    num_cars = len(boxes) if 'boxes' in locals() else 0
    cv2.putText(frame, f'Cars: {num_cars} | FPS: {model.predictor.metrics.fps}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Сохранение и показ
    out.write(frame)
    cv2.imshow('YOLO Car Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Очистка
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Видео сохранено: {save_path}")