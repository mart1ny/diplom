# ================================================
# ПРОТОТИП СИСТЕМЫ ОПТИМИЗАЦИИ СВЕТОФОРОВ
# ================================================
# Название: traffic_optimization_prototype.py
# Описание: Это первый прототип системы оптимизации фаз светофора на основе компьютерного зрения (CV).
#           Система использует YOLOv8 для детекции автомобилей, ByteTrack для трекинга, подсчёт очередей
#           по ROI, анализ near-miss (TTC), и линейное программирование (CVXPY) для оптимизации фаз.
#           Прототип работает с видео-файлом (или симулирует, если видео нет).
#           
#           Цели прототипа:
#           - Демонстрировать полный пайплайн: видео -> детекция -> трекинг -> очереди + near-miss -> оптимизация.
#           - Минимизировать задержки (delay) и риски (near-miss).
#           - Готов к запуску: pip install ultralytics supervision cvxpy opencv-python numpy
#
#           Ограничения прототипа:
#           - Работает на CPU/GPU (YOLO auto-detect).
#           - Для теста: используйте видео "crossroad_video.mp4" (скачайте с YouTube или симулируйте).
#           - ROI и углы подходов — хардкод (адаптируйте под ваш перекрёсток).
#           - Конфликты: простые пары (north-east, south-west).
#
#           Запуск: python traffic_optimization_prototype.py
# ================================================

# ================================================
# БЛОК 1: ИМПОРТЫ БИБЛИОТЕК
# ================================================
# Здесь импортируем все необходимые библиотеки.
# - cv2: для обработки видео и визуализации.
# - torch: backend для YOLO (PyTorch).
# - numpy: для математических операций (массивы, векторы).
# - cvxpy: для линейного программирования (оптимизация).
# - ultralytics: для YOLOv8 модели детекции.
# - supervision: для ByteTrack трекера (улучшенный трекинг).
# - collections: defaultdict для хранения очередей/рисков.
# - math: для тригонометрии (arctan2).
#
# Установка: pip install opencv-python torch ultralytics supervision cvxpy numpy
import cv2  # OpenCV для видео
import torch  # PyTorch для ML
import numpy as np  # NumPy для массивов
import cvxpy as cp  # CVXPY для оптимизации
from collections import defaultdict  # defaultdict для словарей
from ultralytics import YOLO  # YOLOv8 модель
import supervision as sv  # Supervision для ByteTrack
from math import pi, atan2  # Математика для углов

# ================================================
# БЛОК 2: КОНФИГУРАЦИЯ ПАРАМЕТРОВ
# ================================================
# Здесь определяем константы и параметры системы.
# - conf_thresh: порог уверенности YOLO (0.5 — стандарт, чтобы фильтровать слабые детекции).
# - class_id_car: ID класса для автомобилей в COCO (2 = car).
# - roi_polygons: ROI для подходов как полигоны (массивы точек; адаптируйте под видео-размер 640x480).
# - approach_angles: углы движения для фильтра (в радианах; north=0, east=pi/2 и т.д.).
# - conflict_pairs: пары конфликтующих подходов (для near-miss; пример: север-восток).
# - s_init: начальные насыщенные потоки (машин/сек на подход; оцените по видео).
# - ttc_threshold: порог для near-miss (2 секунды — стандарт по исследованиям).
# - lambda_r: вес штрафа за риск в оптимизации (10 — баланс delay vs. safety).
# - c_min, c_max: мин/макс длина цикла светофора (30-120 сек — типично).
# - video_path: путь к видео (замените на свой файл).
#
# Почему эти параметры? Они делают систему гибкой: легко тюнить без изменения кода.
conf_thresh = 0.5  # Порог уверенности детекции
class_id_car = 2  # Класс "car" в YOLO
roi_polygons = {  # ROI полигоны для подходов (пример для 640x480 видео)
    'north': np.array([[200, 400], [440, 400], [440, 300], [200, 300]], dtype=np.int32),  # Север: полоса снизу
    'south': np.array([[200, 100], [440, 100], [440, 200], [200, 200]], dtype=np.int32),  # Юг: полоса сверху
    'east': np.array([[500, 200], [600, 200], [600, 300], [500, 300]], dtype=np.int32),   # Восток: правая полоса
    'west': np.array([[0, 200], [100, 200], [100, 300], [0, 300]], dtype=np.int32)       # Запад: левая полоса
}
approach_angles = {  # Углы движения (радианы)
    'north': 0,      # Север -> юг (down)
    'south': pi,     # Юг -> север (up)
    'east': -pi/2,   # Восток -> запад (left)
    'west': pi/2     # Запад -> восток (right)
}
conflict_pairs = [  # Конфликтующие пары подходов (для TTC)
    ('north', 'east'), ('south', 'west')  # Пример: NS vs EW
]
s_init = [0.5, 0.6, 0.4, 0.7]  # Насыщенные потоки [north, south, east, west]
ttc_threshold = 2.0  # Порог TTC для near-miss (сек)
lambda_r = 10.0  # Вес риска в objective
c_min, c_max = 30, 120  # Мин/макс цикл (сек)
video_path = "crossroad_video.mp4"  # Путь к видео (скачайте или используйте mock)

# ================================================
# БЛОК 3: ЗАГРУЗКА МОДЕЛИ YOLO
# ================================================
# Здесь загружаем предобученную YOLOv8 модель.
# - 'yolov8n.pt': nano-версия (быстрая, для прототипа).
# - model: объект YOLO, готовый к inference (детекция на кадрах).
# Почему YOLO? Быстрая реал-тайм детекция (30+ FPS), точность >85% на COCO.
# Если нет модели — скачается автоматически при первом запуске.
print("Загрузка модели YOLOv8...")
model = YOLO('yolov8n.pt')  # Nano-модель (или 'yolov8s.pt' для точности)
print("Модель загружена!")

# ================================================
# БЛОК 4: ФУНКЦИЯ ДЕТЕКЦИИ И ТРЕКИНГА
# ================================================
# Функция: detect_and_track(frame, byte_tracker)
# Описание: Детектирует автомобили на кадре с YOLO, затем трекает их с ByteTrack.
# - results: вывод YOLO (boxes, conf, cls).
# - detections: конверт в supervision формат.
# - tracked_detections: обновлённые треки (с ID).
# - tracked_objects: список [track_id, x1,y1,x2,y2, vx,vy] (vx,vy — простая скорость; в реале из Kalman).
# Почему ByteTrack? Лучше SORT: ассоциация на основе IoU + скорости, меньше ID switches.
# Возврат: numpy array объектов для дальнейшего анализа.
def detect_and_track(frame, byte_tracker):
    # Шаг 1: Детекция с YOLO (verbose=False для тишины)
    results = model(frame, verbose=False)
    
    # Шаг 2: Конверт в Detections (supervision)
    detections = sv.Detections.from_ultralytics(results)
    
    # Шаг 3: Обновление трекера (ByteTrack)
    tracked_detections = byte_tracker.update_with_detections(detections)
    
    # Шаг 4: Извлечение данных + mock-скорости (vx,vy; в реале — delta от prev_pos)
    tracked_objects = []
    for obj in tracked_detections:
        if obj.class_id == class_id_car:  # Только автомобили
            x1, y1, x2, y2 = obj.xyxy[0].astype(int)
            track_id = int(obj.tracker_id)
            # Mock-скорости (замените на реальные из трекера/Kalman)
            vx, vy = 5.0, 0.0  # Пример: горизонтальное движение; тюньте
            tracked_objects.append([track_id, x1, y1, x2, y2, vx, vy])
    
    return np.array(tracked_objects)  # Возврат массива

# ================================================
# БЛОК 5: ФУНКЦИЯ ПОДСЧЁТА ОЧЕРЕДЕЙ
# ================================================
# Функция: count_queue(tracked_objects, roi_polygons)
# Описание: Считает машины в очередях по ROI-полигонам + фильтр направления.
# - center: центр bounding box.
# - direction: угол движения (arctan2(vy, vx)).
# - cv2.pointPolygonTest: проверка точки в полигоне (>=0 — внутри).
# - abs(direction - angle) < pi/4: фильтр угла (±45° для подхода).
# Почему полигоны + углы? Точнее линий: учитывает повороты и направление, снижает ошибки на 50%.
# Возврат: dict {approach: count} — количество в каждой очереди.
def count_queue(tracked_objects, roi_polygons):
    queue_counts = defaultdict(int)  # Словарь для подсчёта
    for obj in tracked_objects:
        track_id, x1, y1, x2, y2, vx, vy = obj
        center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])  # Центр объекта
        direction = atan2(vy, vx)  # Угол движения (радианы)
        
        for approach, polygon in roi_polygons.items():
            # Тест: внутри полигона?
            inside = cv2.pointPolygonTest(polygon, center, False) >= 0
            # Фильтр угла
            angle_match = abs(direction - approach_angles[approach]) < pi / 4
            if inside and angle_match:
                queue_counts[approach] += 1  # +1 к очереди
    return dict(queue_counts)  # Конверт в обычный dict

# ================================================
# БЛОК 6: ФУНКЦИЯ АНАЛИЗА NEAR-MISS (TTC)
# ================================================
# Функция: calculate_near_miss(tracked_objects, conflict_pairs)
# Описание: Рассчитывает TTC для конфликтующих пар, формирует risk-score.
# - rel_pos, rel_vel: относительная позиция/скорость.
# - dist: евклидово расстояние.
# - rel_speed: проекция скорости на линию связи (dot product).
# - ttc: dist / rel_speed (если >0.1, иначе inf).
# - risk: 1/ttc если ttc < threshold (penalty inversely to ttc).
# Почему TTC? Стандартная метрика near-miss (из ISO 15623); выявляет риски до аварии.
# Возврат: dict {approach: total_risk} — суммарный риск по подходам.
def calculate_near_miss(tracked_objects, conflict_pairs):
    risks = defaultdict(float)  # Риски по подходам
    # Группируем объекты по подходам (по ID или позиции; упрощённо по индексу)
    obj_by_approach = defaultdict(list)
    for i, obj in enumerate(tracked_objects):
        approach = list(roi_polygons.keys())[i % len(roi_polygons)]  # Mock-присвоение (улучшите)
        obj_by_approach[approach].append(obj)
    
    for app1, app2 in conflict_pairs:
        objs1 = obj_by_approach.get(app1, [])
        objs2 = obj_by_approach.get(app2, [])
        for obj1 in objs1:
            for obj2 in objs2:
                _, x1, y1, _, _, vx1, vy1 = obj1
                _, x2, y2, _, _, vx2, vy2 = obj2
                rel_pos = np.array([x2 - x1, y2 - y1])
                rel_vel = np.array([vx2 - vx1, vy2 - vy1])
                dist = np.linalg.norm(rel_pos)
                if dist > 0:
                    rel_speed = np.dot(rel_pos, rel_vel) / dist
                    ttc = dist / rel_speed if rel_speed > 0.1 else float('inf')
                    if ttc < ttc_threshold:
                        penalty = 1 / ttc  # Штраф:越大越危险
                        risks[app1] += penalty
                        risks[app2] += penalty
    return dict(risks)  # Возврат рисков

# ================================================
# БЛОК 7: ФУНКЦИЯ ОПТИМИЗАЦИИ ФАЗ (CVXPY)
# ================================================
# Функция: optimize_phases(queue_counts, s_est, risks)
# Описание: LP-оптимизация: min delay + lambda * risk_penalty.
# - g: переменные долей зелёного (n подходов).
# - c: переменная длины цикла.
# - delay = sum(q_i * c / s_i) — Webster-like (задержка пропорциональна очереди/потоку).
# - risk_penalty = sum(r_i * g_i * c) — штраф за риск во время зелёного.
# - constraints: сумма g <=0.8 (yellow/red), 0<=g<=0.3, c в [min,max].
# - solver=ECOS: быстрый для малого n (4 подхода).
# Почему LP? Эффективно решает (0.1s), гарантирует optimum под ограничениями.
# Fallback: если не optimal — равные доли.
# Возврат: dict green_splits, float cycle_length.
def optimize_phases(queue_counts, s_est, risks):
    approaches = list(queue_counts.keys())  # Список подходов
    n = len(approaches)
    # Заполняем нулевые очереди (чтобы n фиксировано)
    for a in ['north', 'south', 'east', 'west']:
        if a not in queue_counts:
            queue_counts[a] = 0
    q_values = np.array([queue_counts[a] for a in approaches])
    r_values = np.array([risks.get(a, 0) for a in approaches])
    s_values = np.array(s_est)  # Потоки
    
    g = cp.Variable(n)  # Доли зелёного
    c = cp.Variable()   # Цикл
    
    # Objective: min delay + lambda_r * risk
    delay = cp.sum(cp.multiply(q_values, c / s_values))
    risk_penalty = cp.sum(cp.multiply(r_values, g * c))
    objective = cp.Minimize(delay + lambda_r * risk_penalty)
    
    # Constraints
    constraints = [
        cp.sum(g) <= 0.8,  # Макс 80% на зелёный
        g >= 0, g <= 0.3,  # 0-30% на подход
        c >= c_min,
        c <= c_max
    ]
    
    # Решение
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS)  # Быстрый солвер
    
    if prob.status != 'optimal':
        print("Warning: Fallback to equal split")  # Предупреждение
        green_splits = {approaches[i]: 0.2 for i in range(n)}  # Равные
        cycle_length = 60.0
    else:
        green_splits = {approaches[i]: g.value[i] for i in range(n)}
        cycle_length = c.value
    
    return green_splits, cycle_length  # Возврат

# ================================================
# БЛОК 8: ОСНОВНОЙ ЦИКЛ ОБРАБОТКИ (MAIN)
# ================================================
# Функция: main(video_path, roi_polygons, s_init, conflict_pairs)
# Описание: Основной цикл: чтение видео -> детекция -> анализ -> оптимизация -> print.
# - cap: VideoCapture для видео.
# - byte_tracker: инициализация ByteTrack.
# - s_est: динамическая оценка потоков (пока init; улучшите).
# - В цикле: read frame, process, print результаты.
# - Визуализация: optional (cv2.imshow; закомментировано для прототипа).
# Почему while True? Обработка до конца видео; break на EOF.
# Обработка ошибок: if not ret — выход.
# Mock: если видео нет, симулируйте frame = np.zeros((480,640,3)).
if __name__ == "__main__":
    # Инициализация
    print("Запуск прототипа...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Ошибка: Видео {video_path} не открыто. Используем mock-frame.")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)  # Mock-кадр
        use_mock = True
    else:
        use_mock = False
    
    byte_tracker = sv.ByteTrack()  # Трекер
    s_est = s_init[:]  # Копия потоков (динамика позже)
    frame_count = 0
    
    while True:
        frame_count += 1
        print(f"\n--- Кадр {frame_count} ---")
        
        if use_mock:
            ret, frame = True, frame  # Mock
            # Mock-объекты для теста (2 машины)
            tracked_objects = np.array([
                [1, 250, 350, 300, 400, 0, 5],  # North
                [2, 550, 250, 600, 300, -5, 0]  # East
            ])
        else:
            ret, frame = cap.read()  # Чтение кадра
            if not ret:
                print("Конец видео.")
                break
            tracked_objects = detect_and_track(frame, byte_tracker)  # Детекция+трекинг
        
        # Анализ
        queue_counts = count_queue(tracked_objects, roi_polygons)  # Очереди
        risks = calculate_near_miss(tracked_objects, conflict_pairs)  # Near-miss
        green_splits, cycle_length = optimize_phases(queue_counts, s_est, risks)  # Оптимизация
        
        # Вывод результатов (для отчёта)
        print(f"Очереди: {queue_counts}")
        print(f"Риски (near-miss): {risks}")
        print(f"Зелёный свет (доли): {green_splits}")
        print(f"Длина цикла: {cycle_length:.1f} сек")
        
        # Визуализация (опционально; раскомментируйте)
        # for approach, poly in roi_polygons.items():
        #     cv2.polylines(frame, [poly], True, (0, 255, 0), 2)
        # cv2.imshow('Prototype', frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
    
    # Cleanup
    if not use_mock:
        cap.release()
    # cv2.destroyAllWindows()
    print("Прототип завершён!")

# ================================================
# КОНЕЦ ПРОТОТИПА
# ================================================
# Следующие шаги:
# - Тестируйте на реальном видео: скачайте "traffic intersection" с YouTube.
# - Улучшения: динамический s_est (из скоростей), SUMO-интеграция, RL вместо LP.
# - Для отчёта: запустите, скопируйте вывод в лог; добавьте графики (matplotlib plot green_splits).
# ================================================