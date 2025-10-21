from ultralytics import YOLO
import yaml

def train_model():
    """Дообучение YOLOv8 на датасете."""
    # Загрузка конфига
    with open('configs/aic24_track2.yaml', 'r') as f:
        data = yaml.safe_load(f)

    # Модель (nano для сервера; s/m для точности)
    model = YOLO('yolov8n.pt')  # Скачает auto

    # Тренировка
    results = model.train(
        data=data,                    # YAML путь
        epochs=50,                    # Из YAML или hardcode
        imgsz=640,
        batch=16,                     # Адаптируй под VRAM сервера (8-16 GB)
        name='aicity_near_miss_det',  # Имя runs/
        plots=True,                   # Графики в runs/
        save=True,
        device=0 if torch.cuda.is_available() else 'cpu',  # GPU на сервере
        conf=0.5,
        classes=[1]                   # Фокус на 'car' (index 1 в names)
    )

    # Валидация
    metrics = model.val()
    print(f"mAP@0.5: {metrics.box.map50:.3f} | Precision: {metrics.box.mp50:.3f} | Цель: >0.90")

    # Экспорт для edge (ONNX/TensorRT)
    model.export(format='onnx')  # Для Jetson позже

    print("Обучение завершено! Модель в runs/detect/train/weights/best.pt")

if __name__ == "__main__":
    import torch  # Для device check
    train_model()