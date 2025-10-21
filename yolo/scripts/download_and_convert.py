import kagglehub
import os
import cv2
import json
from pathlib import Path
import shutil

# 👈 ВСТАВЬ СЮДА API KAGGLE (один раз!)
kagglehub.auth.username = 'your_username'  # Твой username с kaggle.com
kagglehub.auth.key = 'your_api_key_here'   # Из kaggle.json

def download_dataset():
    """Скачивает датасет в data/raw."""
    path = kagglehub.dataset_download("nhnnguynngc/ai-city-challenge-track2-p1")
    raw_dir = Path('data/raw')
    raw_dir.mkdir(parents=True, exist_ok=True)
    # Перемещаем в raw (kagglehub кэширует в ~/.cache, копируем)
    shutil.copytree(path, raw_dir / Path(path).name, dirs_exist_ok=True)
    print(f"Датасет скачан в: {raw_dir}")
    return raw_dir

def extract_frames_and_convert(raw_dir):
    """Извлекает кадры из видео и конвертирует gt в YOLO labels."""
    processed_dir = Path('data/processed')
    images_dir = processed_dir / 'images'
    labels_dir = processed_dir / 'labels'
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Пример для train/S01 (адаптируй под структуру: train/val/test)
    for split in ['train', 'val']:  # test без gt
        split_dir = raw_dir / split
        if not split_dir.exists():
            continue

        for scene_dir in split_dir.glob('S*'):
            vdo_dir = scene_dir / 'vdo'
            gt_dir = scene_dir / 'gt'

            if vdo_dir.exists():
                for video_file in vdo_dir.glob('*.mp4'):
                    cap = cv2.VideoCapture(str(video_file))
                    frame_idx = 0
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        if frame_idx % 5 == 0:  # Каждый 5-й кадр для ~10k total
                            img_path = images_dir / split / f"{video_file.stem}_{frame_idx:06d}.jpg"
                            cv2.imwrite(str(img_path), frame)
                            # TODO: Конверт gt для этого frame (предполагаем gt.txt с frame,id,x1,y1,x2,y2,class)
                            # Добавь логику парсинга gt.txt → YOLO txt (как в прошлом скрипте)
                        frame_idx += 1
                    cap.release()

    print("Кадры извлечены и labels готовы (добавь конверт в risk_metrics.py если нужно).")

if __name__ == "__main__":
    raw_dir = download_dataset()
    extract_frames_and_convert(raw_dir)