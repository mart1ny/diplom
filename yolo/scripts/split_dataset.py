import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
import random

def split_dataset():
    """Разделяет processed на train/val (80/20), сохраняя баланс (ex. по классам или near-miss)."""
    images_dir = Path('data/processed/images')
    labels_dir = Path('data/processed/labels')

    # Собираем все изображения (предполагаем raw в одной папке; адаптируй)
    all_images = list(images_dir.rglob('*.jpg'))  # Или из raw после extract
    if not all_images:
        print("Нет изображений! Запусти download_and_convert.py сначала.")
        return

    # Разделение (stratify=None; для баланса по классам — парси labels и используй GroupKFold)
    train_imgs, val_imgs = train_test_split(all_images, test_size=0.2, random_state=42)

    # Создаем папки
    for split, imgs in [('train', train_imgs), ('val', val_imgs)]:
        (images_dir / split).mkdir(exist_ok=True)
        (labels_dir / split).mkdir(exist_ok=True)

        for img_path in imgs:
            # Копируем img
            rel_path = img_path.relative_to(images_dir)
            shutil.copy(img_path, images_dir / split / rel_path.name)

            # Копируем label (тот же имя, но .txt)
            label_path = labels_dir / rel_path.with_suffix('.txt').name
            if label_path.exists():
                shutil.copy(label_path, labels_dir / split / rel_path.with_suffix('.txt').name)
            else:
                # Создаем пустой txt если нет label
                (labels_dir / split / rel_path.with_suffix('.txt').name).touch()

    print(f"Разделено: {len(train_imgs)} train, {len(val_imgs)} val.")

if __name__ == "__main__":
    split_dataset() 