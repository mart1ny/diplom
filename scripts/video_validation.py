from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2

try:  # pragma: no cover
    from scripts.settings import get_settings
except ImportError:  # pragma: no cover
    from settings import get_settings

ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
MAX_UPLOAD_SIZE_BYTES = get_settings().api.max_upload_size_bytes
MAX_VIDEO_DURATION_SECONDS = get_settings().api.max_video_duration_seconds


class VideoValidationError(Exception):
    def __init__(self, detail: str, status_code: int = 400):
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code


def validate_upload_filename(filename: str | None) -> str:
    suffix = Path(filename or "").suffix.lower()
    if not suffix:
        raise VideoValidationError("У файла отсутствует расширение видео.", status_code=400)
    if suffix not in ALLOWED_VIDEO_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_VIDEO_EXTENSIONS))
        raise VideoValidationError(
            f"Неподдерживаемый формат файла '{suffix}'. Разрешены: {allowed}.",
            status_code=415,
        )
    return suffix


def validate_upload_size(size_bytes: int) -> None:
    if size_bytes <= MAX_UPLOAD_SIZE_BYTES:
        return
    max_size_mb = MAX_UPLOAD_SIZE_BYTES / (1024 * 1024)
    raise VideoValidationError(
        f"Файл слишком большой: {size_bytes} байт. Максимальный размер {max_size_mb:.0f} MB.",
        status_code=413,
    )


def probe_video(path: str | Path) -> dict[str, Any]:
    video_path = Path(path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise VideoValidationError(
            "Не удалось открыть видеофайл. Проверьте, что файл не повреждён.", status_code=400
        )

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()

    if width <= 0 or height <= 0:
        raise VideoValidationError("Видео не содержит валидных кадров.", status_code=400)

    duration_seconds = frame_count / fps if fps > 0 and frame_count > 0 else None
    if duration_seconds is not None and duration_seconds > MAX_VIDEO_DURATION_SECONDS:
        raise VideoValidationError(
            "Видео слишком длинное: "
            f"{duration_seconds:.1f} сек. Максимальная длительность {MAX_VIDEO_DURATION_SECONDS} сек.",
            status_code=400,
        )

    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_seconds": duration_seconds,
    }
