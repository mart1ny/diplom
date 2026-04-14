from __future__ import annotations

import pytest

from scripts import video_validation
from scripts.video_validation import (
    VideoValidationError,
    probe_video,
    validate_upload_filename,
    validate_upload_size,
)


def test_validate_upload_filename_rejects_unknown_extension() -> None:
    with pytest.raises(VideoValidationError) as exc_info:
        validate_upload_filename("sample.exe")
    assert exc_info.value.status_code == 415


def test_validate_upload_size_rejects_large_files(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(video_validation, "MAX_UPLOAD_SIZE_BYTES", 16)
    with pytest.raises(VideoValidationError) as exc_info:
        validate_upload_size(32)
    assert exc_info.value.status_code == 413


def test_probe_video_rejects_too_long_video(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setattr(video_validation, "MAX_VIDEO_DURATION_SECONDS", 10)

    class FakeCapture:
        def isOpened(self) -> bool:
            return True

        def get(self, prop: int) -> float:
            values = {
                video_validation.cv2.CAP_PROP_FPS: 10.0,
                video_validation.cv2.CAP_PROP_FRAME_COUNT: 200.0,
                video_validation.cv2.CAP_PROP_FRAME_WIDTH: 1280.0,
                video_validation.cv2.CAP_PROP_FRAME_HEIGHT: 720.0,
            }
            return values.get(prop, 0.0)

        def release(self) -> None:
            return None

    monkeypatch.setattr(video_validation.cv2, "VideoCapture", lambda _: FakeCapture())
    fake_video = tmp_path / "sample.mp4"
    fake_video.write_bytes(b"not-a-real-video")

    with pytest.raises(VideoValidationError) as exc_info:
        probe_video(fake_video)

    assert exc_info.value.status_code == 400
    assert "слишком длинное" in exc_info.value.detail
