from __future__ import annotations

from pathlib import Path

from scripts import api_server
from tests.support import create_client, fixed_uuid, load_snapshot, normalize_api_payload


def _submit_video(client) -> str:
    response = client.post(
        "/api/process-video",
        files={"file": ("sample.mp4", b"fake-video-bytes", "video/mp4")},
    )
    assert response.status_code == 202
    return response.json()["job_id"]


def test_completed_job_response_matches_snapshot(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        api_server,
        "probe_video",
        lambda _: {
            "fps": 25.0,
            "frame_count": 100,
            "width": 1280,
            "height": 720,
            "duration_seconds": 4.0,
        },
    )
    monkeypatch.setattr(api_server.uuid, "uuid4", lambda: fixed_uuid())

    with create_client(monkeypatch, tmp_path, auto_complete=True) as client:
        job_id = _submit_video(client)
        response = client.get(f"/api/jobs/{job_id}")

    assert response.status_code == 200
    assert normalize_api_payload(response.json()) == load_snapshot("api_completed_job.json")


def test_job_list_response_matches_snapshot(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        api_server,
        "probe_video",
        lambda _: {
            "fps": 25.0,
            "frame_count": 100,
            "width": 1280,
            "height": 720,
            "duration_seconds": 4.0,
        },
    )
    monkeypatch.setattr(api_server.uuid, "uuid4", lambda: fixed_uuid())

    with create_client(monkeypatch, tmp_path, auto_complete=False) as client:
        _submit_video(client)
        response = client.get("/api/jobs")

    assert response.status_code == 200
    assert normalize_api_payload(response.json()) == load_snapshot("api_job_list.json")
