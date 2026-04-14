from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from scripts import api_server


class FakePipeline:
    def process_video(
        self,
        source: str,
        output_dir: Path,
        show,
        save_txt,
        events_filename: str,
        collect_metrics,
        mode: str,
        write_video,
    ) -> dict[str, object]:
        return {
            "output_video": str(output_dir / "annotated.mp4"),
            "events_file": str(output_dir / events_filename),
            "frames_processed": 42,
            "latest_plan": {"cycle": 60.0, "greens": {"north": 0.5}},
            "queue_history": [{"frame": 0, "queues": {"north": 1}}],
            "plan_history": [
                {"frame": 0, "plan": {"cycle": 60.0, "greens": {"north": 0.5}}, "risk": {}}
            ],
            "events": [{"frame": 0, "id1": 1, "id2": 2, "risk_score": 0.8, "severity": "high"}],
            "logs": [{"message": "ok", "level": "info", "timestamp": 0.0}],
            "total_events": 1,
            "mode": mode,
        }


def create_client(monkeypatch, tmp_path: Path) -> TestClient:
    monkeypatch.setattr(api_server, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(api_server, "RESULTS_DIR", tmp_path / "results")
    api_server._ensure_dirs()
    monkeypatch.setattr(api_server, "build_pipeline", lambda: FakePipeline())
    return TestClient(api_server.app)


def test_health_reports_pipeline_ready(monkeypatch, tmp_path: Path) -> None:
    with create_client(monkeypatch, tmp_path) as client:
        response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "pipeline_ready": True}


def test_process_video_rejects_invalid_extension(monkeypatch, tmp_path: Path) -> None:
    with create_client(monkeypatch, tmp_path) as client:
        response = client.post(
            "/api/process-video",
            files={"file": ("notes.txt", b"hello", "text/plain")},
        )

    assert response.status_code == 415
    assert "Неподдерживаемый формат" in response.json()["detail"]


def test_process_video_returns_api_payload(monkeypatch, tmp_path: Path) -> None:
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

    with create_client(monkeypatch, tmp_path) as client:
        response = client.post(
            "/api/process-video",
            files={"file": ("sample.mp4", b"fake-video-bytes", "video/mp4")},
        )

    payload = response.json()
    assert response.status_code == 200
    assert payload["frames_processed"] == 42
    assert payload["summary"]["total_events"] == 1
    assert payload["input_video"]["duration_seconds"] == 4.0
    assert payload["output_video_url"] == "/results/annotated.mp4"
