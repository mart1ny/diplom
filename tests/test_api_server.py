from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from scripts import api_server
from scripts.job_service import JobStatus, VideoProcessingJob


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
            "latest_plan": {
                "cycle": 60.0,
                "greens": {"north": 0.5},
                "durations": {"north": 30.0, "east": 20.0},
                "optimizer": "lp",
                "solver_status": "optimal",
                "objective_value": 12.5,
            },
            "queue_history": [{"frame": 0, "queues": {"north": 1}}],
            "plan_history": [
                {
                    "frame": 0,
                    "plan": {
                        "cycle": 60.0,
                        "greens": {"north": 0.5},
                        "durations": {"north": 30.0, "east": 20.0},
                        "optimizer": "lp",
                        "solver_status": "optimal",
                        "objective_value": 12.5,
                    },
                    "risk": {},
                }
            ],
            "events": [{"frame": 0, "id1": 1, "id2": 2, "risk_score": 0.8, "severity": "high"}],
            "logs": [{"message": "ok", "level": "info", "timestamp": 0.0}],
            "total_events": 1,
            "mode": mode,
            "tracking_summary": {
                "unique_tracks": 7,
                "avg_active_tracks": 2.5,
                "peak_active_tracks": 4,
            },
            "scene_calibration": {"name": "intersection-a", "is_calibrated": True},
        }


class FakeJobService:
    def __init__(self, pipeline: FakePipeline, auto_complete: bool = False):
        self.pipeline = pipeline
        self.auto_complete = auto_complete
        self.jobs: dict[str, VideoProcessingJob] = {}
        self.results: dict[str, dict[str, object]] = {}

    def shutdown(self) -> None:
        return None

    def submit_job(
        self,
        *,
        job_id: str,
        source_filename: str,
        upload_path: Path,
        input_video: dict[str, object],
    ) -> VideoProcessingJob:
        job = VideoProcessingJob(
            job_id=job_id,
            status=JobStatus.QUEUED,
            source_filename=source_filename,
            upload_path=str(upload_path),
            created_at=1.0,
            input_video=input_video,
        )
        self.jobs[job_id] = job
        if self.auto_complete:
            self.complete(job_id)
        return job

    def complete(self, job_id: str) -> None:
        job = self.jobs[job_id]
        job.status = JobStatus.COMPLETED
        job.started_at = 2.0
        job.finished_at = 3.0
        self.results[job_id] = self.pipeline.process_video(
            source=job.upload_path,
            output_dir=Path(api_server.RESULTS_DIR) / "jobs" / job_id,
            show=False,
            save_txt=False,
            events_filename=f"{job_id}_events.jsonl",
            collect_metrics=True,
            mode="api",
            write_video=True,
        )

    def get_job(self, job_id: str) -> VideoProcessingJob | None:
        return self.jobs.get(job_id)

    def get_result(self, job_id: str) -> dict[str, object] | None:
        return self.results.get(job_id)

    def list_jobs(self, limit: int = 20) -> list[VideoProcessingJob]:
        return list(self.jobs.values())[:limit]


def create_client(monkeypatch, tmp_path: Path, *, auto_complete: bool = False) -> TestClient:
    api_server.DEFAULT_REGISTRY.reset()
    monkeypatch.setattr(api_server, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(api_server, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(api_server, "JOBS_DIR", tmp_path / "results" / "jobs")
    api_server._ensure_dirs()
    monkeypatch.setattr(api_server, "build_pipeline", lambda: FakePipeline())
    monkeypatch.setattr(
        api_server,
        "build_job_service",
        lambda pipeline: FakeJobService(pipeline, auto_complete=auto_complete),
    )
    return TestClient(api_server.app)


def test_health_reports_pipeline_ready(monkeypatch, tmp_path: Path) -> None:
    with create_client(monkeypatch, tmp_path) as client:
        response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "pipeline_ready": True, "jobs_ready": True}


def test_readiness_reports_service_state(monkeypatch, tmp_path: Path) -> None:
    with create_client(monkeypatch, tmp_path) as client:
        response = client.get("/api/ready")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"


def test_readiness_returns_503_when_pipeline_is_not_initialized(monkeypatch, tmp_path: Path) -> None:
    api_server.DEFAULT_REGISTRY.reset()
    monkeypatch.setattr(api_server, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(api_server, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(api_server, "JOBS_DIR", tmp_path / "results" / "jobs")
    api_server._ensure_dirs()
    monkeypatch.setattr(api_server, "build_pipeline", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(api_server, "build_job_service", lambda pipeline: FakeJobService(pipeline))

    with TestClient(api_server.app) as client:
        response = client.get("/api/ready")

    assert response.status_code == 503
    assert response.json()["detail"]["status"] == "initializing"


def test_process_video_rejects_invalid_extension(monkeypatch, tmp_path: Path) -> None:
    with create_client(monkeypatch, tmp_path) as client:
        response = client.post(
            "/api/process-video",
            files={"file": ("notes.txt", b"hello", "text/plain")},
        )

    assert response.status_code == 415
    assert "Неподдерживаемый формат" in response.json()["detail"]


def test_process_video_submits_background_job(monkeypatch, tmp_path: Path) -> None:
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
    assert response.status_code == 202
    assert payload["status"] == "queued"
    assert payload["status_url"].startswith("/api/jobs/")
    assert payload["input_video"]["duration_seconds"] == 4.0


def test_job_status_returns_completed_result(monkeypatch, tmp_path: Path) -> None:
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
        create_response = client.post(
            "/api/process-video",
            files={"file": ("sample.mp4", b"fake-video-bytes", "video/mp4")},
        )
        job_id = create_response.json()["job_id"]
        assert api_server.job_service is not None
        api_server.job_service.complete(job_id)
        response = client.get(f"/api/jobs/{job_id}")

    payload = response.json()
    result = payload["result"]
    assert response.status_code == 200
    assert payload["status"] == "completed"
    assert result["frames_processed"] == 42
    assert result["summary"]["optimizer"] == "lp"
    assert result["summary"]["solver_status"] == "optimal"
    assert result["summary"]["durations"]["north"] == 30.0
    assert result["summary"]["tracking_summary"]["unique_tracks"] == 7
    assert result["summary"]["scene_calibration"]["is_calibrated"] is True
    assert result["output_video_url"].endswith("/annotated.mp4")


def test_metrics_endpoint_exposes_request_and_upload_metrics(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        api_server,
        "probe_video",
        lambda _: {
            "fps": 25.0,
            "frame_count": 50,
            "width": 640,
            "height": 360,
            "duration_seconds": 2.0,
        },
    )

    with create_client(monkeypatch, tmp_path) as client:
        client.get("/api/health")
        client.post(
            "/api/process-video",
            files={"file": ("sample.mp4", b"fake-video-bytes", "video/mp4")},
        )
        response = client.get("/metrics")

    assert response.status_code == 200
    assert "# TYPE traffic_api_requests_total counter" in response.text
    assert 'traffic_api_upload_size_bytes_count{extension=".mp4"} 1' in response.text
    assert (
        'traffic_api_requests_total{method="GET",path="/api/health",status="200"} 1.0'
        in response.text
    )


def test_api_helpers_cover_summary_truncation_and_env(monkeypatch, tmp_path: Path) -> None:
    summary = api_server._build_summary(
        {
            "queue_history": [{"queues": {"north": 2, "east": 1}}],
            "plan_history": [{"risk": {"north": 0.4}}],
            "latest_plan": None,
        }
    )
    assert summary["latest_cycle"] is None
    assert summary["risk_peaks"]["north"] == pytest.approx(0.4)

    assert api_server._truncate_list([1, 2, 3], 0) == ([1, 2, 3], {"total": 3, "returned": 3})
    assert api_server._truncate_list([1, 2, 3], 2)[0] == [2, 3]

    payload = api_server._build_completed_payload(
        result_id="job-1",
        source_filename="sample.mp4",
        saved_bytes=5,
        video_meta={"fps": 25.0},
        result={
            "output_video": str(tmp_path / "outside.mp4"),
            "events_file": str(tmp_path / "events.jsonl"),
            "frames_processed": 1,
            "queue_history": [],
            "plan_history": [],
            "events": [],
            "logs": [],
            "latest_plan": {},
        },
    )
    assert payload["output_video_url"] == "/results/outside.mp4"

    fake_module = types.ModuleType("scripts.pipeline_runner")

    class StubPipeline:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_module.TrafficPipeline = StubPipeline
    monkeypatch.setitem(sys.modules, "scripts.pipeline_runner", fake_module)
    monkeypatch.setenv("DISTANCE_THRESHOLD_METERS", "12.5")
    monkeypatch.setenv("TRACKER_BACKEND", "simple")
    monkeypatch.setenv("SCENE_CALIBRATION_PATH", "scene.json")
    built = api_server.build_pipeline()
    assert built.kwargs["distance_threshold_meters"] == pytest.approx(12.5)
    assert built.kwargs["tracker_backend"] == "simple"


def test_api_routes_cover_list_jobs_and_errors(monkeypatch, tmp_path: Path) -> None:
    with create_client(monkeypatch, tmp_path) as client:
        list_response = client.get("/api/jobs")
        missing_response = client.get("/api/jobs/missing")

    assert list_response.status_code == 200
    assert list_response.json()["total"] == 0
    assert missing_response.status_code == 404


def test_get_job_returns_500_when_result_missing(monkeypatch, tmp_path: Path) -> None:
    with create_client(monkeypatch, tmp_path) as client:
        assert api_server.job_service is not None
        job = api_server.job_service.submit_job(
            job_id="ready-job",
            source_filename="sample.mp4",
            upload_path=tmp_path / "sample.mp4",
            input_video={"size_bytes": 12},
        )
        job.status = JobStatus.COMPLETED
        response = client.get("/api/jobs/ready-job")

    assert response.status_code == 500
    assert "missing" in response.json()["detail"]


def test_collect_request_metrics_records_server_error(monkeypatch) -> None:
    class DummyRequest:
        method = "GET"

        class URL:
            path = "/boom"

        url = URL()

    async def raising_call_next(_request):
        raise RuntimeError("boom")

    api_server.DEFAULT_REGISTRY.reset()
    with pytest.raises(RuntimeError, match="boom"):
        asyncio.run(api_server.collect_request_metrics(DummyRequest(), raising_call_next))

    rendered = api_server.DEFAULT_REGISTRY.render_prometheus()
    assert 'traffic_api_errors_total{endpoint="/boom",status="500"} 1.0' in rendered
