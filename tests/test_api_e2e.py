from __future__ import annotations

from pathlib import Path

from scripts import api_server
from scripts.job_service import JobStore, VideoProcessingJobService
from tests.support import (
    FakeAsyncPipeline,
    create_client,
    fixed_uuid,
    wait_for_job_completion,
    write_test_video,
)


def test_e2e_upload_poll_and_fetch_result(monkeypatch, tmp_path: Path) -> None:
    video_path = write_test_video(tmp_path / "upload.avi", frame_count=6, fps=4.0)
    pipeline = FakeAsyncPipeline()
    monkeypatch.setattr(api_server.uuid, "uuid4", lambda: fixed_uuid("job-e2e"))

    with create_client(
        monkeypatch,
        tmp_path,
        pipeline_factory=lambda: pipeline,
        job_service_factory=lambda resolved_pipeline: VideoProcessingJobService(
            pipeline=resolved_pipeline,
            store=JobStore(tmp_path / "results" / "jobs"),
            max_workers=1,
        ),
    ) as client:
        response = client.post(
            "/api/process-video",
            files={"file": ("upload.avi", video_path.read_bytes(), "video/x-msvideo")},
        )
        assert response.status_code == 202
        queued_payload = response.json()
        assert queued_payload["status"] in {"queued", "running"}
        assert queued_payload["job_id"] == "job-e2e"

        completed_payload = wait_for_job_completion(client, "job-e2e")

    result = completed_payload["result"]
    summary = result["summary"]
    assert completed_payload["status"] == "completed"
    assert result["frames_processed"] == 6
    assert result["input_video"]["frame_count"] == 6
    assert summary["optimizer"] == "lp"
    assert summary["solver_status"] == "optimal"
    assert summary["durations"]["north"] > summary["durations"]["east"]
    assert summary["performance_metrics"]["lp_iterations"] == 1
    assert pipeline.calls and Path(pipeline.calls[0]).exists()
    assert (tmp_path / "results" / "jobs" / "job-e2e" / "job.json").exists()
    assert (tmp_path / "results" / "jobs" / "job-e2e" / "result.json").exists()
