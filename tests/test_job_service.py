from __future__ import annotations

import time
from pathlib import Path

from scripts.job_service import JobStatus, JobStore, VideoProcessingJobService


class FakePipeline:
    def __init__(self, should_fail: bool = False):
        self.should_fail = should_fail

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
        if self.should_fail:
            raise RuntimeError("boom")
        return {
            "output_video": str(output_dir / "annotated.mp4"),
            "events_file": str(output_dir / events_filename),
            "frames_processed": 12,
            "events": [],
            "queue_history": [],
            "plan_history": [],
            "logs": [],
            "latest_plan": {"cycle": 60.0, "durations": {"north": 30.0}},
        }


def wait_for_status(service: VideoProcessingJobService, job_id: str, expected: str) -> None:
    for _ in range(40):
        job = service.get_job(job_id)
        if job is not None and job.status == expected:
            return
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not reach status {expected}")


def test_job_service_persists_completed_result(tmp_path: Path) -> None:
    service = VideoProcessingJobService(
        pipeline=FakePipeline(),
        store=JobStore(tmp_path / "jobs"),
        max_workers=1,
    )
    try:
        job = service.submit_job(
            job_id="job-ok",
            source_filename="sample.mp4",
            upload_path=tmp_path / "sample.mp4",
            input_video={"size_bytes": 128},
        )
        wait_for_status(service, job.job_id, JobStatus.COMPLETED)

        saved_job = service.get_job(job.job_id)
        saved_result = service.get_result(job.job_id)
        assert saved_job is not None
        assert saved_job.status == JobStatus.COMPLETED
        assert saved_result is not None
        assert saved_result["frames_processed"] == 12
    finally:
        service.shutdown()


def test_job_service_marks_failed_jobs(tmp_path: Path) -> None:
    service = VideoProcessingJobService(
        pipeline=FakePipeline(should_fail=True),
        store=JobStore(tmp_path / "jobs"),
        max_workers=1,
    )
    try:
        job = service.submit_job(
            job_id="job-fail",
            source_filename="broken.mp4",
            upload_path=tmp_path / "broken.mp4",
            input_video={"size_bytes": 256},
        )
        wait_for_status(service, job.job_id, JobStatus.FAILED)

        saved_job = service.get_job(job.job_id)
        assert saved_job is not None
        assert saved_job.error == "boom"
    finally:
        service.shutdown()


def test_job_store_loaders_cover_missing_and_existing_records(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs")
    assert store.load_job("missing") is None
    assert store.load_result("missing") is None

    job = service_job = VideoProcessingJobService(
        pipeline=FakePipeline(),
        store=store,
        max_workers=1,
    )
    try:
        submitted = service_job.submit_job(
            job_id="job-existing",
            source_filename="sample.mp4",
            upload_path=tmp_path / "sample.mp4",
            input_video={"size_bytes": 64},
        )
        wait_for_status(service_job, submitted.job_id, JobStatus.COMPLETED)
        loaded_jobs = store.load_all_jobs()
        assert "job-existing" in loaded_jobs
        assert store.load_result("job-existing") is not None
    finally:
        service_job.shutdown()
