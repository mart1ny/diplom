from __future__ import annotations

import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Protocol

try:  # pragma: no cover
    from scripts.logging_utils import log_event
    from scripts.observability import metric_gauge, metric_histogram
    from scripts.run_modes import PipelineRunMode
except ImportError:  # pragma: no cover
    from logging_utils import log_event
    from observability import metric_gauge, metric_histogram
    from run_modes import PipelineRunMode


class PipelineJobRunner(Protocol):
    def process_video(
        self,
        source: str,
        output_dir: Path,
        show: Optional[bool] = None,
        save_txt: Optional[bool] = None,
        events_filename: Optional[str] = None,
        collect_metrics: Optional[bool] = None,
        mode: Optional[str] = None,
        write_video: Optional[bool] = None,
    ) -> dict[str, Any]: ...


class JobStatus:
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class VideoProcessingJob:
    job_id: str
    status: str
    source_filename: str
    upload_path: str
    created_at: float
    input_video: dict[str, Any]
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    error: Optional[str] = None
    result_path: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "VideoProcessingJob":
        return cls(**payload)


class JobStore:
    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def job_dir(self, job_id: str) -> Path:
        return self.root_dir / job_id

    def metadata_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "job.json"

    def result_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "result.json"

    def save_job(self, job: VideoProcessingJob) -> None:
        job_dir = self.job_dir(job.job_id)
        job_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path(job.job_id).write_text(
            json.dumps(job.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def save_result(self, job_id: str, result: dict[str, Any]) -> Path:
        job_dir = self.job_dir(job_id)
        job_dir.mkdir(parents=True, exist_ok=True)
        path = self.result_path(job_id)
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def load_job(self, job_id: str) -> Optional[VideoProcessingJob]:
        path = self.metadata_path(job_id)
        if not path.exists():
            return None
        return VideoProcessingJob.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def load_result(self, job_id: str) -> Optional[dict[str, Any]]:
        path = self.result_path(job_id)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def load_all_jobs(self) -> dict[str, VideoProcessingJob]:
        jobs: dict[str, VideoProcessingJob] = {}
        for metadata_path in sorted(self.root_dir.glob("*/job.json")):
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            job = VideoProcessingJob.from_dict(payload)
            jobs[job.job_id] = job
        return jobs


class VideoProcessingJobService:
    def __init__(
        self,
        pipeline: PipelineJobRunner,
        store: JobStore,
        max_workers: int = 1,
    ) -> None:
        self.pipeline = pipeline
        self.store = store
        self.executor = ThreadPoolExecutor(max_workers=max(1, int(max_workers)))
        self.lock = threading.Lock()
        self.jobs = self.store.load_all_jobs()
        self.logger = logging.getLogger("traffic_jobs")
        metric_gauge(
            "traffic_jobs_total",
            "Current number of known traffic video jobs.",
            len(self.jobs),
        )

    def shutdown(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=False)

    def submit_job(
        self,
        *,
        job_id: str,
        source_filename: str,
        upload_path: Path,
        input_video: dict[str, Any],
    ) -> VideoProcessingJob:
        job = VideoProcessingJob(
            job_id=job_id,
            status=JobStatus.QUEUED,
            source_filename=source_filename,
            upload_path=str(upload_path),
            created_at=time.time(),
            input_video=input_video,
        )
        with self.lock:
            self.jobs[job_id] = job
            self.store.save_job(job)
            metric_gauge(
                "traffic_jobs_total",
                "Current number of known traffic video jobs.",
                len(self.jobs),
            )
        log_event(
            self.logger,
            logging.INFO,
            "Submitted traffic video job",
            job_id=job_id,
            source_filename=source_filename,
        )
        self.executor.submit(self._run_job, job_id)
        return job

    def get_job(self, job_id: str) -> Optional[VideoProcessingJob]:
        with self.lock:
            job = self.jobs.get(job_id)
        if job is not None:
            return job
        return self.store.load_job(job_id)

    def get_result(self, job_id: str) -> Optional[dict[str, Any]]:
        return self.store.load_result(job_id)

    def list_jobs(self, limit: int = 20) -> list[VideoProcessingJob]:
        with self.lock:
            jobs = sorted(self.jobs.values(), key=lambda item: item.created_at, reverse=True)
        return jobs[:limit]

    def _run_job(self, job_id: str) -> None:
        with self.lock:
            job = self.jobs[job_id]
            job.status = JobStatus.RUNNING
            job.started_at = time.time()
            self.store.save_job(job)
        log_event(self.logger, logging.INFO, "Started traffic video job", job_id=job_id)

        try:
            output_dir = self.store.job_dir(job_id)
            result = self.pipeline.process_video(
                source=job.upload_path,
                output_dir=output_dir,
                show=False,
                save_txt=False,
                events_filename=f"{job_id}_events.jsonl",
                collect_metrics=True,
                mode=PipelineRunMode.API.value,
                write_video=True,
            )
            result_path = self.store.save_result(job_id, result)
            with self.lock:
                job = self.jobs[job_id]
                job.status = JobStatus.COMPLETED
                job.finished_at = time.time()
                job.result_path = str(result_path)
                self.store.save_job(job)
            duration = max((job.finished_at or time.time()) - (job.started_at or time.time()), 0.0)
            metric_histogram(
                "traffic_job_duration_seconds",
                "Duration of background traffic video jobs.",
                duration,
                buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 300.0),
                labels={"status": JobStatus.COMPLETED},
            )
            log_event(
                self.logger,
                logging.INFO,
                "Completed traffic video job",
                job_id=job_id,
                duration_seconds=round(duration, 4),
                result_path=str(result_path),
            )
        except Exception as exc:  # pragma: no cover - exercised via API/service tests
            with self.lock:
                job = self.jobs[job_id]
                job.status = JobStatus.FAILED
                job.finished_at = time.time()
                job.error = str(exc)
                self.store.save_job(job)
            duration = max((job.finished_at or time.time()) - (job.started_at or time.time()), 0.0)
            metric_histogram(
                "traffic_job_duration_seconds",
                "Duration of background traffic video jobs.",
                duration,
                buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 300.0),
                labels={"status": JobStatus.FAILED},
            )
            log_event(
                self.logger,
                logging.ERROR,
                "Traffic video job failed",
                job_id=job_id,
                duration_seconds=round(duration, 4),
                error=str(exc),
            )
