from __future__ import annotations

import logging
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

try:  # pragma: no cover
    from scripts.job_service import (
        JobStatus,
        JobStore,
        VideoProcessingJob,
        VideoProcessingJobService,
    )
    from scripts.logging_utils import configure_logging
    from scripts.run_modes import PipelineRunMode
    from scripts.video_validation import (
        MAX_UPLOAD_SIZE_BYTES,
        VideoValidationError,
        probe_video,
        validate_upload_filename,
        validate_upload_size,
    )
except ImportError:  # pragma: no cover
    from job_service import (
        JobStatus,
        JobStore,
        VideoProcessingJob,
        VideoProcessingJobService,
    )
    from logging_utils import configure_logging
    from run_modes import PipelineRunMode
    from video_validation import (
        MAX_UPLOAD_SIZE_BYTES,
        VideoValidationError,
        probe_video,
        validate_upload_filename,
        validate_upload_size,
    )

if TYPE_CHECKING:  # pragma: no cover
    try:
        from scripts.pipeline_runner import TrafficPipeline
    except ImportError:  # pragma: no cover
        from pipeline_runner import TrafficPipeline

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
RESULTS_DIR = BASE_DIR / "results"
JOBS_DIR = RESULTS_DIR / "jobs"


configure_logging(os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("traffic_api")

pipeline: Optional[TrafficPipeline] = None
job_service: Optional[VideoProcessingJobService] = None
MAX_RESPONSE_ITEMS = 200


def _ensure_dirs() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    JOBS_DIR.mkdir(parents=True, exist_ok=True)


_ensure_dirs()


def _build_summary(result: Dict[str, object]) -> Dict[str, object]:
    queue_history = result.get("queue_history", [])  # type: ignore[arg-type]
    plan_history = result.get("plan_history", [])  # type: ignore[arg-type]
    events = result.get("events", [])  # type: ignore[arg-type]
    total_events = int(result.get("total_events") or len(events))

    max_queues: Dict[str, int] = {}
    for entry in queue_history:
        for approach, value in entry["queues"].items():
            prev = max_queues.get(approach, 0)
            max_queues[approach] = max(prev, int(value))

    last_plan = result.get("latest_plan") or {}
    latest_cycle = None
    greens = {}
    durations = {}
    optimizer = None
    solver_status = None
    objective_value = None
    if isinstance(last_plan, dict):
        greens = last_plan.get("greens", {}) or {}
        durations = last_plan.get("durations", {}) or {}
        latest_cycle = last_plan.get("cycle")
        optimizer = last_plan.get("optimizer")
        solver_status = last_plan.get("solver_status")
        objective_value = last_plan.get("objective_value")

    risk_peaks: Dict[str, float] = {}
    for entry in plan_history:
        for approach, risk in entry.get("risk", {}).items():
            prev = risk_peaks.get(approach, 0.0)
            risk_peaks[approach] = max(prev, float(risk))

    return {
        "total_events": total_events,
        "max_queue_by_approach": max_queues,
        "latest_cycle": latest_cycle,
        "greens": greens,
        "durations": durations,
        "risk_peaks": risk_peaks,
        "optimizer": optimizer,
        "solver_status": solver_status,
        "objective_value": objective_value,
        "tracking_summary": result.get("tracking_summary"),
        "scene_calibration": result.get("scene_calibration"),
    }


def _truncate_list(items: list, limit: int) -> tuple[list, Dict[str, int]]:
    total = len(items)
    if limit <= 0 or total <= limit:
        return items, {"total": total, "returned": total}
    return items[-limit:], {"total": total, "returned": limit}


def build_pipeline() -> "TrafficPipeline":
    try:  # pragma: no cover
        from scripts.pipeline_runner import TrafficPipeline
    except ImportError:  # pragma: no cover
        from pipeline_runner import TrafficPipeline
    return TrafficPipeline(
        model_path=str(BASE_DIR / "yolov8n.pt"),
        device="cpu",
        cycle_bounds=(50.0, 90.0),
        lambda_risk=5.0,
        risk_threshold=0.6,
        distance_threshold=60.0,
        distance_threshold_meters=(
            float(os.getenv("DISTANCE_THRESHOLD_METERS"))
            if os.getenv("DISTANCE_THRESHOLD_METERS")
            else None
        ),
        tracker_backend=os.getenv("TRACKER_BACKEND", "bytetrack"),
        scene_calibration_path=os.getenv("SCENE_CALIBRATION_PATH"),
    )


def build_job_service(resolved_pipeline: "TrafficPipeline") -> VideoProcessingJobService:
    max_workers = int(os.getenv("VIDEO_JOB_WORKERS", "1"))
    return VideoProcessingJobService(
        pipeline=resolved_pipeline,
        store=JobStore(JOBS_DIR),
        max_workers=max_workers,
    )


@asynccontextmanager
async def lifespan(_: FastAPI):
    global job_service, pipeline
    _ensure_dirs()
    try:
        pipeline = build_pipeline()
        job_service = build_job_service(pipeline)
    except Exception:  # pragma: no cover - protects health endpoint when model init fails
        pipeline = None
        job_service = None
        logger.exception("Failed to initialize traffic pipeline during startup")
    try:
        yield
    finally:
        if job_service is not None:
            job_service.shutdown()
        job_service = None
        pipeline = None


app = FastAPI(title="Traffic Optimization API", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/results", StaticFiles(directory=RESULTS_DIR), name="results")
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "pipeline_ready": pipeline is not None,
        "jobs_ready": job_service is not None,
    }


async def _save_upload(file: UploadFile, target: Path) -> int:
    total_bytes = 0
    with target.open("wb") as buffer:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            total_bytes += len(chunk)
            validate_upload_size(total_bytes)
            buffer.write(chunk)
    return total_bytes


def _build_completed_payload(
    *,
    result_id: str,
    source_filename: Optional[str],
    saved_bytes: int,
    video_meta: dict[str, object],
    result: Dict[str, object],
) -> Dict[str, object]:
    def to_results_url(path_value: object) -> Optional[str]:
        if not path_value:
            return None
        path = Path(str(path_value))
        try:
            relative = path.resolve().relative_to(RESULTS_DIR.resolve())
        except ValueError:
            relative = Path(path.name)
        return f"/results/{relative.as_posix()}"

    output_video_path = result.get("output_video")
    events_file_path = result.get("events_file")
    output_video_url = to_results_url(output_video_path)
    events_file_url = to_results_url(events_file_path)

    summary = _build_summary(result)
    queue_history, queue_meta = _truncate_list(result.get("queue_history", []), MAX_RESPONSE_ITEMS)
    plan_history, plan_meta = _truncate_list(result.get("plan_history", []), MAX_RESPONSE_ITEMS)
    events, events_meta = _truncate_list(result.get("events", []), MAX_RESPONSE_ITEMS)
    logs, logs_meta = _truncate_list(result.get("logs", []), MAX_RESPONSE_ITEMS)
    return {
        "id": result_id,
        "source_filename": source_filename,
        "output_video": result.get("output_video"),
        "output_video_url": output_video_url,
        "events_file": result.get("events_file"),
        "events_file_url": events_file_url,
        "frames_processed": result.get("frames_processed"),
        "input_video": {
            "size_bytes": saved_bytes,
            **video_meta,
            "max_upload_size_bytes": MAX_UPLOAD_SIZE_BYTES,
        },
        "summary": summary,
        "queue_history": queue_history,
        "plan_history": plan_history,
        "events": events,
        "logs": logs,
        "history_meta": {
            "queue_history": queue_meta,
            "plan_history": plan_meta,
            "events": events_meta,
            "logs": logs_meta,
        },
    }


def _serialize_job(job: VideoProcessingJob) -> Dict[str, object]:
    payload: Dict[str, object] = {
        "job_id": job.job_id,
        "status": job.status,
        "source_filename": job.source_filename,
        "created_at": job.created_at,
        "started_at": job.started_at,
        "finished_at": job.finished_at,
        "error": job.error,
        "input_video": job.input_video,
        "status_url": f"/api/jobs/{job.job_id}",
    }
    if job.status == JobStatus.COMPLETED:
        payload["result_url"] = f"/api/jobs/{job.job_id}"
    return payload


def _build_job_result_payload(job: VideoProcessingJob, result: Dict[str, object]) -> Dict[str, object]:
    input_video = dict(job.input_video)
    saved_bytes = int(input_video.pop("size_bytes", 0))
    return _build_completed_payload(
        result_id=job.job_id,
        source_filename=job.source_filename,
        saved_bytes=saved_bytes,
        video_meta=input_video,
        result=result,
    )


@app.post("/api/process-video", status_code=202)
async def process_video(file: UploadFile = File(...)):
    if pipeline is None or job_service is None:
        raise HTTPException(status_code=503, detail="Pipeline is initializing")
    try:
        suffix = validate_upload_filename(file.filename)
    except VideoValidationError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
    video_id = uuid.uuid4().hex
    upload_path = UPLOAD_DIR / f"{video_id}{suffix}"
    saved_bytes = 0
    try:
        saved_bytes = await _save_upload(file, upload_path)
        video_meta = probe_video(upload_path)
    except VideoValidationError as exc:
        upload_path.unlink(missing_ok=True)
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
    except Exception as exc:
        upload_path.unlink(missing_ok=True)
        logger.exception("Failed to save uploaded video %s", file.filename or video_id)
        raise HTTPException(
            status_code=400, detail="Не удалось обработать загруженный файл как видео."
        ) from exc

    job = job_service.submit_job(
        job_id=video_id,
        source_filename=file.filename or video_id,
        upload_path=upload_path,
        input_video={
            "size_bytes": saved_bytes,
            **video_meta,
            "max_upload_size_bytes": MAX_UPLOAD_SIZE_BYTES,
        },
    )
    return _serialize_job(job)


@app.get("/api/jobs")
async def list_jobs(limit: int = 20):
    if job_service is None:
        raise HTTPException(status_code=503, detail="Job service is initializing")
    jobs = job_service.list_jobs(limit=limit)
    return {
        "items": [_serialize_job(job) for job in jobs],
        "total": len(jobs),
    }


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    if job_service is None:
        raise HTTPException(status_code=503, detail="Job service is initializing")

    job = job_service.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    payload = _serialize_job(job)
    if job.status == JobStatus.COMPLETED:
        result = job_service.get_result(job_id)
        if result is None:
            raise HTTPException(status_code=500, detail="Job result is missing")
        payload["result"] = _build_job_result_payload(job, result)
    return payload
