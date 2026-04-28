from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional

from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

try:  # pragma: no cover
    from scripts.job_service import (
        JobStatus,
        JobStore,
        VideoProcessingJob,
        VideoProcessingJobService,
    )
    from scripts.logging_utils import configure_logging, log_event
    from scripts.observability import (
        DEFAULT_REGISTRY,
        metric_counter,
        metric_gauge,
        metric_histogram,
    )
    from scripts.run_modes import PipelineRunMode
    from scripts.settings import get_settings, load_settings
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
    from logging_utils import configure_logging, log_event
    from observability import (
        DEFAULT_REGISTRY,
        metric_counter,
        metric_gauge,
        metric_histogram,
    )
    from run_modes import PipelineRunMode
    from settings import get_settings, load_settings
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

settings = get_settings()
BASE_DIR = settings.paths.base_dir
UPLOAD_DIR = settings.paths.upload_dir
RESULTS_DIR = settings.paths.results_dir
JOBS_DIR = settings.paths.jobs_dir


configure_logging(settings.logging.level)
logger = logging.getLogger("traffic_api")

pipeline: Optional[TrafficPipeline] = None
job_service: Optional[VideoProcessingJobService] = None
startup_error: Optional[str] = None
MAX_RESPONSE_ITEMS = settings.api.max_response_items


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
        "performance_metrics": result.get("performance_metrics"),
    }


def _truncate_list(items: list, limit: int) -> tuple[list, Dict[str, int]]:
    total = len(items)
    if limit <= 0 or total <= limit:
        return items, {"total": total, "returned": total}
    return items[-limit:], {"total": total, "returned": limit}


def build_pipeline() -> "TrafficPipeline":
    app_settings = load_settings()
    try:  # pragma: no cover
        from scripts.pipeline_runner import TrafficPipeline
    except ImportError:  # pragma: no cover
        from pipeline_runner import TrafficPipeline
    return TrafficPipeline(
        model_path=str(app_settings.model_paths.yolo_model_path),
        device="cpu",
        roi_config=(
            str(app_settings.model_paths.roi_config_path)
            if app_settings.model_paths.roi_config_path is not None
            else None
        ),
        cycle_bounds=(app_settings.optimizer.cycle_min, app_settings.optimizer.cycle_max),
        lambda_risk=app_settings.optimizer.lambda_risk,
        risk_threshold=app_settings.thresholds.risk_threshold,
        distance_threshold=app_settings.thresholds.distance_threshold_px,
        distance_threshold_meters=app_settings.thresholds.distance_threshold_meters,
        tracker_backend=app_settings.tracker.backend,
        scene_calibration_path=(
            str(app_settings.model_paths.scene_calibration_path)
            if app_settings.model_paths.scene_calibration_path is not None
            else None
        ),
        lstm_model_path=(
            str(app_settings.model_paths.lstm_model_path)
            if app_settings.model_paths.lstm_model_path is not None
            else None
        ),
    )


def build_job_service(resolved_pipeline: "TrafficPipeline") -> VideoProcessingJobService:
    app_settings = load_settings()
    return VideoProcessingJobService(
        pipeline=resolved_pipeline,
        store=JobStore(JOBS_DIR),
        max_workers=app_settings.api.video_job_workers,
    )


@asynccontextmanager
async def lifespan(_: FastAPI):
    global job_service, pipeline, startup_error
    _ensure_dirs()
    try:
        pipeline = build_pipeline()
        job_service = build_job_service(pipeline)
        startup_error = None
        log_event(logger, logging.INFO, "Initialized traffic API services")
    except Exception:  # pragma: no cover - protects health endpoint when model init fails
        pipeline = None
        job_service = None
        startup_error = "Failed to initialize traffic pipeline during startup"
        metric_counter(
            "traffic_api_errors_total",
            "Total number of API errors.",
            labels={"endpoint": "startup", "status": "500"},
        )
        logger.exception("Failed to initialize traffic pipeline during startup")
    try:
        yield
    finally:
        if job_service is not None:
            job_service.shutdown()
        job_service = None
        pipeline = None
        startup_error = None


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


@app.middleware("http")
async def collect_request_metrics(request, call_next):
    started_at = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        duration = time.perf_counter() - started_at
        metric_counter(
            "traffic_api_requests_total",
            "Total number of API requests.",
            labels={"method": request.method, "path": request.url.path, "status": "500"},
        )
        metric_counter(
            "traffic_api_errors_total",
            "Total number of API errors.",
            labels={"endpoint": request.url.path, "status": "500"},
        )
        metric_histogram(
            "traffic_api_request_duration_seconds",
            "Duration of HTTP requests handled by the traffic API.",
            duration,
            buckets=(0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 15.0),
            labels={"method": request.method, "path": request.url.path, "status": "500"},
        )
        raise

    duration = time.perf_counter() - started_at
    status = str(response.status_code)
    metric_counter(
        "traffic_api_requests_total",
        "Total number of API requests.",
        labels={"method": request.method, "path": request.url.path, "status": status},
    )
    if response.status_code >= 400:
        metric_counter(
            "traffic_api_errors_total",
            "Total number of API errors.",
            labels={"endpoint": request.url.path, "status": status},
        )
    metric_histogram(
        "traffic_api_request_duration_seconds",
        "Duration of HTTP requests handled by the traffic API.",
        duration,
        buckets=(0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 15.0),
        labels={"method": request.method, "path": request.url.path, "status": status},
    )
    return response


@app.get("/api/health")
async def health():
    metric_gauge(
        "traffic_api_pipeline_ready",
        "Whether the traffic pipeline is initialized.",
        1.0 if pipeline is not None else 0.0,
    )
    metric_gauge(
        "traffic_api_jobs_ready",
        "Whether the background job service is initialized.",
        1.0 if job_service is not None else 0.0,
    )
    return {
        "status": "ok",
        "pipeline_ready": pipeline is not None,
        "jobs_ready": job_service is not None,
    }


@app.get("/api/ready")
async def readiness():
    ready = pipeline is not None and job_service is not None
    payload = {
        "status": "ready" if ready else "initializing",
        "pipeline_ready": pipeline is not None,
        "jobs_ready": job_service is not None,
        "startup_error": startup_error,
    }
    if ready:
        return payload
    raise HTTPException(status_code=503, detail=payload)


@app.get("/metrics")
async def metrics():
    return Response(DEFAULT_REGISTRY.render_prometheus(), media_type="text/plain; version=0.0.4")


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


def _build_job_result_payload(
    job: VideoProcessingJob, result: Dict[str, object]
) -> Dict[str, object]:
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
        metric_counter(
            "traffic_api_errors_total",
            "Total number of API errors.",
            labels={"endpoint": "/api/process-video", "status": "503"},
        )
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
        metric_histogram(
            "traffic_api_upload_size_bytes",
            "Size of uploaded videos accepted by the traffic API.",
            saved_bytes,
            buckets=(1024, 10_240, 102_400, 1_048_576, 5_242_880, 20_971_520, 104_857_600),
            labels={"extension": suffix},
        )
        log_event(
            logger,
            logging.INFO,
            "Accepted uploaded traffic video",
            filename=file.filename or video_id,
            size_bytes=saved_bytes,
            duration_seconds=video_meta.get("duration_seconds"),
        )
    except VideoValidationError as exc:
        upload_path.unlink(missing_ok=True)
        metric_counter(
            "traffic_api_errors_total",
            "Total number of API errors.",
            labels={"endpoint": "/api/process-video", "status": str(exc.status_code)},
        )
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
    except Exception as exc:
        upload_path.unlink(missing_ok=True)
        metric_counter(
            "traffic_api_errors_total",
            "Total number of API errors.",
            labels={"endpoint": "/api/process-video", "status": "400"},
        )
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
    log_event(
        logger,
        logging.INFO,
        "Queued traffic video job",
        job_id=job.job_id,
        filename=job.source_filename,
        size_bytes=saved_bytes,
    )
    return _serialize_job(job)


@app.get("/api/jobs")
async def list_jobs(limit: int = 20):
    if job_service is None:
        metric_counter(
            "traffic_api_errors_total",
            "Total number of API errors.",
            labels={"endpoint": "/api/jobs", "status": "503"},
        )
        raise HTTPException(status_code=503, detail="Job service is initializing")
    jobs = job_service.list_jobs(limit=limit)
    metric_gauge(
        "traffic_api_jobs_listed",
        "Number of jobs returned by the last list operation.",
        len(jobs),
    )
    return {
        "items": [_serialize_job(job) for job in jobs],
        "total": len(jobs),
    }


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    if job_service is None:
        metric_counter(
            "traffic_api_errors_total",
            "Total number of API errors.",
            labels={"endpoint": "/api/jobs/{job_id}", "status": "503"},
        )
        raise HTTPException(status_code=503, detail="Job service is initializing")

    job = job_service.get_job(job_id)
    if job is None:
        metric_counter(
            "traffic_api_errors_total",
            "Total number of API errors.",
            labels={"endpoint": "/api/jobs/{job_id}", "status": "404"},
        )
        raise HTTPException(status_code=404, detail="Job not found")

    payload = _serialize_job(job)
    if job.status == JobStatus.COMPLETED:
        result = job_service.get_result(job_id)
        if result is None:
            metric_counter(
                "traffic_api_errors_total",
                "Total number of API errors.",
                labels={"endpoint": "/api/jobs/{job_id}", "status": "500"},
            )
            raise HTTPException(status_code=500, detail="Job result is missing")
        payload["result"] = _build_job_result_payload(job, result)
    log_event(logger, logging.INFO, "Fetched traffic job status", job_id=job_id, status=job.status)
    return payload
