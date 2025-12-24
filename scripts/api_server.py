from __future__ import annotations

import uuid
from pathlib import Path
from typing import Dict, Optional

import logging

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

try:  # pragma: no cover
    from scripts.pipeline_runner import TrafficPipeline
except ImportError:  # pragma: no cover
    from pipeline_runner import TrafficPipeline

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
RESULTS_DIR = BASE_DIR / "results"


logger = logging.getLogger("traffic_api")

app = FastAPI(title="Traffic Optimization API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline: Optional[TrafficPipeline] = None


def _ensure_dirs() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


_ensure_dirs()
app.mount("/results", StaticFiles(directory=RESULTS_DIR), name="results")
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")


def _build_summary(result: Dict[str, object]) -> Dict[str, object]:
    queue_history = result.get("queue_history", [])  # type: ignore[arg-type]
    plan_history = result.get("plan_history", [])  # type: ignore[arg-type]
    events = result.get("events", [])  # type: ignore[arg-type]

    max_queues: Dict[str, int] = {}
    for entry in queue_history:
        for approach, value in entry["queues"].items():
            prev = max_queues.get(approach, 0)
            max_queues[approach] = max(prev, int(value))

    last_plan = result.get("latest_plan") or {}
    latest_cycle = None
    greens = {}
    if isinstance(last_plan, dict):
        greens = last_plan.get("greens", {}) or {}
        latest_cycle = last_plan.get("cycle")

    risk_peaks: Dict[str, float] = {}
    for entry in plan_history:
        for approach, risk in entry.get("risk", {}).items():
            prev = risk_peaks.get(approach, 0.0)
            risk_peaks[approach] = max(prev, float(risk))

    return {
        "total_events": len(events),
        "max_queue_by_approach": max_queues,
        "latest_cycle": latest_cycle,
        "greens": greens,
        "risk_peaks": risk_peaks,
    }


@app.on_event("startup")
async def startup_event():
    global pipeline
    _ensure_dirs()
    pipeline = TrafficPipeline(
        model_path=str(BASE_DIR / "yolov8n.pt"),
        device="cpu",
        cycle_bounds=(50.0, 90.0),
        lambda_risk=5.0,
        risk_threshold=0.6,
        distance_threshold=60.0,
    )


@app.get("/api/health")
async def health():
    return {"status": "ok"}


async def _save_upload(file: UploadFile, target: Path) -> None:
    with target.open("wb") as buffer:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            buffer.write(chunk)


@app.post("/api/process-video")
async def process_video(file: UploadFile = File(...)):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline is initializing")
    suffix = Path(file.filename or "upload").suffix or ".mp4"
    video_id = uuid.uuid4().hex
    upload_path = UPLOAD_DIR / f"{video_id}{suffix}"
    await _save_upload(file, upload_path)

    async def _run():
        return await run_in_threadpool(
            pipeline.process_video,
            str(upload_path),
            RESULTS_DIR,
            False,
            False,
            f"{video_id}_events.jsonl",
            True,
        )

    try:
        result = await _run()
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("Failed to process video %s", video_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    output_video_path = result.get("output_video")
    events_file_path = result.get("events_file")
    output_video_url = f"/results/{Path(output_video_path).name}" if output_video_path else None
    events_file_url = f"/results/{Path(events_file_path).name}" if events_file_path else None

    summary = _build_summary(result)
    return {
        "id": video_id,
        "source_filename": file.filename,
        "output_video": result.get("output_video"),
        "output_video_url": output_video_url,
        "events_file": result.get("events_file"),
        "events_file_url": events_file_url,
        "frames_processed": result.get("frames_processed"),
        "summary": summary,
        "queue_history": result.get("queue_history", []),
        "plan_history": result.get("plan_history", []),
        "events": result.get("events", []),
        "logs": result.get("logs", []),
    }
