from __future__ import annotations

import json
import logging
import time
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Iterable

import cv2
import numpy as np
from fastapi.testclient import TestClient

from scripts import api_server
from scripts.job_service import JobStatus, JobStore, VideoProcessingJob, VideoProcessingJobService
from scripts.pipeline_runner import TrafficPipeline
from scripts.scene_calibration import SceneCalibration
from scripts.tracker_backends import TrackerBackend


class EmptyResults:
    boxes: list[object] = []


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
            "performance_metrics": {
                "processing_time_seconds": 1.25,
                "processing_fps": 33.6,
                "avg_inference_time_seconds": 0.0123,
                "total_inference_time_seconds": 0.5166,
                "avg_lp_solve_time_seconds": 0.0045,
                "lp_iterations": 1,
            },
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


def create_client(
    monkeypatch,
    tmp_path: Path,
    *,
    auto_complete: bool = False,
    pipeline_factory: Callable[[], object] | None = None,
    job_service_factory: Callable[[object], object] | None = None,
) -> TestClient:
    api_server.DEFAULT_REGISTRY.reset()
    monkeypatch.setattr(api_server, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(api_server, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(api_server, "JOBS_DIR", tmp_path / "results" / "jobs")
    api_server._ensure_dirs()
    resolved_pipeline_factory = pipeline_factory or (lambda: FakePipeline())
    resolved_job_service_factory = job_service_factory or (
        lambda pipeline: FakeJobService(pipeline, auto_complete=auto_complete)
    )
    monkeypatch.setattr(api_server, "build_pipeline", resolved_pipeline_factory)
    monkeypatch.setattr(api_server, "build_job_service", resolved_job_service_factory)
    return TestClient(api_server.app)


def fixed_uuid(value: str = "job-001") -> SimpleNamespace:
    return SimpleNamespace(hex=value)


def normalize_api_payload(payload: dict[str, object]) -> dict[str, object]:
    def normalize_value(value: object) -> object:
        if isinstance(value, dict):
            normalized_dict = {key: normalize_value(item) for key, item in value.items()}
            if "job_id" in normalized_dict:
                normalized_dict["job_id"] = "<job_id>"
            if "id" in normalized_dict and "output_video" in normalized_dict:
                normalized_dict["id"] = "<job_id>"
            for key in ("status_url", "result_url"):
                if key in normalized_dict:
                    normalized_dict[key] = "/api/jobs/<job_id>"
            for key in ("created_at", "started_at", "finished_at"):
                if key in normalized_dict and normalized_dict[key] is not None:
                    normalized_dict[key] = "<ts>"
            return normalized_dict
        if isinstance(value, list):
            return [normalize_value(item) for item in value]
        if isinstance(value, str) and (
            value.endswith(".mp4")
            or value.endswith(".jsonl")
            or value.endswith(".avi")
            or value.endswith(".mov")
        ):
            path = Path(value)
            if value.startswith("/results/"):
                return f"/results/<job>/{path.name}"
            if value.startswith("/api/jobs/"):
                return "/api/jobs/<job_id>"
            if path.is_absolute():
                return f"<abs>/{path.name}"
        return value

    return normalize_value(deepcopy(payload))


def load_snapshot(name: str) -> dict[str, object]:
    path = Path(__file__).parent / "snapshots" / name
    return json.loads(path.read_text(encoding="utf-8"))


def write_test_video(
    path: Path,
    *,
    frame_count: int = 5,
    frame_size: tuple[int, int] = (100, 100),
    fps: float = 5.0,
) -> Path:
    width, height = frame_size
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError("Failed to create test video fixture.")
    for frame_idx in range(frame_count):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :] = (frame_idx * 20) % 255
        writer.write(frame)
    writer.release()
    return path


def build_stub_pipeline(
    *,
    frame_positions: Iterable[dict[int, tuple[float, float]]],
    roi_config: str | None = None,
    scene_calibration: SceneCalibration | None = None,
    cycle_bounds: tuple[float, float] = (20.0, 80.0),
    lambda_risk: float = 5.0,
    risk_threshold: float = 0.01,
    distance_threshold: float = 30.0,
    distance_threshold_meters: float | None = None,
) -> TrafficPipeline:
    pipeline = TrafficPipeline.__new__(TrafficPipeline)
    pipeline.model = None
    pipeline.device = "cpu"
    pipeline.roi_config = roi_config
    pipeline.cycle_bounds = cycle_bounds
    pipeline.lambda_risk = lambda_risk
    pipeline.risk_threshold = risk_threshold
    pipeline.distance_threshold = distance_threshold
    pipeline.distance_threshold_meters = distance_threshold_meters
    pipeline.use_lstm = False
    pipeline.lstm_model_path = None
    pipeline.tracker_backend = TrackerBackend.SIMPLE
    pipeline.scene_calibration = scene_calibration
    pipeline.demand_forecast_settings = SimpleNamespace(
        enabled=False,
        alpha=1.0,
        window_size=12,
        horizon=3,
        hidden_size=64,
        num_layers=2,
    )
    pipeline.demand_forecaster = None
    pipeline.logger = logging.getLogger("traffic_pipeline_test")

    positions_iter = iter(frame_positions)

    def fake_infer(frame, fallback_tracker):
        return EmptyResults(), next(positions_iter, {})

    pipeline._infer_frame = fake_infer
    return pipeline


class FakeAsyncPipeline:
    def __init__(self) -> None:
        self.calls: list[str] = []

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
        self.calls.append(source)
        output_dir.mkdir(parents=True, exist_ok=True)
        annotated_path = output_dir / "annotated.mp4"
        events_path = output_dir / events_filename
        annotated_path.write_bytes(b"fake-video")
        events_path.write_text('{"frame": 1, "severity": "medium"}\n', encoding="utf-8")
        return {
            "output_video": str(annotated_path),
            "events_file": str(events_path),
            "frames_processed": 6,
            "events": [{"frame": 1, "severity": "medium", "risk_score": 0.61}],
            "total_events": 1,
            "queue_history": [{"frame": 0, "queues": {"north": 2, "east": 1}}],
            "plan_history": [
                {
                    "frame": 0,
                    "queues": {"north": 2, "east": 1},
                    "risk": {"north": 0.2, "east": 0.1},
                    "plan": {
                        "cycle": 58.0,
                        "greens": {"north": 0.55, "east": 0.45},
                        "durations": {"north": 26.4, "east": 21.6},
                        "optimizer": "lp",
                        "solver_status": "optimal",
                        "objective_value": 7.1,
                    },
                }
            ],
            "latest_plan": {
                "cycle": 58.0,
                "greens": {"north": 0.55, "east": 0.45},
                "durations": {"north": 26.4, "east": 21.6},
                "optimizer": "lp",
                "solver_status": "optimal",
                "objective_value": 7.1,
            },
            "logs": [{"level": "info", "message": "done", "timestamp": 1.0}],
            "tracking_summary": {
                "unique_tracks": 3,
                "avg_active_tracks": 2.0,
                "peak_active_tracks": 3,
            },
            "scene_calibration": {"name": "e2e-scene", "is_calibrated": True},
            "performance_metrics": {
                "processing_time_seconds": 0.5,
                "processing_fps": 12.0,
                "avg_inference_time_seconds": 0.01,
                "total_inference_time_seconds": 0.06,
                "avg_lp_solve_time_seconds": 0.004,
                "lp_iterations": 1,
            },
            "mode": mode,
        }


def build_real_job_service(
    tmp_path: Path,
    pipeline_factory: Callable[[], object] | None = None,
) -> Callable[[object], VideoProcessingJobService]:
    def factory(pipeline):
        resolved_pipeline = pipeline_factory() if pipeline_factory is not None else pipeline
        return VideoProcessingJobService(
            pipeline=resolved_pipeline,
            store=JobStore(tmp_path / "results" / "jobs"),
            max_workers=1,
        )

    return factory


def wait_for_job_completion(
    client: TestClient, job_id: str, *, attempts: int = 40
) -> dict[str, object]:
    payload: dict[str, object] = {}
    for _ in range(attempts):
        response = client.get(f"/api/jobs/{job_id}")
        payload = response.json()
        if payload.get("status") == JobStatus.COMPLETED:
            return payload
        time.sleep(0.05)
    raise AssertionError(f"Job {job_id} did not complete in time: {payload}")
