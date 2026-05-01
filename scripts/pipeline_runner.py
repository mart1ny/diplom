from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2

try:  # pragma: no cover
    from ultralytics import YOLO
except ModuleNotFoundError:  # pragma: no cover
    YOLO = None

try:  # pragma: no cover
    from scripts.logging_utils import log_event
    from scripts.observability import metric_gauge, metric_histogram
    from scripts.queue_counter import QueueCounter, load_roi_config
    from scripts.rajectory_analysis import RiskAnalyzer, TrajectoryAnalyzer
    from scripts.risk_mapping import aggregate_risk_by_approach
    from scripts.run_modes import PipelineRunMode, build_run_mode_options
    from scripts.scene_calibration import load_scene_calibration
    from scripts.settings import get_settings
    from scripts.tracker_backends import (
        TrackerBackend,
        detection_centers,
        normalize_tracker_backend,
        tracked_centers,
    )
    from scripts.tracking import SimpleKalmanTracker
    from scripts.traffic_optimizer import DEFAULT_PHASE_CONFIG, PhaseOptimizer
except ImportError:  # pragma: no cover
    from logging_utils import log_event
    from observability import metric_gauge, metric_histogram
    from queue_counter import QueueCounter, load_roi_config
    from rajectory_analysis import RiskAnalyzer, TrajectoryAnalyzer
    from risk_mapping import aggregate_risk_by_approach
    from run_modes import PipelineRunMode, build_run_mode_options
    from scene_calibration import load_scene_calibration
    from settings import get_settings
    from tracker_backends import (
        TrackerBackend,
        detection_centers,
        normalize_tracker_backend,
        tracked_centers,
    )
    from tracking import SimpleKalmanTracker
    from traffic_optimizer import DEFAULT_PHASE_CONFIG, PhaseOptimizer

# COCO class index for 'car'
CAR_CLASS_IDX = 2
MAX_PIPELINE_LOGS = 300
MAX_METRICS_HISTORY = 1000


class NullDemandForecaster:
    def predict(self, history):  # pragma: no cover - helper for tests only
        raise RuntimeError("Demand forecaster is not configured.")


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


class TrafficPipeline:
    """
    Обёртка над пайплайном из inference.py.
    Позволяет переиспользовать его как из CLI, так и из веб-сервиса.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cpu",
        roi_config: Optional[str] = None,
        cycle_bounds: Optional[Tuple[float, float]] = None,
        lambda_risk: Optional[float] = None,
        risk_threshold: Optional[float] = None,
        distance_threshold: Optional[float] = None,
        distance_threshold_meters: Optional[float] = None,
        use_lstm: bool = False,
        lstm_model_path: Optional[str] = None,
        tracker_backend: Optional[str] = None,
        scene_calibration_path: Optional[str] = None,
    ) -> None:
        if YOLO is None:
            raise RuntimeError("Ultralytics is required to run the traffic pipeline.")
        settings = get_settings()
        resolved_model_path = model_path or str(settings.model_paths.yolo_model_path)
        resolved_cycle_bounds = cycle_bounds or (
            settings.optimizer.cycle_min,
            settings.optimizer.cycle_max,
        )
        resolved_lambda_risk = (
            settings.optimizer.lambda_risk if lambda_risk is None else lambda_risk
        )
        resolved_risk_threshold = (
            settings.thresholds.risk_threshold if risk_threshold is None else risk_threshold
        )
        resolved_distance_threshold = (
            settings.thresholds.distance_threshold_px
            if distance_threshold is None
            else distance_threshold
        )
        resolved_distance_threshold_meters = (
            settings.thresholds.distance_threshold_meters
            if distance_threshold_meters is None
            else distance_threshold_meters
        )
        resolved_lstm_model_path = lstm_model_path or (
            str(settings.model_paths.lstm_model_path)
            if settings.model_paths.lstm_model_path is not None
            else None
        )
        resolved_tracker_backend = tracker_backend or settings.tracker.backend
        resolved_scene_calibration_path = scene_calibration_path or (
            str(settings.model_paths.scene_calibration_path)
            if settings.model_paths.scene_calibration_path is not None
            else None
        )
        self.model = YOLO(resolved_model_path)
        self.device = device
        self.roi_config = roi_config
        self.cycle_bounds = resolved_cycle_bounds
        self.lambda_risk = resolved_lambda_risk
        self.risk_threshold = resolved_risk_threshold
        self.distance_threshold = resolved_distance_threshold
        self.distance_threshold_meters = resolved_distance_threshold_meters
        self.use_lstm = use_lstm
        self.lstm_model_path = resolved_lstm_model_path
        self.tracker_backend = normalize_tracker_backend(resolved_tracker_backend)
        self.scene_calibration = load_scene_calibration(resolved_scene_calibration_path)
        self.demand_forecast_settings = settings.demand_forecast
        self.demand_forecaster = self._build_demand_forecaster()
        self.logger = logging.getLogger("traffic_pipeline")
        log_event(
            self.logger,
            logging.INFO,
            "Initialized traffic pipeline",
            model=resolved_model_path,
            device=device,
            roi_config=roi_config or "<default>",
            risk_threshold=resolved_risk_threshold,
            distance_threshold=resolved_distance_threshold,
            distance_threshold_meters=resolved_distance_threshold_meters,
            tracker=self.tracker_backend.value,
            calibration=(
                self.scene_calibration.name if self.scene_calibration is not None else "<none>"
            ),
            demand_forecast_enabled=self.demand_forecast_settings.enabled,
        )

    def _build_demand_forecaster(self):
        forecast_settings = self.demand_forecast_settings
        model_path = forecast_settings.model_path
        scaler_path = forecast_settings.scaler_path
        if (
            not forecast_settings.enabled
            or model_path is None
            or scaler_path is None
            or not model_path.is_file()
            or not scaler_path.is_file()
        ):
            return None
        try:
            from lstm.demand_forecaster import DemandForecaster, feature_vector
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency path
            self.logger.warning("Demand forecaster unavailable: %s", exc)
            return None
        try:
            input_size = len(
                feature_vector(
                    {
                        "timestamp": "2026-01-01T00:00:00Z",
                        "queue_len": 0.0,
                        "risk_score": 0.0,
                        "weekday": 3,
                        "hour": 0,
                        "minute": 0,
                        "is_weekend": False,
                        "is_holiday": False,
                        "weather": "other",
                    }
                )
            )
            return DemandForecaster(
                model_path=model_path,
                scaler_path=scaler_path,
                input_size=input_size,
                horizon=forecast_settings.horizon,
                hidden_size=forecast_settings.hidden_size,
                num_layers=forecast_settings.num_layers,
                device=("cuda" if str(self.device).startswith("cuda") else "cpu"),
            )
        except Exception as exc:  # pragma: no cover - safety fallback
            self.logger.warning("Failed to initialize demand forecaster: %s", exc)
            return None

    def _save_txt(self, results, txt_path: Path) -> None:
        with open(txt_path, "w", encoding="utf-8") as f:
            for box in results.boxes:
                if int(box.cls) != CAR_CLASS_IDX:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                x_center = (x1 + x2) / 2
                y_center = (y1 + y2) / 2
                width = x2 - x1
                height = y2 - y1
                f.write(
                    f"{CAR_CLASS_IDX} {x_center:.2f} {y_center:.2f} "
                    f"{width:.2f} {height:.2f} {conf:.4f}\n"
                )

    def _draw_boxes(self, frame, results):
        for box in results.boxes:
            if int(box.cls) != CAR_CLASS_IDX:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            label = f"car {conf:.2f}"
            if getattr(box, "id", None) is not None:
                track_id = int(box.id.item()) if hasattr(box.id, "item") else int(box.id[0])
                label = f"car #{track_id} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 200, 0),
                2,
            )
        return frame

    def _build_phase_optimizer(self, roi_polygons: Dict[str, Iterable]) -> PhaseOptimizer:
        settings = get_settings()
        from config import (
            MAX_PHASE_DURATION,
            MIN_PHASE_DURATION,
        )  # локальный импорт, чтобы избежать циклов

        phase_config = {}
        for name in roi_polygons.keys():
            base = DEFAULT_PHASE_CONFIG.get(
                name,
                {
                    "phase_type": "vehicle",
                    "min_green": MIN_PHASE_DURATION,
                    "max_green": MAX_PHASE_DURATION,
                    "service_rate": 1.0,
                    "delay_weight": 1.0,
                    "queue_weight": settings.optimizer_weights.queue_weight,
                    "risk_weight": settings.optimizer_weights.risk_weight,
                },
            )
            phase_config[name] = dict(base)

        pedestrian_phase = settings.pedestrian_phase
        if pedestrian_phase.enabled:
            phase_config[pedestrian_phase.name] = {
                "phase_type": "pedestrian",
                "min_green": pedestrian_phase.min_green,
                "max_green": pedestrian_phase.max_green,
                "service_rate": pedestrian_phase.service_rate,
                "delay_weight": pedestrian_phase.delay_weight,
                "queue_weight": pedestrian_phase.queue_weight,
                "risk_weight": pedestrian_phase.risk_weight,
                "base_demand": pedestrian_phase.base_demand,
            }

        return PhaseOptimizer(
            phase_config=phase_config,
            cycle_bounds=self.cycle_bounds,
            lambda_risk=self.lambda_risk,
            demand_forecast_alpha=self.demand_forecast_settings.alpha,
        )

    def _build_forecast_record(
        self,
        *,
        source_label: str,
        approach: str,
        queue_len: float,
        risk_score: float,
        frame_idx: int,
        fps: float,
        started_at: float,
    ) -> Dict[str, object]:
        event_time = datetime.fromtimestamp(
            started_at + (frame_idx / max(fps, 1e-6)),
            tz=timezone.utc,
        )
        return {
            "light_id": source_label,
            "timestamp": event_time.isoformat().replace("+00:00", "Z"),
            "approach": approach,
            "queue_len": float(queue_len),
            "risk_score": float(risk_score),
            "risk": float(risk_score),
            "weekday": int(event_time.weekday()),
            "hour": int(event_time.hour),
            "minute": int(event_time.minute),
            "is_weekend": bool(event_time.weekday() >= 5),
            "is_holiday": False,
            "weather": "other",
        }

    def _predict_demand(
        self,
        histories: Dict[str, List[Dict[str, object]]],
    ) -> Optional[Dict[str, float]]:
        if self.demand_forecaster is None:
            return None
        forecast: Dict[str, float] = {}
        window_size = max(int(self.demand_forecast_settings.window_size), 1)
        for approach, series in histories.items():
            if len(series) < window_size:
                continue
            try:
                prediction = self.demand_forecaster.predict(series[-window_size:])
            except Exception as exc:  # pragma: no cover - runtime fallback
                self.logger.warning("Demand forecast failed for %s: %s", approach, exc)
                continue
            if len(prediction) == 0:
                continue
            forecast[approach] = float(max(prediction[0], 0.0))
        return forecast or None

    def _infer_frame(self, frame, fallback_tracker: SimpleKalmanTracker):
        if self.tracker_backend is TrackerBackend.BYTETRACK:
            results = self.model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                classes=[CAR_CLASS_IDX],
                device=self.device,
                verbose=False,
            )[0]
            positions = tracked_centers(results, CAR_CLASS_IDX)
            detections = detection_centers(results, CAR_CLASS_IDX)
            if not positions and detections:
                self.logger.debug(
                    "ByteTrack returned detections without ids; using simple tracker fallback"
                )
                positions = fallback_tracker.step(detections)
            return results, positions

        results = self.model(frame, classes=[CAR_CLASS_IDX], device=self.device, verbose=False)[0]
        detections = detection_centers(results, CAR_CLASS_IDX)
        return results, fallback_tracker.step(detections)

    def process_image(
        self,
        img_path: str,
        output_dir: Path,
        show: bool = False,
        save_txt: bool = False,
    ) -> Path:
        output_dir = _ensure_dir(output_dir)
        img = cv2.imread(str(img_path))
        results = self.model(img)
        annotated = self._draw_boxes(img.copy(), results[0])
        out_path = output_dir / f"{Path(img_path).stem}_car.jpg"
        cv2.imwrite(str(out_path), annotated)
        if save_txt:
            self._save_txt(results[0], output_dir / f"{Path(img_path).stem}_car.txt")
        if show:
            cv2.imshow("Car Detection", annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return out_path

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
    ) -> Dict[str, object]:
        output_dir = _ensure_dir(output_dir)
        log_entries: List[Dict[str, object]] = []
        resolved_mode = PipelineRunMode.RESEARCH if mode is None else PipelineRunMode(mode)
        mode_options = build_run_mode_options(
            resolved_mode,
            show=show,
            save_txt=save_txt,
            collect_metrics=collect_metrics,
            persist_events=None if events_filename is None else True,
            persist_video=write_video,
        )
        if mode_options.persist_events and not events_filename:
            events_filename = f"{Path(str(source)).stem}_events.jsonl"

        def push_log(
            level: str, message: str, frame: Optional[int] = None, **payload: object
        ) -> None:
            entry: Dict[str, object] = {
                "timestamp": time.time(),
                "level": level,
                "message": message,
            }
            if frame is not None:
                entry["frame"] = int(frame)
            if payload:
                entry["details"] = payload
            log_entries.append(entry)
            if len(log_entries) > MAX_PIPELINE_LOGS:
                log_entries.pop(0)
            log_method = getattr(self.logger, level, self.logger.info)
            try:
                numeric_level = getattr(logging, level.upper(), logging.INFO)
                log_event(self.logger, numeric_level, message, frame=frame, **payload)
            except Exception:  # pragma: no cover - safety fallback
                log_method("%s", message)

        source_for_cv = int(source) if str(source).isdigit() else source
        source_label = f"camera_{source}" if str(source).isdigit() else Path(str(source)).stem
        cap = cv2.VideoCapture(source_for_cv)
        if not cap.isOpened():
            push_log("error", "Не удалось открыть источник видео", frame=None, source=str(source))
            raise RuntimeError(f"Не удалось открыть источник: {source}")
        push_log(
            "info",
            "Запущена обработка видео",
            frame=0,
            source=str(source),
            mode=resolved_mode.value,
        )
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_path = (
            output_dir / f"{Path(str(source)).stem}_car.mp4" if mode_options.persist_video else None
        )
        writer = None
        if out_path is not None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

        tracker = SimpleKalmanTracker()
        traj_analyzer = TrajectoryAnalyzer()
        risk_analyzer = RiskAnalyzer(
            traj_analyzer,
            ttc_threshold=2.0,
            pet_threshold=2.0,
            use_lstm=self.use_lstm,
            model_path=self.lstm_model_path,
            device=("cuda" if str(self.device).startswith("cuda") else "cpu"),
            fps=float(fps),
            scene_calibration=self.scene_calibration,
        )
        roi_polygons = load_roi_config(self.roi_config, width, height)
        queue_counter = QueueCounter(roi_polygons)
        phase_optimizer = self._build_phase_optimizer(roi_polygons)
        events_path = (
            output_dir / events_filename
            if mode_options.persist_events and events_filename
            else None
        )

        frame_idx = 0
        start_time = time.time()
        optimization_interval = max(1, int(fps))
        current_plan = None

        events_collected: Optional[List[Dict[str, object]]] = (
            [] if mode_options.collect_metrics else None
        )
        queue_history: Optional[List[Dict[str, object]]] = (
            [] if mode_options.collect_metrics else None
        )
        plan_history: Optional[List[Dict[str, object]]] = (
            [] if mode_options.collect_metrics else None
        )
        total_events = 0
        forecast_iterations = 0
        track_retention_frames = max(5, int(fps * 2))
        trajectory_history_frames = max(10, int(fps * 5))
        seen_track_ids: set[int] = set()
        active_track_counts: List[int] = []
        total_inference_time = 0.0
        inference_samples = 0
        lp_solve_durations: List[float] = []
        demand_histories: Dict[str, List[Dict[str, object]]] = {
            name: [] for name in roi_polygons.keys()
        }

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            inference_started_at = time.perf_counter()
            results, positions = self._infer_frame(frame, tracker)
            inference_duration = time.perf_counter() - inference_started_at
            total_inference_time += inference_duration
            inference_samples += 1
            metric_histogram(
                "traffic_inference_duration_seconds",
                "YOLO/tracker inference duration per frame.",
                inference_duration,
                buckets=(0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0),
                labels={"tracker": self.tracker_backend.value},
            )
            seen_track_ids.update(int(tid) for tid in positions.keys())
            active_track_counts.append(len(positions))
            for tid, (x, y) in positions.items():
                traj_analyzer.add_position(tid, frame_idx, x, y)
            traj_analyzer.prune(
                current_frame=frame_idx,
                max_age_frames=track_retention_frames,
                max_history=trajectory_history_frames,
                active_ids=positions.keys(),
            )

            queues = queue_counter.update(positions, frame_idx)
            if queue_history is not None:
                queue_history.append({"frame": frame_idx, "queues": dict(queues)})
                if len(queue_history) > MAX_METRICS_HISTORY:
                    queue_history[:] = queue_history[-MAX_METRICS_HISTORY:]

            events = risk_analyzer.analyze_and_get_events(
                distance_threshold=self.distance_threshold,
                distance_threshold_meters=self.distance_threshold_meters,
                risk_threshold=self.risk_threshold,
            )
            risk_by_approach = aggregate_risk_by_approach(
                events,
                queue_counter,
                roi_polygons.keys(),
            )
            if events_path is not None and events:
                with open(events_path, "a", encoding="utf-8") as f:
                    for ev in events:
                        f.write(json.dumps(ev, ensure_ascii=False) + "\n")
            if events:
                top_event = max(events, key=lambda item: item.get("risk_score", 0.0))
                push_log(
                    "warning",
                    f"Обнаружено {len(events)} near-miss",
                    frame=frame_idx,
                    max_risk=round(top_event.get("risk_score", 0.0), 3),
                    min_ttc_sec=top_event.get("ttc"),
                    min_pet_sec=top_event.get("pet"),
                    ids=f"{top_event.get('id1')} vs {top_event.get('id2')}",
                )
            if events_collected is not None and events:
                total_events += len(events)
                events_collected.extend(events)
                if len(events_collected) > MAX_METRICS_HISTORY:
                    events_collected[:] = events_collected[-MAX_METRICS_HISTORY:]

            if frame_idx % optimization_interval == 0:
                for approach in roi_polygons.keys():
                    history = demand_histories.setdefault(approach, [])
                    history.append(
                        self._build_forecast_record(
                            source_label=source_label,
                            approach=approach,
                            queue_len=queues.get(approach, 0.0),
                            risk_score=risk_by_approach.get(approach, 0.0),
                            frame_idx=frame_idx,
                            fps=float(fps),
                            started_at=start_time,
                        )
                    )
                    if len(history) > MAX_METRICS_HISTORY:
                        demand_histories[approach] = history[-MAX_METRICS_HISTORY:]
                forecast_by_approach = self._predict_demand(demand_histories)
                if forecast_by_approach is not None:
                    forecast_iterations += 1
                current_plan = phase_optimizer.optimize(
                    queues,
                    risk_by_approach,
                    forecast_queues=forecast_by_approach,
                )
                if current_plan.get("solve_duration_seconds") is not None:
                    lp_solve_durations.append(float(current_plan["solve_duration_seconds"]))
                plan_entry = {
                    "frame": frame_idx,
                    "plan": current_plan,
                    "risk": risk_by_approach,
                    "queues": dict(queues),
                    "forecast": dict(forecast_by_approach or {}),
                }
                if plan_history is not None:
                    plan_history.append(plan_entry)
                    if len(plan_history) > MAX_METRICS_HISTORY:
                        plan_history[:] = plan_history[-MAX_METRICS_HISTORY:]
                push_log(
                    "info",
                    "Пересчитан цикл светофора",
                    frame=frame_idx,
                    plan=current_plan,
                    queues=dict(queues),
                    risk=risk_by_approach,
                    forecast=forecast_by_approach or {},
                )

            annotated = self._draw_boxes(frame, results)
            if writer is not None:
                writer.write(annotated)
            if mode_options.save_txt:
                txt_path = output_dir / f"{Path(str(source)).stem}_car_{frame_idx:06d}.txt"
                self._save_txt(results, txt_path)
            if mode_options.show:
                cv2.imshow("Car Detection", annotated)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            frame_idx += 1

        cap.release()
        if writer is not None:
            writer.release()
        if mode_options.show:
            cv2.destroyAllWindows()

        processing_time = time.time() - start_time
        processing_fps = (frame_idx / processing_time) if processing_time > 0 else 0.0
        avg_inference_time = (
            total_inference_time / inference_samples if inference_samples > 0 else 0.0
        )
        avg_lp_solve_time = (
            sum(lp_solve_durations) / len(lp_solve_durations) if lp_solve_durations else 0.0
        )
        metric_histogram(
            "traffic_processing_fps",
            "Effective FPS for processed traffic videos.",
            processing_fps,
            buckets=(1, 5, 10, 15, 20, 30, 60),
            labels={"mode": resolved_mode.value},
        )
        metric_gauge(
            "traffic_pipeline_last_processing_fps",
            "Last observed traffic pipeline processing FPS.",
            processing_fps,
            labels={"mode": resolved_mode.value},
        )
        push_log(
            "info",
            "Обработка видео завершена",
            frame=frame_idx,
            frames_processed=frame_idx,
            duration_sec=round(processing_time, 2),
            total_events=len(events_collected or []),
            processing_fps=round(processing_fps, 3),
            avg_inference_time_sec=round(avg_inference_time, 4),
            avg_lp_solve_time_sec=round(avg_lp_solve_time, 4),
        )

        result: Dict[str, object] = {
            "output_video": str(out_path) if out_path else None,
            "frames_processed": frame_idx,
            "events_file": str(events_path) if events_path else None,
            "latest_plan": current_plan,
            "logs": log_entries,
            "mode": resolved_mode.value,
            "tracking_summary": {
                "unique_tracks": len(seen_track_ids),
                "avg_active_tracks": (
                    round(sum(active_track_counts) / len(active_track_counts), 3)
                    if active_track_counts
                    else 0.0
                ),
                "peak_active_tracks": max(active_track_counts, default=0),
            },
            "scene_calibration": (
                self.scene_calibration.as_metadata()
                if self.scene_calibration is not None
                else {"name": "uncalibrated", "is_calibrated": False}
            ),
            "demand_forecast": {
                "enabled": bool(self.demand_forecast_settings.enabled),
                "model_loaded": self.demand_forecaster is not None,
                "alpha": float(self.demand_forecast_settings.alpha),
                "window_size": int(self.demand_forecast_settings.window_size),
                "horizon": int(self.demand_forecast_settings.horizon),
            },
            "performance_metrics": {
                "processing_time_seconds": round(processing_time, 4),
                "processing_fps": round(processing_fps, 4),
                "avg_inference_time_seconds": round(avg_inference_time, 6),
                "total_inference_time_seconds": round(total_inference_time, 4),
                "avg_lp_solve_time_seconds": round(avg_lp_solve_time, 6),
                "lp_iterations": len(lp_solve_durations),
                "forecast_iterations": forecast_iterations,
            },
        }
        if mode_options.collect_metrics:
            result["events"] = events_collected or []
            result["total_events"] = total_events
            result["queue_history"] = queue_history or []
            result["plan_history"] = plan_history or []
        return result
