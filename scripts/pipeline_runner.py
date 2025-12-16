from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
from ultralytics import YOLO

try:  # pragma: no cover
    from scripts.queue_counter import QueueCounter, load_roi_config
    from scripts.rajectory_analysis import RiskAnalyzer, TrajectoryAnalyzer
    from scripts.tracking import SimpleKalmanTracker
    from scripts.traffic_optimizer import DEFAULT_PHASE_CONFIG, PhaseOptimizer
except ImportError:  # pragma: no cover
    from queue_counter import QueueCounter, load_roi_config
    from rajectory_analysis import RiskAnalyzer, TrajectoryAnalyzer
    from tracking import SimpleKalmanTracker
    from traffic_optimizer import DEFAULT_PHASE_CONFIG, PhaseOptimizer

# COCO class index for 'car'
CAR_CLASS_IDX = 2


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
        model_path: str,
        device: str = "cpu",
        roi_config: Optional[str] = None,
        cycle_bounds: Tuple[float, float] = (50.0, 90.0),
        lambda_risk: float = 5.0,
        risk_threshold: float = 0.6,
        distance_threshold: float = 60.0,
        use_lstm: bool = False,
        lstm_model_path: Optional[str] = None,
    ) -> None:
        self.model = YOLO(model_path)
        self.device = device
        self.roi_config = roi_config
        self.cycle_bounds = cycle_bounds
        self.lambda_risk = lambda_risk
        self.risk_threshold = risk_threshold
        self.distance_threshold = distance_threshold
        self.use_lstm = use_lstm
        self.lstm_model_path = lstm_model_path

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
        phase_config = {
            name: DEFAULT_PHASE_CONFIG.get(
                name,
                {"min_green": 0.05, "max_green": 0.5, "saturation_flow": 0.25},
            )
            for name in roi_polygons.keys()
        }
        return PhaseOptimizer(
            phase_config=phase_config,
            cycle_bounds=self.cycle_bounds,
            lambda_risk=self.lambda_risk,
        )

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
        show: bool = False,
        save_txt: bool = False,
        events_filename: Optional[str] = None,
        collect_metrics: bool = False,
    ) -> Dict[str, object]:
        output_dir = _ensure_dir(output_dir)
        source_for_cv = int(source) if str(source).isdigit() else source
        cap = cv2.VideoCapture(source_for_cv)
        if not cap.isOpened():
            raise RuntimeError(f"Не удалось открыть источник: {source}")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_path = output_dir / f"{Path(str(source)).stem}_car.mp4"
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
        )
        roi_polygons = load_roi_config(self.roi_config, width, height)
        queue_counter = QueueCounter(roi_polygons)
        phase_optimizer = self._build_phase_optimizer(roi_polygons)
        events_path = output_dir / events_filename if events_filename else None

        frame_idx = 0
        optimization_interval = max(1, int(fps))
        current_plan = None

        events_collected: Optional[List[Dict[str, object]]] = [] if collect_metrics else None
        queue_history: Optional[List[Dict[str, object]]] = [] if collect_metrics else None
        plan_history: Optional[List[Dict[str, object]]] = [] if collect_metrics else None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = self.model(frame)
            detections = []
            for box in results[0].boxes:
                if int(box.cls) != CAR_CLASS_IDX:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x_center = (x1 + x2) / 2.0
                y_center = (y1 + y2) / 2.0
                detections.append((x_center, y_center))

            positions = tracker.step(detections)
            for tid, (x, y) in positions.items():
                traj_analyzer.add_position(tid, frame_idx, x, y)

            queues = queue_counter.update(positions, frame_idx)
            if queue_history is not None:
                queue_history.append({"frame": frame_idx, "queues": dict(queues)})

            events = risk_analyzer.analyze_and_get_events(
                distance_threshold=self.distance_threshold,
                risk_threshold=self.risk_threshold,
            )
            risk_by_approach = {name: 0.0 for name in roi_polygons.keys()}
            if events_path is not None and events:
                with open(events_path, "a", encoding="utf-8") as f:
                    for ev in events:
                        f.write(json.dumps(ev, ensure_ascii=False) + "\n")
            if events_collected is not None and events:
                events_collected.extend(events)
                for ev in events:
                    approach1 = queue_counter.get_track_approach(ev["id1"])
                    approach2 = queue_counter.get_track_approach(ev["id2"])
                    if approach1:
                        risk_by_approach[approach1] += ev["risk_score"] * 0.5
                    if approach2:
                        risk_by_approach[approach2] += ev["risk_score"] * 0.5

            if frame_idx % optimization_interval == 0:
                current_plan = phase_optimizer.optimize(queues, risk_by_approach)
                plan_entry = {
                    "frame": frame_idx,
                    "plan": current_plan,
                    "risk": risk_by_approach,
                }
                if plan_history is not None:
                    plan_history.append(plan_entry)
                print(
                    f"[frame {frame_idx}] queues: {queues} risk: {risk_by_approach} plan: {current_plan}"
                )

            annotated = self._draw_boxes(frame, results[0])
            writer.write(annotated)
            if save_txt:
                txt_path = output_dir / f"{Path(str(source)).stem}_car_{frame_idx:06d}.txt"
                self._save_txt(results[0], txt_path)
            if show:
                cv2.imshow("Car Detection", annotated)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            frame_idx += 1

        cap.release()
        writer.release()
        if show:
            cv2.destroyAllWindows()

        result: Dict[str, object] = {
            "output_video": str(out_path),
            "frames_processed": frame_idx,
            "events_file": str(events_path) if events_path else None,
            "latest_plan": current_plan,
        }
        if collect_metrics:
            result["events"] = events_collected or []
            result["queue_history"] = queue_history or []
            result["plan_history"] = plan_history or []
        return result
