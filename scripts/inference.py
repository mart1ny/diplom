import argparse
import os
import sys
from pathlib import Path

import cv2

try:  # pragma: no cover
    from scripts.logging_utils import configure_logging
    from scripts.pipeline_runner import TrafficPipeline
    from scripts.run_modes import PipelineRunMode
    from scripts.settings import get_settings
    from scripts.tracker_backends import TrackerBackend
except ImportError:  # pragma: no cover
    from logging_utils import configure_logging
    from pipeline_runner import TrafficPipeline
    from run_modes import PipelineRunMode
    from settings import get_settings
    from tracker_backends import TrackerBackend


def parse_args():
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description="YOLOv8/YOLOv9 inference for car detection (images, video, RTSP)."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(settings.model_paths.yolo_model_path),
        help="Path to YOLO model weights (.pt) or model name from ultralytics/hub.",
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to image, video, folder, or RTSP stream.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs",
        help="Directory to save results (annotated frames).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=[mode.value for mode in PipelineRunMode],
        default=PipelineRunMode.RESEARCH.value,
        help="Run profile: demo (visual), research (metrics + exports), api (headless service profile).",
    )
    parser.add_argument(
        "--tracker-backend",
        type=str,
        choices=[backend.value for backend in TrackerBackend],
        default=settings.tracker.backend,
        help="Tracking backend for videos: bytetrack (Ultralytics ByteTrack) or simple (Kalman fallback).",
    )
    parser.add_argument(
        "--show",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override whether to show results in a real-time window.",
    )
    parser.add_argument(
        "--save-txt",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override whether to save detection results as .txt files (YOLO format).",
    )
    parser.add_argument(
        "--write-video",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override whether to save annotated output video.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu",
        help="Device to run inference on (cuda or cpu).",
    )
    parser.add_argument(
        "--risk-threshold",
        type=float,
        default=settings.thresholds.risk_threshold,
        help="Risk score threshold to emit near-miss events.",
    )
    parser.add_argument(
        "--distance-threshold",
        type=float,
        default=settings.thresholds.distance_threshold_px,
        help="Distance threshold (pixels) to consider conflict candidates.",
    )
    parser.add_argument(
        "--distance-threshold-meters",
        type=float,
        default=settings.thresholds.distance_threshold_meters,
        help="Optional metric threshold for conflict candidates after scene calibration.",
    )
    parser.add_argument(
        "--use-lstm",
        action="store_true",
        help="Enable LSTM risk model (requires PyTorch).",
    )
    parser.add_argument(
        "--lstm-model-path",
        type=str,
        default=(
            str(settings.model_paths.lstm_model_path)
            if settings.model_paths.lstm_model_path is not None
            else None
        ),
        help="Path to trained LSTM weights (.pt).",
    )
    parser.add_argument(
        "--save-events",
        type=str,
        default="events.jsonl",
        help="Filename to save near-miss events (JSONL) in the output directory.",
    )
    parser.add_argument(
        "--roi-config",
        type=str,
        default=(
            str(settings.model_paths.roi_config_path)
            if settings.model_paths.roi_config_path is not None
            else None
        ),
        help="Path to ROI config JSON (per-approach polygons). If not provided, uses defaults.",
    )
    parser.add_argument(
        "--scene-calibration",
        type=str,
        default=(
            str(settings.model_paths.scene_calibration_path)
            if settings.model_paths.scene_calibration_path is not None
            else None
        ),
        help="Path to scene calibration JSON with meters_per_pixel or homography.",
    )
    parser.add_argument(
        "--cycle-min",
        type=float,
        default=settings.optimizer.cycle_min,
        help="Minimum cycle length (seconds) for LP optimizer.",
    )
    parser.add_argument(
        "--cycle-max",
        type=float,
        default=settings.optimizer.cycle_max,
        help="Maximum cycle length (seconds) for LP optimizer.",
    )
    parser.add_argument(
        "--lambda-risk",
        type=float,
        default=settings.optimizer.lambda_risk,
        help="Weight for risk penalty in phase optimization.",
    )
    return parser.parse_args()


def main():
    configure_logging()
    args = parse_args()
    output_dir = Path(args.output)

    pipeline = TrafficPipeline(
        model_path=args.model,
        device=args.device,
        roi_config=args.roi_config,
        cycle_bounds=(args.cycle_min, args.cycle_max),
        lambda_risk=args.lambda_risk,
        risk_threshold=args.risk_threshold,
        distance_threshold=args.distance_threshold,
        distance_threshold_meters=args.distance_threshold_meters,
        use_lstm=args.use_lstm,
        lstm_model_path=args.lstm_model_path,
        tracker_backend=args.tracker_backend,
        scene_calibration_path=args.scene_calibration,
    )

    def run_video(src: str) -> None:
        pipeline.process_video(
            source=src,
            output_dir=output_dir,
            mode=args.mode,
            show=args.show,
            save_txt=args.save_txt,
            events_filename=args.save_events,
            write_video=args.write_video,
        )

    image_show = args.show if args.show is not None else args.mode == PipelineRunMode.DEMO.value
    image_save_txt = (
        args.save_txt if args.save_txt is not None else args.mode == PipelineRunMode.RESEARCH.value
    )

    if args.source.lower().startswith(("rtsp://", "http://", "https://")) or args.source.endswith(
        (".mp4", ".avi", ".mov")
    ):
        run_video(args.source)
    elif args.source.isdigit():
        run_video(args.source)
    elif os.path.isdir(args.source):
        for img_file in Path(args.source).glob("*.[jp][pn]g"):
            pipeline.process_image(
                str(img_file), output_dir, show=image_show, save_txt=image_save_txt
            )
    elif args.source.endswith((".jpg", ".jpeg", ".png")):
        pipeline.process_image(args.source, output_dir, show=image_show, save_txt=image_save_txt)
    else:
        print(f"Unsupported source: {args.source}")
        sys.exit(1)


if __name__ == "__main__":
    main()


# Для запуска
# python scripts/inference.py --model yolov8n.pt --source your_video.mp4 --output results --show --save-txt
# Для RTSP:
# python scripts/inference.py --model yolov8n.pt --source rtsp://... --output results

"""
Выход детекции на кадр:
 • detection: {trackless} список объектов вида:
 ▫ id: None (пока)
 ▫ cls: “car”
 ▫ bbox_xyxy: [x1, y1, x2, y2]
 ▫ conf: float
 ▫ center: (xc, yc)
 ▫ t: frame_idx
 """
