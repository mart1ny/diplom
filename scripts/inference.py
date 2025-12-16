

import argparse
import os
import sys
from pathlib import Path

import cv2

try:  # pragma: no cover
    from scripts.pipeline_runner import TrafficPipeline
except ImportError:  # pragma: no cover
    from pipeline_runner import TrafficPipeline

def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLOv8/YOLOv9 inference for car detection (images, video, RTSP)."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
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
        "--show",
        action="store_true",
        help="Show results in real-time window.",
    )
    parser.add_argument(
        "--save-txt",
        action="store_true",
        help="Save detection results as .txt files (YOLO format).",
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
        default=0.6,
        help="Risk score threshold to emit near-miss events.",
    )
    parser.add_argument(
        "--distance-threshold",
        type=float,
        default=60.0,
        help="Distance threshold (pixels) to consider conflict candidates.",
    )
    parser.add_argument(
        "--use-lstm",
        action="store_true",
        help="Enable LSTM risk model (requires PyTorch).",
    )
    parser.add_argument(
        "--lstm-model-path",
        type=str,
        default=None,
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
        default=None,
        help="Path to ROI config JSON (per-approach polygons). If not provided, uses defaults.",
    )
    parser.add_argument(
        "--cycle-min",
        type=float,
        default=50.0,
        help="Minimum cycle length (seconds) for LP optimizer.",
    )
    parser.add_argument(
        "--cycle-max",
        type=float,
        default=90.0,
        help="Maximum cycle length (seconds) for LP optimizer.",
    )
    parser.add_argument(
        "--lambda-risk",
        type=float,
        default=5.0,
        help="Weight for risk penalty in phase optimization.",
    )
    return parser.parse_args()

def main():
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
        use_lstm=args.use_lstm,
        lstm_model_path=args.lstm_model_path,
    )

    def run_video(src: str) -> None:
        pipeline.process_video(
            source=src,
            output_dir=output_dir,
            show=args.show,
            save_txt=args.save_txt,
            events_filename=args.save_events,
        )

    if args.source.lower().startswith(("rtsp://", "http://", "https://")) or args.source.endswith((".mp4", ".avi", ".mov")):
        run_video(args.source)
    elif args.source.isdigit():
        run_video(args.source)
    elif os.path.isdir(args.source):
        for img_file in Path(args.source).glob("*.[jp][pn]g"):
            pipeline.process_image(str(img_file), output_dir, show=args.show, save_txt=args.save_txt)
    elif args.source.endswith((".jpg", ".jpeg", ".png")):
        pipeline.process_image(args.source, output_dir, show=args.show, save_txt=args.save_txt)
    else:
        print(f"Unsupported source: {args.source}")
        sys.exit(1)

if __name__ == "__main__":
    main()


#Для запуска
# python scripts/inference.py --model yolov8n.pt --source your_video.mp4 --output results --show --save-txt
#Для RTSP:
#python scripts/inference.py --model yolov8n.pt --source rtsp://... --output results

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
