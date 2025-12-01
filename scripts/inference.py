

import argparse
import sys
import os
from pathlib import Path

import cv2
from ultralytics import YOLO

# COCO class index for 'car'
CAR_CLASS_IDX = 2

import json
from tracking import SimpleKalmanTracker
from rajectory_analysis import TrajectoryAnalyzer, RiskAnalyzer
from queue_counter import QueueCounter, load_roi_config
from traffic_optimizer import PhaseOptimizer, DEFAULT_PHASE_CONFIG

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

def save_txt(results, txt_path):
    """
    Save detection results in YOLO format: <class> <x_center> <y_center> <width> <height> <confidence>
    """
    with open(txt_path, "w") as f:
        for box in results.boxes:
            if int(box.cls) == CAR_CLASS_IDX:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                x_center = (x1 + x2) / 2
                y_center = (y1 + y2) / 2
                width = x2 - x1
                height = y2 - y1
                f.write(f"{CAR_CLASS_IDX} {x_center:.2f} {y_center:.2f} {width:.2f} {height:.2f} {conf:.4f}\n")

def draw_boxes(frame, results):
    """
    Draw bounding boxes for cars on the frame.
    """
    for box in results.boxes:
        if int(box.cls) == CAR_CLASS_IDX:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            label = f"car {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
            cv2.putText(
                frame, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2
            )
    return frame

def process_image(model, img_path, output_dir, show=False, save_txt_flag=False):
    img = cv2.imread(str(img_path))
    results = model(img)
    frame = img.copy()
    frame = draw_boxes(frame, results[0])

    out_path = output_dir / f"{Path(img_path).stem}_car.jpg"
    cv2.imwrite(str(out_path), frame)
    if save_txt_flag:
        txt_path = output_dir / f"{Path(img_path).stem}_car.txt"
        save_txt(results[0], txt_path)
    if show:
        cv2.imshow("Car Detection", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def process_video(model, args, output_dir, show=False, save_txt_flag=False):
    source = int(args.source) if str(args.source).isdigit() else args.source
    cap = cv2.VideoCapture(source)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = output_dir / f"{Path(str(args.source)).stem}_car.mp4"
    out_vid = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    # Init tracking and risk analysis
    tracker = SimpleKalmanTracker()
    traj_analyzer = TrajectoryAnalyzer()
    risk_analyzer = RiskAnalyzer(
        traj_analyzer,
        ttc_threshold=2.0,
        pet_threshold=2.0,
        use_lstm=args.use_lstm,
        model_path=args.lstm_model_path,
        device=("cuda" if str(args.device).startswith("cuda") else "cpu"),
    )
    roi_polygons = load_roi_config(args.roi_config, width, height)
    queue_counter = QueueCounter(roi_polygons)
    phase_config = {
        name: DEFAULT_PHASE_CONFIG.get(
            name,
            {"min_green": 0.05, "max_green": 0.5, "saturation_flow": 0.25},
        )
        for name in roi_polygons.keys()
    }
    phase_optimizer = PhaseOptimizer(
        phase_config=phase_config,
        cycle_bounds=(args.cycle_min, args.cycle_max),
        lambda_risk=args.lambda_risk,
    )
    events_file = output_dir / args.save_events if args.save_events else None

    frame_idx = 0
    optimization_interval = max(1, int(fps))
    current_plan = None
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        # Build detections (centers) for cars
        detections = []
        for box in results[0].boxes:
            if int(box.cls) == CAR_CLASS_IDX:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x_center = (x1 + x2) / 2.0
                y_center = (y1 + y2) / 2.0
                detections.append((x_center, y_center))

        # Update tracker and accumulate trajectories
        positions = tracker.step(detections)
        for tid, (x, y) in positions.items():
            traj_analyzer.add_position(tid, frame_idx, x, y)

        queues = queue_counter.update(positions, frame_idx)

        # Analyze risk and emit near-miss events
        events = risk_analyzer.analyze_and_get_events(
            distance_threshold=args.distance_threshold,
            risk_threshold=args.risk_threshold,
        )
        risk_by_approach = {name: 0.0 for name in roi_polygons.keys()}
        if events and events_file is not None:
            with open(events_file, "a", encoding="utf-8") as f:
                for ev in events:
                    f.write(json.dumps(ev, ensure_ascii=False) + "\n")
        if events:
            for ev in events:
                approach1 = queue_counter.get_track_approach(ev["id1"])
                approach2 = queue_counter.get_track_approach(ev["id2"])
                if approach1:
                    risk_by_approach[approach1] += ev["risk_score"] * 0.5
                if approach2:
                    risk_by_approach[approach2] += ev["risk_score"] * 0.5

        if frame_idx % optimization_interval == 0:
            current_plan = phase_optimizer.optimize(queues, risk_by_approach)
            print(
                f"[frame {frame_idx}] queues: {queues} risk: {risk_by_approach} plan: {current_plan}"
            )

        frame_annot = draw_boxes(frame, results[0])
        out_vid.write(frame_annot)
        if save_txt_flag:
            txt_path = output_dir / f"{Path(str(args.source)).stem}_car_{frame_idx:06d}.txt"
            save_txt(results[0], txt_path)
        if show:
            cv2.imshow("Car Detection", frame_annot)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        frame_idx += 1
    cap.release()
    out_vid.release()
    if show:
        cv2.destroyAllWindows()

def main():
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)

    # Determine source type
    if args.source.lower().startswith(("rtsp://", "http://", "https://")) or args.source.endswith((".mp4", ".avi", ".mov")):
        process_video(model, args, output_dir, show=args.show, save_txt_flag=args.save_txt)
    elif args.source.isdigit():
        process_video(model, args, output_dir, show=args.show, save_txt_flag=args.save_txt)
    elif os.path.isdir(args.source):
        for img_file in Path(args.source).glob("*.[jp][pn]g"):
            process_image(model, img_file, output_dir, show=args.show, save_txt_flag=args.save_txt)
    elif args.source.endswith((".jpg", ".jpeg", ".png")):
        process_image(model, args.source, output_dir, show=args.show, save_txt_flag=args.save_txt)
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