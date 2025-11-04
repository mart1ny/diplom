

import argparse
import sys
import os
from pathlib import Path

import cv2
from ultralytics import YOLO

# COCO class index for 'car'
CAR_CLASS_IDX = 2

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

def process_video(model, source, output_dir, show=False, save_txt_flag=False):
    cap = cv2.VideoCapture(source)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = output_dir / f"{Path(str(source)).stem}_car.mp4"
    out_vid = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        frame_annot = draw_boxes(frame, results[0])
        out_vid.write(frame_annot)
        if save_txt_flag:
            txt_path = output_dir / f"{Path(str(source)).stem}_car_{frame_idx:06d}.txt"
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
        process_video(model, args.source, output_dir, show=args.show, save_txt_flag=args.save_txt)
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