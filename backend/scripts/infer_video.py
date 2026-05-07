import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", default="data/output/annotated.mp4")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--track", action="store_true")
    args = parser.parse_args()

    weights_path = Path(args.weights)
    video_path = Path(args.video)
    output_path = Path(args.output)

    if not weights_path.exists():
        raise SystemExit(f"Weights not found: {weights_path}")
    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights_path))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    frame_count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if args.track:
            results = model.track(frame, conf=args.conf, imgsz=args.imgsz, persist=True, verbose=False)
        else:
            results = model(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)

        annotated = results[0].plot()
        writer.write(annotated)
        frame_count += 1

    cap.release()
    writer.release()

    print(f"Processed {frame_count} frames")
    print(f"Saved annotated video: {output_path.resolve()}")


if __name__ == "__main__":
    main()
