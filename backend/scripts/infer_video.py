import argparse
import logging
import sys
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def infer_video(
    weights_path: Path,
    video_path: Path,
    output_path: Path,
    conf: float = 0.25,
    imgsz: int = 640,
    track: bool = False,
) -> dict:
    logger.info(f"Starting inference: weights={weights_path}, video={video_path}, output={output_path}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    logger.info(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading YOLO model...")
    model = YOLO(str(weights_path))
    logger.info(f"Model loaded. Task: {model.task}")

    logger.info(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    logger.info(f"Video info: {width}x{height}, {fps} fps, ~{total_frames} frames")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        logger.warning("VideoWriter failed to open, may have codec issues")

    frame_count = 0
    detection_count_over_time = []
    time_on_screen_by_track_id: dict[str, float] = {}

    logger.info("Starting frame processing loop...")
    while True:
        ok, frame = cap.read()
        if not ok:
            logger.info("End of video reached")
            break

        if frame_count % 30 == 0:
            logger.info(f"Processing frame {frame_count}/{total_frames}")

        if track:
            results = model.track(frame, conf=conf, imgsz=imgsz, persist=True, verbose=False)
        else:
            results = model(frame, conf=conf, imgsz=imgsz, verbose=False)

        result = results[0]
        box_count = int(len(result.boxes))
        detection_count_over_time.append(
            {
                "frame": frame_count,
                "timestamp_sec": frame_count / fps,
                "count": box_count,
            }
        )

        if track and result.boxes.id is not None:
            ids = {str(int(i)) for i in result.boxes.id.cpu().numpy().tolist()}
            for track_id in ids:
                time_on_screen_by_track_id[track_id] = time_on_screen_by_track_id.get(track_id, 0.0) + (1.0 / fps)

        annotated = result.plot()
        writer.write(annotated)
        frame_count += 1

    logger.info(f"Frame processing complete. Total frames: {frame_count}")
    cap.release()
    writer.release()
    logger.info(f"Output video saved to: {output_path}")

    return {
        "frame_count": frame_count,
        "fps": fps,
        "metrics": {
            "detection_count_over_time": detection_count_over_time,
            "time_on_screen_by_track_id": {k: round(v, 3) for k, v in time_on_screen_by_track_id.items()},
        },
    }


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

    try:
        result = infer_video(
            weights_path=weights_path,
            video_path=video_path,
            output_path=output_path,
            conf=args.conf,
            imgsz=args.imgsz,
            track=args.track,
        )
    except (FileNotFoundError, RuntimeError) as exc:
        raise SystemExit(str(exc))

    print(f"Processed {result['frame_count']} frames")
    print(f"Saved annotated video: {output_path.resolve()}")


if __name__ == "__main__":
    main()
