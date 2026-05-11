import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


def infer_video(
    weights_path: Path,
    video_path: Path,
    output_path: Path,
    conf: float = 0.25,
    imgsz: int = 640,
    track: bool = False,
) -> dict:
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights_path))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    frame_count = 0
    detection_count_over_time = []
    time_on_screen_by_track_id: dict[str, float] = {}

    while True:
        ok, frame = cap.read()
        if not ok:
            break

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

    cap.release()
    writer.release()

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
