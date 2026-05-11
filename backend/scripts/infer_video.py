"""Batch video inference: run YOLO on every frame and write an annotated MP4.

Usable two ways:
  * Imported by `app.py` via `infer_video(...)` for the `/api/process-video`
    endpoint.
  * Run directly from the CLI for ad-hoc processing (see `main`).
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import cv2
import torch
from ultralytics import YOLO

from scripts.detection_metrics import (
    add_frame_to_track_durations,
    build_frame_detection_record,
    round_track_durations,
)


logger = logging.getLogger(__name__)

DEFAULT_FPS = 30.0
LOG_PROGRESS_EVERY_N_FRAMES = 30


def infer_video(
    weights_path: Path,
    video_path: Path,
    output_path: Path,
    conf: float = 0.25,
    imgsz: int = 640,
    track: bool = False,
) -> dict[str, Any]:
    """Process `video_path` end-to-end and write an annotated MP4 to `output_path`.

    Returns a dict with the total frame count, source FPS, and per-frame
    detection / tracking metrics suitable for direct JSON serialization.
    """
    _validate_input_paths(weights_path, video_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading YOLO model from %s", weights_path)
    logger.info("Device: %s", "GPU" if torch.cuda.is_available() else "CPU")
    model = YOLO(str(weights_path))

    capture = _open_video_capture(video_path)
    frames_per_second, width, height, total_frames = _read_video_properties(capture)
    writer = _open_video_writer(output_path, frames_per_second, width, height)

    detection_count_over_time: list[dict[str, Any]] = []
    duration_by_track_id: dict[str, float] = {}
    frames_processed = 0

    try:
        while True:
            frame_was_read, raw_frame = capture.read()
            if not frame_was_read:
                break

            yolo_result = _run_inference(model, raw_frame, conf, imgsz, track)
            detection_count_over_time.append(
                build_frame_detection_record(
                    frames_processed, frames_per_second, int(len(yolo_result.boxes))
                )
            )
            if track:
                add_frame_to_track_durations(
                    yolo_result, frames_per_second, duration_by_track_id
                )

            writer.write(yolo_result.plot(labels=False, conf=False))
            frames_processed += 1

            if frames_processed % LOG_PROGRESS_EVERY_N_FRAMES == 0:
                logger.info("Processed %d / %d frames", frames_processed, total_frames)
    finally:
        capture.release()
        writer.release()

    logger.info("Wrote annotated video to %s", output_path)
    return {
        "frame_count": frames_processed,
        "fps": frames_per_second,
        "metrics": {
            "detection_count_over_time": detection_count_over_time,
            "time_on_screen_by_track_id": round_track_durations(duration_by_track_id),
        },
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_input_paths(weights_path: Path, video_path: Path) -> None:
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")


def _open_video_capture(video_path: Path) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    return capture


def _read_video_properties(
    capture: cv2.VideoCapture,
) -> tuple[float, int, int, int]:
    """Return (fps, width, height, total_frames) for the open capture."""
    frames_per_second = capture.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(
        "Source video: %dx%d @ %.1f fps, ~%d frames",
        width,
        height,
        frames_per_second,
        total_frames,
    )
    return frames_per_second, width, height, total_frames


def _open_video_writer(
    output_path: Path, fps: float, width: int, height: int
) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        logger.warning("VideoWriter failed to open %s — codec may be missing", output_path)
    return writer


def _run_inference(
    model: YOLO,
    raw_frame: Any,
    confidence_threshold: float,
    image_size: int,
    tracking_enabled: bool,
) -> Any:
    """Run a single inference (or tracking) call and return the first result."""
    if tracking_enabled:
        results = model.track(
            raw_frame,
            conf=confidence_threshold,
            imgsz=image_size,
            persist=True,
            verbose=False,
        )
    else:
        results = model(
            raw_frame,
            conf=confidence_threshold,
            imgsz=image_size,
            verbose=False,
        )
    return results[0]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Run inference from the command line: see `--help` for arguments."""
    parser = argparse.ArgumentParser(description="Run YOLO over a video file.")
    parser.add_argument("--weights", required=True, help="Path to YOLO weights (.pt)")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument(
        "--output",
        default="data/output/annotated.mp4",
        help="Path to write the annotated output video",
    )
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--track", action="store_true", help="Enable persistent tracking IDs")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    try:
        run_result = infer_video(
            weights_path=Path(args.weights),
            video_path=Path(args.video),
            output_path=Path(args.output),
            conf=args.conf,
            imgsz=args.imgsz,
            track=args.track,
        )
    except (FileNotFoundError, RuntimeError) as exc:
        raise SystemExit(str(exc))

    print(f"Processed {run_result['frame_count']} frames")
    print(f"Saved annotated video: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
