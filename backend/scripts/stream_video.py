"""Stream video frames with YOLO detection as MJPEG."""
import logging
import sys
from collections import defaultdict
from pathlib import Path
from threading import Lock
from typing import Generator

import cv2
import torch
from ultralytics import YOLO

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


class VideoStreamer:
    """Handles MJPEG streaming with real-time metrics collection."""

    def __init__(
        self,
        weights_path: Path,
        video_path: Path,
        conf: float = 0.25,
        imgsz: int = 640,
        track: bool = False,
    ):
        self.weights_path = Path(weights_path)
        self.video_path = Path(video_path)
        self.conf = conf
        self.imgsz = imgsz
        self.track = track

        # Verify files exist
        if not self.weights_path.exists():
            raise FileNotFoundError(f"Weights not found: {self.weights_path}")
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {self.video_path}")

        # Metrics state
        self.metrics_lock = Lock()
        self.detection_count_over_time = []
        self.time_on_screen_by_track_id = {}
        self.frame_count = 0
        self.fps = 30.0
        self.is_streaming = False

        # Load model once
        logger.info(f"Loading YOLO model from {self.weights_path}")
        self.model = YOLO(str(self.weights_path))
        logger.info(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")

    def stream_frames(self) -> Generator[bytes, None, None]:
        """Yield MJPEG frame boundaries with JPEG-encoded annotated frames."""
        logger.info(f"Starting stream from {self.video_path}")

        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {self.video_path}")

        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(
            f"Video: {width}x{height} @ {self.fps} fps, ~{total_frames} frames"
        )

        self.is_streaming = True
        self.frame_count = 0

        try:
            while self.is_streaming:
                ok, frame = cap.read()
                if not ok:
                    logger.info("End of video reached")
                    break

                if self.frame_count % 30 == 0:
                    logger.info(f"Stream frame {self.frame_count}/{total_frames}")

                # Run inference
                if self.track:
                    results = self.model.track(
                        frame,
                        conf=self.conf,
                        imgsz=self.imgsz,
                        persist=True,
                        verbose=False,
                    )
                else:
                    results = self.model(
                        frame, conf=self.conf, imgsz=self.imgsz, verbose=False
                    )

                result = results[0]
                annotated = result.plot(labels=False, conf=False)

                # Collect metrics
                box_count = int(len(result.boxes))
                with self.metrics_lock:
                    self.detection_count_over_time.append(
                        {
                            "frame": self.frame_count,
                            "timestamp_sec": self.frame_count / self.fps,
                            "count": box_count,
                        }
                    )

                    if self.track and result.boxes.id is not None:
                        ids = {
                            str(int(i))
                            for i in result.boxes.id.cpu().numpy().tolist()
                        }
                        for track_id in ids:
                            self.time_on_screen_by_track_id[track_id] = (
                                self.time_on_screen_by_track_id.get(track_id, 0.0)
                                + (1.0 / self.fps)
                            )

                # Encode frame as JPEG
                _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
                frame_data = buffer.tobytes()

                # Yield MJPEG boundary
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(frame_data)).encode() + b"\r\n\r\n"
                    + frame_data
                    + b"\r\n"
                )

                self.frame_count += 1

        finally:
            cap.release()
            self.is_streaming = False
            logger.info("Stream ended")

    def get_metrics(self) -> dict:
        """Get current metrics snapshot."""
        with self.metrics_lock:
            return {
                "frame_count": self.frame_count,
                "fps": self.fps,
                "is_streaming": self.is_streaming,
                "detection_count_over_time": list(self.detection_count_over_time),
                "time_on_screen_by_track_id": {
                    k: round(v, 3) for k, v in self.time_on_screen_by_track_id.items()
                },
            }

    def stop(self) -> None:
        """Stop streaming."""
        logger.info("Stopping stream")
        self.is_streaming = False
