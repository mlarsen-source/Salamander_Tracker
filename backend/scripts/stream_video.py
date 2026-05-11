"""Live MJPEG streamer that runs YOLO on each frame and tracks metrics.

The Flask app constructs one `VideoStreamer` per upload, calls
`stream_frames()` to push annotated JPEGs to the browser, and polls
`get_metrics()` on a separate thread for the dashboard.
"""
from __future__ import annotations

import logging
from pathlib import Path
from threading import Lock
from typing import Any, Generator

import cv2
import torch
from ultralytics import YOLO

from scripts.detection_metrics import (
    add_frame_to_track_durations,
    build_frame_detection_record,
    round_track_durations,
)


logger = logging.getLogger(__name__)

# MJPEG part boundary used to delimit each frame in the HTTP response.
MJPEG_BOUNDARY = b"--frame"
JPEG_QUALITY = 85
DEFAULT_FPS = 30.0
LOG_PROGRESS_EVERY_N_FRAMES = 30


class VideoStreamer:
    """Runs YOLO over a video file and yields annotated JPEG frames as MJPEG."""

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
        self.confidence_threshold = conf
        self.image_size = imgsz
        self.tracking_enabled = track

        if not self.weights_path.exists():
            raise FileNotFoundError(f"Weights not found: {self.weights_path}")
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {self.video_path}")

        # Metrics state — guarded by a lock because the Flask metrics endpoint
        # reads from a different thread than the generator that writes to it.
        self._metrics_lock = Lock()
        self._detection_count_over_time: list[dict[str, Any]] = []
        self._duration_by_track_id: dict[str, float] = {}
        self._frames_processed = 0
        self._frames_per_second = DEFAULT_FPS
        self._is_streaming = False

        logger.info("Loading YOLO model from %s", self.weights_path)
        logger.info("Device: %s", "GPU" if torch.cuda.is_available() else "CPU")
        self._model = YOLO(str(self.weights_path))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def stream_frames(self) -> Generator[bytes, None, None]:
        """Yield successive MJPEG chunks until the video ends or `stop()` is called."""
        capture = self._open_video_capture()
        try:
            yield from self._stream_loop(capture)
        finally:
            capture.release()
            self._is_streaming = False
            logger.info("Stream ended after %d frames", self._frames_processed)

    def get_metrics(self) -> dict[str, Any]:
        """Return a thread-safe snapshot of the current metrics."""
        with self._metrics_lock:
            return {
                "frame_count": self._frames_processed,
                "fps": self._frames_per_second,
                "is_streaming": self._is_streaming,
                "detection_count_over_time": list(self._detection_count_over_time),
                "time_on_screen_by_track_id": round_track_durations(
                    self._duration_by_track_id
                ),
            }

    def stop(self) -> None:
        """Signal the streaming loop to exit on its next iteration."""
        logger.info("Stop requested")
        self._is_streaming = False

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _open_video_capture(self) -> cv2.VideoCapture:
        """Open the source video and cache its frame rate."""
        capture = cv2.VideoCapture(str(self.video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {self.video_path}")

        self._frames_per_second = capture.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(
            "Streaming %s (%dx%d @ %.1f fps, ~%d frames)",
            self.video_path.name,
            width,
            height,
            self._frames_per_second,
            total_frames,
        )
        return capture

    def _stream_loop(
        self, capture: cv2.VideoCapture
    ) -> Generator[bytes, None, None]:
        """Read → infer → record metrics → encode → yield, one frame at a time."""
        self._is_streaming = True
        self._frames_processed = 0

        while self._is_streaming:
            frame_was_read, raw_frame = capture.read()
            if not frame_was_read:
                logger.info("End of video reached")
                break

            yolo_result = self._run_inference(raw_frame)
            self._record_frame_metrics(yolo_result)

            annotated_frame = yolo_result.plot(labels=False, conf=False)
            yield self._encode_as_mjpeg_part(annotated_frame)

            self._frames_processed += 1
            if self._frames_processed % LOG_PROGRESS_EVERY_N_FRAMES == 0:
                logger.info("Streamed %d frames", self._frames_processed)

    def _run_inference(self, raw_frame: Any) -> Any:
        """Run YOLO on a single frame, optionally with persistent tracking."""
        if self.tracking_enabled:
            results = self._model.track(
                raw_frame,
                conf=self.confidence_threshold,
                imgsz=self.image_size,
                persist=True,
                verbose=False,
            )
        else:
            results = self._model(
                raw_frame,
                conf=self.confidence_threshold,
                imgsz=self.image_size,
                verbose=False,
            )
        return results[0]

    def _record_frame_metrics(self, yolo_result: Any) -> None:
        """Append this frame's detection count and update tracking durations."""
        detection_count = int(len(yolo_result.boxes))
        with self._metrics_lock:
            self._detection_count_over_time.append(
                build_frame_detection_record(
                    self._frames_processed,
                    self._frames_per_second,
                    detection_count,
                )
            )
            if self.tracking_enabled:
                add_frame_to_track_durations(
                    yolo_result,
                    self._frames_per_second,
                    self._duration_by_track_id,
                )

    @staticmethod
    def _encode_as_mjpeg_part(annotated_frame: Any) -> bytes:
        """Encode an annotated frame as a single multipart/x-mixed-replace chunk."""
        _, jpeg_buffer = cv2.imencode(
            ".jpg", annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
        )
        jpeg_bytes = jpeg_buffer.tobytes()
        return (
            MJPEG_BOUNDARY + b"\r\n"
            b"Content-Type: image/jpeg\r\n"
            b"Content-Length: " + str(len(jpeg_bytes)).encode() + b"\r\n\r\n"
            + jpeg_bytes
            + b"\r\n"
        )
