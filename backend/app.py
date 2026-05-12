"""Flask entrypoint for the Salamander Tracker web app.

Exposes two ways of running the YOLO model against an uploaded video:

  * Batch  (`POST /api/process-video`)  — process the whole file, then return
    a URL to a downloadable annotated MP4.
  * Live   (`POST /api/stream-video`)   — start a stream, then the browser
    pulls frames from `GET /api/video-stream` (MJPEG) and metrics from
    `GET /api/stream-metrics`. `POST /api/stream-stop` ends it early.

Only one live stream is active at a time; it is held in `active_streamer`.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional
from uuid import uuid4

from flask import Flask, Response, jsonify, request, send_from_directory
from flask_cors import CORS
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

from scripts.infer_video import infer_video
from scripts.stream_video import VideoStreamer


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

BACKEND_DIR = Path(__file__).resolve().parent
UPLOADS_DIR = BACKEND_DIR / "uploads"
PROCESSED_DIR = BACKEND_DIR / "processed"
DEFAULT_WEIGHTS_PATH = BACKEND_DIR / "models" / "best.pt"

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
CORS(app)

# The single in-flight live stream (or None when no stream is running).
active_streamer: Optional[VideoStreamer] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_uploaded_video() -> FileStorage:
    """Return the uploaded video file or raise ValueError with a user message."""
    if "video" not in request.files:
        raise ValueError("Missing 'video' file field")

    uploaded = request.files["video"]
    if not uploaded or not uploaded.filename:
        raise ValueError("No file selected")

    return uploaded


def save_upload_to_disk(uploaded: FileStorage, run_id: str) -> Path:
    """Persist the uploaded video under uploads/ and return its path."""
    safe_filename = secure_filename(uploaded.filename or "video.mp4")
    destination = UPLOADS_DIR / f"{run_id}_{safe_filename}"
    uploaded.save(destination)
    logger.info("Saved upload (%d bytes) to %s", destination.stat().st_size, destination)
    return destination


def resolve_weights_path() -> Path:
    """Pick model weights from form data, env var, or fall back to default."""
    weights_from_form = request.form.get("weights_path")
    if weights_from_form:
        return Path(weights_from_form)

    weights_from_env = os.getenv("MODEL_WEIGHTS")
    if weights_from_env:
        return Path(weights_from_env)

    return DEFAULT_WEIGHTS_PATH


def get_inference_options() -> tuple[float, int]:
    """Pull `conf` and `imgsz` from the form with sensible defaults."""
    confidence_threshold = float(request.form.get("conf", 0.25))
    image_size = int(request.form.get("imgsz", 640))
    return confidence_threshold, image_size


def absolute_url_for(path: str) -> str:
    """Build an absolute URL the browser can hit (e.g. for the MJPEG stream)."""
    return request.host_url.rstrip("/") + path


def error_response(message: str, status_code: int) -> tuple:
    return jsonify({"error": message}), status_code


def replace_active_streamer(next_streamer: VideoStreamer) -> None:
    """Stop the previous live stream and make `next_streamer` active."""
    global active_streamer

    if active_streamer is not None:
        active_streamer.stop()
        logger.info("Stopped previous live stream before starting replacement")

    active_streamer = next_streamer


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/api/health")
def health_check() -> tuple:
    """Liveness probe used by clients to confirm the backend is reachable."""
    return jsonify({"ok": True}), 200


@app.post("/api/process-video")
def process_video_batch() -> tuple:
    """Batch-process an uploaded video and return a link to the annotated MP4."""
    try:
        uploaded_video = get_uploaded_video()
    except ValueError as exc:
        return error_response(str(exc), 400)

    run_id = uuid4().hex[:12]
    input_video_path = save_upload_to_disk(uploaded_video, run_id)
    output_filename = f"{run_id}_annotated.mp4"
    output_video_path = PROCESSED_DIR / output_filename

    weights_path = resolve_weights_path()
    confidence_threshold, image_size = get_inference_options()

    logger.info("Starting batch inference (run_id=%s)", run_id)
    try:
        run_result = infer_video(
            weights_path=weights_path,
            video_path=input_video_path,
            output_path=output_video_path,
            conf=confidence_threshold,
            imgsz=image_size,
            track=True,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        logger.exception("Inference failed")
        return error_response(str(exc), 400)
    except Exception as exc:  # noqa: BLE001 - surface unexpected errors as 500
        logger.exception("Unexpected error during inference")
        return error_response(f"Unexpected error: {exc}", 500)

    return (
        jsonify(
            {
                "run_id": run_id,
                "annotated_video_url": absolute_url_for(f"/processed/{output_filename}"),
                "metrics": run_result["metrics"],
            }
        ),
        200,
    )


@app.get("/processed/<path:filename>")
def serve_processed_video(filename: str):
    """Static-file route for downloading processed videos."""
    return send_from_directory(PROCESSED_DIR, filename)


@app.post("/api/stream-video")
def start_live_stream() -> tuple:
    """Spin up a new live MJPEG stream for the uploaded video."""
    global active_streamer

    try:
        uploaded_video = get_uploaded_video()
    except ValueError as exc:
        return error_response(str(exc), 400)

    run_id = uuid4().hex[:12]
    input_video_path = save_upload_to_disk(uploaded_video, run_id)

    weights_path = resolve_weights_path()
    confidence_threshold, image_size = get_inference_options()

    try:
        next_streamer = VideoStreamer(
            weights_path=weights_path,
            video_path=input_video_path,
            conf=confidence_threshold,
            imgsz=image_size,
            track=True,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        logger.exception("Failed to create streamer")
        return error_response(str(exc), 400)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error creating streamer")
        return error_response(f"Unexpected error: {exc}", 500)

    replace_active_streamer(next_streamer)

    logger.info("Live stream ready (run_id=%s)", run_id)
    return (
        jsonify(
            {
                "run_id": run_id,
                "stream_url": absolute_url_for(f"/api/video-stream?run_id={run_id}"),
                "metrics_url": absolute_url_for("/api/stream-metrics"),
            }
        ),
        200,
    )


@app.get("/api/video-stream")
def serve_live_mjpeg() -> Response:
    """MJPEG endpoint the browser displays inside an `<img>` tag."""
    if active_streamer is None:
        return error_response("No stream in progress", 400)

    return Response(
        active_streamer.stream_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/stream-metrics")
def get_live_stream_metrics() -> tuple:
    """Return a JSON snapshot of detection + tracking metrics for the live stream."""
    if active_streamer is None:
        return error_response("No stream in progress", 400)

    return jsonify(active_streamer.get_metrics()), 200


@app.post("/api/stream-stop")
def stop_live_stream() -> tuple:
    """Tell the active streamer to halt and clear it."""
    global active_streamer

    if active_streamer is not None:
        active_streamer.stop()
        active_streamer = None
        logger.info("Live stream stopped by client")

    return jsonify({"ok": True}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
