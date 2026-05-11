import logging
import os
import sys
from pathlib import Path
from uuid import uuid4

from flask import Flask, jsonify, request, send_from_directory, Response
from flask_cors import CORS
from werkzeug.utils import secure_filename

from scripts.infer_video import infer_video
from scripts.stream_video import VideoStreamer

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
UPLOADS_DIR = BASE_DIR / "uploads"
PROCESSED_DIR = BASE_DIR / "processed"
DEFAULT_WEIGHTS = BASE_DIR / "models" / "best.pt"

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
CORS(app)

# Global streamer instance
current_streamer = None


@app.get("/api/health")
def health() -> tuple:
    return jsonify({"ok": True}), 200


@app.post("/api/process-video")
def process_video() -> tuple:
    logger.info("=== /api/process-video request received ===")
    
    if "video" not in request.files:
        logger.error("Missing 'video' file field in request")
        return jsonify({"error": "Missing 'video' file field"}), 400

    video_file = request.files["video"]
    if not video_file or not video_file.filename:
        logger.error("No file selected")
        return jsonify({"error": "No file selected"}), 400

    logger.info(f"Video file received: {video_file.filename}")

    run_id = uuid4().hex[:12]
    safe_name = secure_filename(video_file.filename)
    input_path = UPLOADS_DIR / f"{run_id}_{safe_name}"
    output_name = f"{run_id}_annotated.mp4"
    output_path = PROCESSED_DIR / output_name

    logger.info(f"Saving upload to {input_path}")
    video_file.save(input_path)
    logger.info(f"File saved. Size: {input_path.stat().st_size} bytes")

    weights_override = request.form.get("weights_path")
    if weights_override:
        weights_path = Path(weights_override)
    else:
        env_weights = os.getenv("MODEL_WEIGHTS")
        weights_path = Path(env_weights) if env_weights else DEFAULT_WEIGHTS

    logger.info(f"Using weights: {weights_path}")
    logger.info(f"Weights exist: {weights_path.exists()}")

    try:
        logger.info("Starting video inference...")
        run_result = infer_video(
            weights_path=weights_path,
            video_path=input_path,
            output_path=output_path,
            conf=float(request.form.get("conf", 0.25)),
            imgsz=int(request.form.get("imgsz", 640)),
            track=True,
        )
        logger.info("Inference completed successfully")
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        logger.error(f"Inference failed: {exc}", exc_info=True)
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logger.error(f"Unexpected error during inference: {exc}", exc_info=True)
        return jsonify({"error": f"Unexpected error: {str(exc)}"}), 500

    annotated_video_url = request.host_url.rstrip("/") + f"/processed/{output_name}"
    logger.info(f"Returning response with URL: {annotated_video_url}")

    return (
        jsonify(
            {
                "run_id": run_id,
                "annotated_video_url": annotated_video_url,
                "metrics": run_result["metrics"],
            }
        ),
        200,
    )


@app.get("/processed/<path:filename>")
def serve_processed(filename: str):
    return send_from_directory(PROCESSED_DIR, filename)


@app.post("/api/stream-video")
def stream_video_start() -> tuple:
    """Start streaming a video with YOLO detection."""
    global current_streamer

    logger.info("=== /api/stream-video request received ===")

    if "video" not in request.files:
        logger.error("Missing 'video' file field in request")
        return jsonify({"error": "Missing 'video' file field"}), 400

    video_file = request.files["video"]
    if not video_file or not video_file.filename:
        logger.error("No file selected")
        return jsonify({"error": "No file selected"}), 400

    logger.info(f"Video file received: {video_file.filename}")

    run_id = uuid4().hex[:12]
    safe_name = secure_filename(video_file.filename)
    input_path = UPLOADS_DIR / f"{run_id}_{safe_name}"

    logger.info(f"Saving upload to {input_path}")
    video_file.save(input_path)
    logger.info(f"File saved. Size: {input_path.stat().st_size} bytes")

    weights_override = request.form.get("weights_path")
    if weights_override:
        weights_path = Path(weights_override)
    else:
        env_weights = os.getenv("MODEL_WEIGHTS")
        weights_path = Path(env_weights) if env_weights else DEFAULT_WEIGHTS

    logger.info(f"Using weights: {weights_path}")
    logger.info(f"Weights exist: {weights_path.exists()}")

    try:
        logger.info("Creating video streamer...")
        current_streamer = VideoStreamer(
            weights_path=weights_path,
            video_path=input_path,
            conf=float(request.form.get("conf", 0.25)),
            imgsz=int(request.form.get("imgsz", 640)),
            track=True,
        )
        logger.info(f"Streamer created and set globally. current_streamer={current_streamer}")
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        logger.error(f"Failed to create streamer: {exc}", exc_info=True)
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logger.error(f"Unexpected error creating streamer: {exc}", exc_info=True)
        return jsonify({"error": f"Unexpected error: {str(exc)}"}), 500

    return (
        jsonify(
            {
                "run_id": run_id,
                "stream_url": request.host_url.rstrip("/") + "/api/video-stream",
                "metrics_url": request.host_url.rstrip("/") + "/api/stream-metrics",
            }
        ),
        200,
    )


@app.get("/api/video-stream")
def video_stream() -> Response:
    """Stream video frames as MJPEG."""
    global current_streamer

    if current_streamer is None:
        return jsonify({"error": "No stream in progress"}), 400

    try:
        return Response(
            current_streamer.stream_frames(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )
    except Exception as exc:
        logger.error(f"Stream error: {exc}", exc_info=True)
        return jsonify({"error": str(exc)}), 500


@app.get("/api/stream-metrics")
def stream_metrics() -> tuple:
    """Get current streaming metrics."""
    global current_streamer

    logger.info(f"Metrics request. current_streamer={current_streamer}")
    
    if current_streamer is None:
        logger.warning("Metrics requested but no stream in progress")
        return jsonify({"error": "No stream in progress"}), 400

    try:
        metrics = current_streamer.get_metrics()
        logger.debug(f"Returned metrics: frame_count={metrics.get('frame_count')}")
        return jsonify(metrics), 200
    except Exception as exc:
        logger.error(f"Failed to get metrics: {exc}", exc_info=True)
        return jsonify({"error": str(exc)}), 500


@app.post("/api/stream-stop")
def stream_stop() -> tuple:
    """Stop the current stream."""
    global current_streamer

    if current_streamer is not None:
        current_streamer.stop()
        current_streamer = None
        logger.info("Stream stopped by user")

    return jsonify({"ok": True}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
