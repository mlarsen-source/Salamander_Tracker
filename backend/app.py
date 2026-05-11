import os
from pathlib import Path
from uuid import uuid4

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

from scripts.infer_video import infer_video

BASE_DIR = Path(__file__).resolve().parent
UPLOADS_DIR = BASE_DIR / "uploads"
PROCESSED_DIR = BASE_DIR / "processed"
DEFAULT_WEIGHTS = BASE_DIR / "models" / "best.pt"

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
CORS(app)


@app.get("/api/health")
def health() -> tuple:
    return jsonify({"ok": True}), 200


@app.post("/api/process-video")
def process_video() -> tuple:
    if "video" not in request.files:
        return jsonify({"error": "Missing 'video' file field"}), 400

    video_file = request.files["video"]
    if not video_file or not video_file.filename:
        return jsonify({"error": "No file selected"}), 400

    run_id = uuid4().hex[:12]
    safe_name = secure_filename(video_file.filename)
    input_path = UPLOADS_DIR / f"{run_id}_{safe_name}"
    output_name = f"{run_id}_annotated.mp4"
    output_path = PROCESSED_DIR / output_name

    video_file.save(input_path)

    weights_override = request.form.get("weights_path")
    if weights_override:
        weights_path = Path(weights_override)
    else:
        env_weights = os.getenv("MODEL_WEIGHTS")
        weights_path = Path(env_weights) if env_weights else DEFAULT_WEIGHTS

    try:
        run_result = infer_video(
            weights_path=weights_path,
            video_path=input_path,
            output_path=output_path,
            conf=float(request.form.get("conf", 0.25)),
            imgsz=int(request.form.get("imgsz", 640)),
            track=True,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400

    annotated_video_url = request.host_url.rstrip("/") + f"/processed/{output_name}"

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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
