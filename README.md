# Salamander Tracker

A full-stack app for detecting and tracking salamanders in video with a
custom-trained YOLO11 model. The frontend uploads a video and displays a live
MJPEG stream from the Flask backend while the backend runs YOLO inference,
draws bounding boxes, and reports tracking metrics.

## Features

- Live video inference with YOLO object detection
- MJPEG stream rendered in the browser
- Detection count over time
- Live bounding-box center coordinates
- Per-track time on screen for detected salamanders
- Single-class model trained from Label Studio YOLO annotations

## Project Layout

```text
Salamander_Tracker/
  backend/
    app.py                         Flask API server
    models/best.pt                 Active model used by the app
    yolo11n.pt                     Base YOLO11 nano weights
    requirements.txt
    scripts/
      extract_frames.py            Video frame extraction
      prepare_dataset.py           Label Studio export to YOLO dataset split
      train.py                     YOLO training entrypoint
      infer_video.py               Batch video inference
      stream_video.py              Live MJPEG inference stream
      detection_metrics.py         Metrics helpers
    data/
      frames/raw/                  Raw frames for Label Studio upload
      images/                      Label Studio YOLO export images
      labels/                      Label Studio YOLO export labels
      classes.txt                  Label Studio class names
      dataset/                     Prepared YOLO train/val dataset
    runs/detect/                   Training outputs

  frontend/
    src/app/page.tsx               Main app page
    src/app/components/            Upload, stream, and metrics components
    src/lib/api.ts                 Backend API client
    src/lib/useStreamMetrics.ts    Metrics polling hook
```

## Current Model

The active model at `backend/models/best.pt` was trained from the current
Label Studio export in `backend/data`. This custom-trained YOLO model file is
included in the repo at `backend/models/best.pt`, and the Flask app loads it by
default.

Labeling count:

- 370 frames were reviewed/labeled in Label Studio
- 299 frames contain at least one salamander bounding box
- 71 frames are background/negative examples with no salamander box

Dataset summary:

- 370 images
- 370 label files
- 329 bounding boxes
- 71 background/negative images
- Class: `salamander`
- Train split: 296 images
- Validation split: 74 images

Training summary:

- Base model: `yolo11n.pt`
- Epochs: 50
- Image size: 640
- Batch size: 8
- Device used: CPU
- Output run: `backend/runs/detect/salamander_run2`

Final validation metrics:

- Precision: 0.958
- Recall: 0.915
- mAP50: 0.940
- mAP50-95: 0.725

Dataset and training pipeline: frames were extracted from salamander videos into
`backend/data/frames/raw`, imported into Label Studio, labeled with a single
`salamander` bounding-box class, and exported in YOLO format to
`backend/data/images`, `backend/data/labels`, and `backend/data/classes.txt`.
The `scripts/prepare_dataset.py` script rebuilt a YOLO train/validation split
under `backend/data/dataset` using an 80/20 split. The `scripts/train.py` script
then fine-tuned `yolo11n.pt` for 50 epochs at 640 px image size and batch size 8,
writing training artifacts to `backend/runs/detect/salamander_run2`.

## Color Masking vs YOLO

Color masking is best when the scene is controlled: stable lighting, a plain
background, and a target color that is clearly separated from everything else.
It is fast, simple, and useful for quick lab-style tracking, but it breaks down
when shadows, substrate texture, camera exposure, or similar-colored objects
change the pixel colors. YOLO is the better choice for this salamander tracker
because the videos contain natural backgrounds, changing lighting, and irregular
animal poses. Use color masking for simple controlled experiments where speed is
more important than robustness; use YOLO for field footage, mixed backgrounds,
multiple animals, partial occlusion, and any workflow where detection quality
matters more than raw processing speed.

## Prerequisites

- Python 3.12+
- Node.js 18+
- Git Bash, PowerShell, or another terminal
- 8 GB RAM recommended

The examples below use Bash-style commands. On this Windows workspace, the
virtual environment Python executable is:

```bash
../.venv/Scripts/python.exe
```

If you are on macOS or Linux, replace that with `python` or your virtualenv's
Python path.

## Run the App

Terminal 1, start the backend:

```bash
cd backend
../.venv/Scripts/python.exe app.py
```

Terminal 2, start the frontend:

```bash
cd frontend
npm install
npm run dev
```

Open:

```text
http://localhost:3000
```

Then select a video file and click `Start Stream`.

## Backend Setup

```bash
cd backend
../.venv/Scripts/python.exe -m pip install -r requirements.txt
```

Start the Flask backend:

```bash
cd backend
../.venv/Scripts/python.exe app.py
```

The backend runs at:

```text
http://127.0.0.1:8000
```

## Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

The frontend runs at:

```text
http://localhost:3000
```

## Using the App

1. Start the backend.
2. Start the frontend.
3. Open `http://localhost:3000`.
4. Select a video file.
5. Click `Start Stream`.
6. Watch the annotated detection stream and metrics.

## API Endpoints

- `GET /api/health`
  - Health check.

- `POST /api/stream-video`
  - Uploads a video and starts a live inference stream.
  - Body: multipart form data with a `video` file field.
  - Returns: `run_id`, `stream_url`, and `metrics_url`.

- `GET /api/video-stream`
  - Returns the live annotated MJPEG stream.

- `GET /api/stream-metrics`
  - Returns current frame count, FPS, detection counts, current positions, and
    per-track time on screen.

- `POST /api/stream-stop`
  - Stops the current live stream.

- `POST /api/process-video`
  - Batch-processes a full video and writes an annotated MP4.

## Frame Extraction

Raw frames for labeling live in:

```text
backend/data/frames/raw
```

Extract evenly spaced frames:

```bash
cd backend
../.venv/Scripts/python.exe scripts/extract_frames.py \
  --video uploads/videoplayback.mp4 \
  --output-dir data/frames/raw \
  --target-frames 150 \
  --prefix videoplayback
```

Extract random frames:

```bash
cd backend
../.venv/Scripts/python.exe scripts/extract_frames.py \
  --video uploads/videoplayback.mp4 \
  --output-dir data/frames/raw \
  --target-frames 150 \
  --prefix videoplayback \
  --random \
  --seed 20260524
```

The random extraction mode skips existing frame filenames so it does not
overwrite previously extracted frames.

## Label Studio Workflow

1. Upload frames from `backend/data/frames/raw` to Label Studio.
2. Label salamanders with bounding boxes.
3. Use a single class named `salamander`.
4. Export the project in YOLO format.
5. Place the YOLO export contents in `backend/data`:

```text
backend/data/images/
backend/data/labels/
backend/data/classes.txt
backend/data/notes.json
```

`notes.json` is export metadata. The dataset preparation script uses
`images/`, `labels/`, and `classes.txt`.

## Prepare the Dataset

From `backend`, run:

```bash
../.venv/Scripts/python.exe scripts/prepare_dataset.py \
  --export-dir data \
  --output data/dataset \
  --val-fraction 0.2 \
  --seed 42
```

This rebuilds:

```text
backend/data/dataset/images/train
backend/data/dataset/images/val
backend/data/dataset/labels/train
backend/data/dataset/labels/val
backend/data/dataset/dataset.yaml
```

The script deletes and recreates the output dataset directory each time.

## Train the Model

From `backend`, run:

```bash
../.venv/Scripts/python.exe scripts/train.py \
  --data data/dataset/dataset.yaml \
  --model yolo11n.pt \
  --epochs 50 \
  --imgsz 640 \
  --batch 8 \
  --name salamander_run2 \
  --project runs/detect \
  --device cpu
```

Training outputs are written under:

```text
backend/runs/detect/<run-name>
```

The important output files are:

```text
backend/runs/detect/<run-name>/weights/best.pt
backend/runs/detect/<run-name>/weights/last.pt
backend/runs/detect/<run-name>/results.csv
backend/runs/detect/<run-name>/results.png
```

Use `best.pt` for the app unless you have a specific reason to use `last.pt`.

## Install the Trained Model

Copy the best checkpoint into the active model path:

```bash
cd backend
cp runs/detect/salamander_run2/weights/best.pt models/best.pt
```

Restart the Flask backend after replacing `models/best.pt`.

## Environment Variables

Optional backend model override:

```bash
MODEL_WEIGHTS=models/best.pt
```

Optional frontend API base URL:

```bash
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000
```

## Troubleshooting

### `No module named ultralytics`

Install backend dependencies:

```bash
cd backend
../.venv/Scripts/python.exe -m pip install -r requirements.txt
```

### Frontend cannot reach backend

- Confirm the backend is running at `http://127.0.0.1:8000`.
- Confirm `NEXT_PUBLIC_API_BASE_URL` points to the backend.
- Restart the frontend after changing environment variables.

### Training writes outside this repo

Use the `--project runs/detect` argument. The training script resolves this to
an absolute path under `backend/runs/detect`.

### Training is slow

CPU training is expected to be slow. The 50-epoch `salamander_run2` training
run took about 100 minutes on CPU. Use a CUDA-capable GPU if available.

## References

- Ultralytics YOLO: https://docs.ultralytics.com/
- Label Studio: https://labelstud.io/
- Flask: https://flask.palletsprojects.com/
- Next.js: https://nextjs.org/
