# Salamander Tracker

A real-time web application for detecting and tracking salamanders in video using a custom-trained YOLO model. Upload a video and watch live detection with bounding boxes as the model processes frames.

## Features

- 🎥 **Live Streaming** - See detection results in real-time as frames are processed
- 📊 **Real-time Metrics** - Track detections per frame and individual salamander screen time
- 🤖 **Custom YOLO Model** - Trained specifically on salamander footage
- 🌐 **Web Interface** - Clean, responsive UI for easy use
- 📈 **Tracking** - Individual salamanders tracked across frames

## Architecture

```
Salamander_Tracker/
├── frontend/                    # Next.js 15 React app
│   ├── src/app/
│   │   ├── page.tsx            # Main upload + streaming UI
│   │   └── globals.css         # Styling
│   └── src/lib/
│       └── api.ts              # API client functions
│
├── backend/                     # Python Flask app
│   ├── app.py                  # Main Flask server
│   ├── models/
│   │   └── best.pt             # Trained YOLO weights
│   ├── scripts/
│   │   ├── stream_video.py     # MJPEG streaming engine
│   │   ├── infer_video.py      # Frame inference logic
│   │   ├── train.py            # Model training
│   │   └── extract_frames.py   # Video → frames
│   ├── uploads/                # Temporary uploaded videos
│   └── processed/              # Processed videos (optional)
```

## Component Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Browser (Next.js Frontend)                   │
│                                                                     │
│  ┌─────────────┐   ┌─────────────────┐   ┌──────────────────────┐  │
│  │ UploadForm  │   │   LiveStream    │   │   MetricsPanel       │  │
│  │             │   │                 │   │                      │  │
│  │ Pick a video│   │ Renders the     │   │ Displays detection   │  │
│  │ file and    │   │ live MJPEG feed │   │ counts over time,    │  │
│  │ start / stop│   │ with scan-line  │   │ time on screen per   │  │
│  │ the stream  │   │ overlay         │   │ salamander, and live │  │
│  │             │   │                 │   │ coordinate readouts  │  │
│  └──────┬──────┘   └────────┬────────┘   └──────────┬───────────┘  │
│         └────────────────────┴────────────────────────┘             │
│                        api.ts — typed fetch helpers                 │
│                        useStreamMetrics — polls every 500 ms        │
└──────────────────────────────┬──────────────────────────────────────┘
                               │  HTTP / MJPEG
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        Python Flask Backend                         │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  app.py  —  API routes, file handling, CORS                 │   │
│  └───────────────────────┬──────────────────────────────────────┘   │
│                          │                                          │
│          ┌───────────────┴──────────────┐                           │
│          ▼                              ▼                           │
│  ┌────────────────────┐     ┌───────────────────────┐              │
│  │  stream_video.py   │     │  detection_metrics.py │              │
│  │                    │────▶│                       │              │
│  │  Reads video frames│     │  Calculates per-frame │              │
│  │  runs YOLO, encodes│     │  counts, track dura-  │              │
│  │  MJPEG, tracks IDs │     │  tions, and bounding  │              │
│  │                    │     │  box coordinates      │              │
│  └─────────┬──────────┘     └───────────────────────┘              │
│            │                                                        │
│            ▼                                                        │
│  ┌─────────────────────┐                                            │
│  │   models/best.pt    │                                            │
│  │                     │                                            │
│  │  YOLOv11 Nano model │                                            │
│  │  trained on 145     │                                            │
│  │  salamander frames  │                                            │
│  └─────────────────────┘                                            │
└─────────────────────────────────────────────────────────────────────┘
```

## Prerequisites

- Python 3.12+
- Node.js 18+
- 4GB RAM minimum (8GB recommended)

## Quick Start

### 1. Backend Setup

```bash
cd backend
pip install -r requirements.txt
```

### 2. Frontend Setup

```bash
cd frontend
npm install
```

### 3. Start the Backend Server

```bash
cd backend
python app.py
```

The server runs on `http://127.0.0.1:8000`

### 4. Start the Frontend (in another terminal)

```bash
cd frontend
npm run dev
```

The app opens at `http://localhost:3000`

### 5. Upload and Stream

1. Go to http://localhost:3000
2. Select a video file
3. Click **"Start Stream"**
4. Watch the live detection stream with real-time metrics

## API Endpoints

### Stream Processing

- `POST /api/stream-video` - Start streaming a video
  - Body: FormData with `video` field
  - Returns: `{ run_id, stream_url, metrics_url }`

- `GET /api/video-stream` - MJPEG stream of annotated frames
  - Returns: Continuous MJPEG stream

- `GET /api/stream-metrics` - Current stream metrics
  - Returns: `{ frame_count, fps, is_streaming, detection_count_over_time, time_on_screen_by_track_id }`

- `POST /api/stream-stop` - Stop the current stream
  - Returns: `{ ok: true }`

### Health

- `GET /api/health` - Server health check
  - Returns: `{ ok: true }`

## Metrics Explained

### Detection Count Over Time

Array of detections per frame:

```json
[
  { "frame": 0, "timestamp_sec": 0.0, "count": 2 },
  { "frame": 1, "timestamp_sec": 0.033, "count": 2 }
]
```

### Time on Screen by Track ID

Total duration each tracked salamander appears:

```json
{
  "1": 5.23, // Track ID 1: 5.23 seconds
  "2": 3.45 // Track ID 2: 3.45 seconds
}
```

## Dataset & Training Pipeline

### Dataset Overview

The model was trained on **145 labeled frames** extracted from camera-trap salamander footage. The dataset captures salamanders in realistic field conditions with varying lighting, backgrounds, and body positions. This ensures the model generalizes well to real-world detection scenarios.

**Dataset Split:**

- **Training set:** 123 frames (~85%)
- **Validation set:** 22 frames (~15%)

**Data Characteristics:**

- Single-class detection (Salamander)
- Annotated with bounding boxes using Label Studio
- Resolution: 640×640 (standard for YOLOv11)
- Covers varied conditions: different times of day, substrate types, and lighting angles

**Training Configuration:**

- Model: YOLOv11 Nano (`yolo11n.pt`) — compact for real-time inference
- Epochs: 50 (early stopping at convergence)
- Image size: 640×640
- Batch size: 8
- Optimizer: SGD with momentum
- Device: Automatic (GPU if available, CPU fallback)

**Training Results:**
The nano model achieves good performance on validation data while maintaining fast inference (~0.4 seconds/frame on CPU), making it ideal for real-time streaming without requiring expensive GPU hardware.

### Retrain the Model with New Data

If you want to train with additional labeled frames:

#### 1. Extract Frames from Video

```bash
python backend/scripts/extract_frames.py \
  --video path/to/video.mp4 \
  --target-frames 100 \
  --output-dir backend/data/frames/raw
```

#### 2. Label Frames in Label Studio

1. Create a Label Studio project
2. Import frames from `backend/data/frames/raw`
3. Label salamanders with bounding boxes (single class: "Salamander")
4. Export as "YOLO" format to `backend/data/labelstudio/`

#### 3. Prepare Dataset

```bash
python backend/scripts/prepare_dataset.py \
  --export-dir backend/data/labelstudio
```

Creates `backend/data/dataset/` with train/val split and YOLO annotations.

#### 4. Train Model

```bash
python backend/scripts/train.py \
  --data backend/data/dataset/dataset.yaml \
  --model yolo11n.pt \
  --epochs 50 \
  --imgsz 640 \
  --batch 8
```

Training outputs saved to `runs/detect/salamander_run*/`

#### 5. Copy Best Weights to App

```bash
cp runs/detect/salamander_run1/weights/best.pt backend/models/best.pt
```

## Color Masking vs YOLO: When to Use Which

### Color Masking

**How it works:** Select a target color range, threshold pixels in that range, compute the centroid, and log coordinates.

**Strengths:**

- ✅ Extremely fast (~1–2 ms/frame; orders of magnitude faster than ML)
- ✅ No training required; works immediately on new videos
- ✅ Requires minimal compute; runs on any device
- ✅ Deterministic and interpretable

**Limitations:**

- ❌ Fails with textured or non-uniform backgrounds
- ❌ Breaks when lighting changes (mid-recording, shadows, sun angle shift)
- ❌ Cannot distinguish multiple salamanders reliably if they overlap or have similar colors
- ❌ No confidence scores; false positives (matching background pixels)
- ❌ Requires manual color calibration per video

### YOLO Object Detection

**How it works:** Neural network trained to detect salamanders via learned visual features. Returns bounding boxes, confidence scores, and optional tracking IDs across frames.

**Strengths:**

- ✅ Robust to lighting changes, backgrounds, and textures
- ✅ Handles multiple salamanders, overlaps, and occlusions
- ✅ Generalizes to new videos without recalibration
- ✅ Provides confidence scores for each detection
- ✅ Integrated tracking assigns persistent IDs across frames
- ✅ Computes derived metrics (time on screen, path trails, heatmaps)

**Limitations:**

- ❌ Slower than color masking (~0.4 sec/frame on CPU, ~50–100 ms on GPU)
- ❌ Requires labeled training data (145 frames for this model)
- ❌ Model quality depends on training data diversity
- ❌ Initial setup overhead; needs environment and GPU optional

### Conclusion

**Use color masking** when conditions are tightly controlled and speed is paramount (e.g., laboratory setup with constant lighting and plain backgrounds).

**Use YOLO** for real-world ecological fieldwork where lighting, backgrounds, and salamander density vary. The ~0.4 sec/frame cost is negligible for analysis of 30-minute recordings and far outweighs manual recalibration of color thresholds per video.

## Environment Variables

Optional configuration via `.env`:

```bash
# Backend
MODEL_WEIGHTS=backend/models/best.pt

# Frontend
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000
```

## Performance Notes

- Processing speed: ~0.4 seconds/frame on CPU (nano model)
- ~30-second video = ~5-7 minutes processing time on CPU
- GPU support available if CUDA is detected automatically
- JPEG quality: 85% (good balance of quality and bandwidth)

## Troubleshooting

### "No module named ultralytics"

```bash
cd backend
pip install -r requirements.txt
```

### Frontend can't reach backend

- Ensure backend is running on http://127.0.0.1:8000
- Check `NEXT_PUBLIC_API_BASE_URL` in frontend env
- Check CORS is enabled in Flask app

### Video upload fails

- Check file size (backend accepts up to Flask's default 16MB)
- Verify video codec is supported (h264, VP9 recommended)
- Check disk space in `backend/uploads/`

### Slow processing

- Running on CPU is normal (~0.4 sec/frame)
- Disable debug mode in production: change `debug=True` to `debug=False` in `backend/app.py`

## Project Timeline

- Frame extraction: 1-2 hours
- Dataset labeling: 2-3 hours
- Model training: 30 mins - 2 hours (depending on data size)
- Web app development: ~4 hours
- **Total: 8-12 hours from raw video to deployed app**

## References

- [Ultralytics YOLOv11](https://docs.ultralytics.com/)
- [Label Studio](https://labelstud.io/)
- [Flask](https://flask.palletsprojects.com/)
- [Next.js](https://nextjs.org/)
