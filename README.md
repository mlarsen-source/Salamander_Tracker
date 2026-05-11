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

## Model Training Workflow

If you want to retrain the model with new data:

### 1. Extract Frames from Video

```bash
python backend/scripts/extract_frames.py \
  --video path/to/video.mp4 \
  --target-frames 100 \
  --output-dir backend/data/frames/raw
```

### 2. Label Frames in Label Studio

1. Create a Label Studio project
2. Import frames from `backend/data/frames/raw`
3. Label salamanders with bounding boxes
4. Export as "YOLO" format

### 3. Prepare Dataset

```bash
python backend/scripts/prepare_dataset.py \
  --export-dir path/to/label-studio-export
```

Creates `backend/data/dataset/` with train/val split.

### 4. Train Model

```bash
python backend/scripts/train.py \
  --data backend/data/dataset/dataset.yaml \
  --model yolo11n.pt \
  --epochs 50 \
  --imgsz 640
```

Weights saved to `backend/models/best.pt`

### 5. Copy Weights to App

```bash
cp runs/detect/salamander_run1/weights/best.pt backend/models/best.pt
```

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
