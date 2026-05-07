# Salamander Tracker

Base code for training a YOLO salamander detector from video footage.

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Workflow

1. Extract frames from a salamander video for labeling:

```bash
python scripts/extract_frames.py --video path/to/video.mp4 --every-n 15 --output-dir data/frames/raw
```

2. Label extracted frames in Label Studio and export in YOLO with Images format.

3. Build train/val dataset from Label Studio export:

```bash
python scripts/prepare_dataset.py --export-dir path/to/project-export
```

4. Train YOLO on the prepared dataset:

```bash
python scripts/train.py --data data/dataset/dataset.yaml --model yolo11n.pt --epochs 50 --imgsz 640
```

5. Run video inference and write an annotated output video:

```bash
python scripts/infer_video.py --weights runs/detect/salamander_run1/weights/best.pt --video path/to/video.mp4 --output data/output/annotated.mp4 --track
```

Use `--track` to enable stable track IDs across frames for downstream metrics.
