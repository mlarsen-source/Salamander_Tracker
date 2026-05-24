import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/dataset/dataset.yaml")
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--name", default="salamander_run1")
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    project_path = Path(args.project).resolve()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
        project=str(project_path),
        device=args.device,
    )


if __name__ == "__main__":
    main()
