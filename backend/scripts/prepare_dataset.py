import argparse
import random
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-dir", required=True)
    parser.add_argument("--output", default="data/dataset")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    src = Path(args.export_dir)
    dst = Path(args.output)
    src_images = src / "images"
    src_labels = src / "labels"
    classes_file = src / "classes.txt"

    if not src_images.is_dir() or not src_labels.is_dir():
        raise SystemExit("Expected YOLO export folder with images/ and labels/")

    image_paths = sorted(p for p in src_images.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not image_paths:
        raise SystemExit(f"No images found in {src_images}")

    random.seed(args.seed)
    random.shuffle(image_paths)
    n_val = max(1, int(len(image_paths) * args.val_fraction))
    val_paths = image_paths[:n_val]
    train_paths = image_paths[n_val:]

    if dst.exists():
        shutil.rmtree(dst)
    for split in ("train", "val"):
        (dst / "images" / split).mkdir(parents=True)
        (dst / "labels" / split).mkdir(parents=True)

    def copy_pair(img_path: Path, split: str) -> None:
        label_path = src_labels / (img_path.stem + ".txt")
        shutil.copy(img_path, dst / "images" / split / img_path.name)
        if label_path.exists():
            shutil.copy(label_path, dst / "labels" / split / label_path.name)

    for p in train_paths:
        copy_pair(p, "train")
    for p in val_paths:
        copy_pair(p, "val")

    if classes_file.exists():
        names = [line.strip() for line in classes_file.read_text().splitlines() if line.strip()]
    else:
        names = ["salamander"]

    yaml_path = dst / "dataset.yaml"
    lines = [
        f"path: {dst.resolve()}",
        "train: images/train",
        "val: images/val",
        "",
        "names:",
    ]
    for i, name in enumerate(names):
        lines.append(f"  {i}: {name}")
    yaml_path.write_text("\n".join(lines) + "\n")

    print(f"Wrote {yaml_path}")
    print(f"train images: {len(train_paths)}")
    print(f"val images: {len(val_paths)}")
    print(f"classes: {names}")


if __name__ == "__main__":
    main()
