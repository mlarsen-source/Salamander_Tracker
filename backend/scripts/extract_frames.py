import argparse
from pathlib import Path

import cv2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--output-dir", default="data/frames/raw")
    parser.add_argument("--every-n", type=int, default=None)
    parser.add_argument("--target-frames", type=int, default=150)
    parser.add_argument("--prefix", default="frame")
    args = parser.parse_args()

    if args.every_n is not None and args.every_n < 1:
        raise SystemExit("--every-n must be >= 1")
    if args.target_frames < 1:
        raise SystemExit("--target-frames must be >= 1")

    video_path = Path(args.video)
    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")

    frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    every_n = args.every_n
    if every_n is None:
        if frame_total > 0:
            every_n = max(1, round(frame_total / args.target_frames))
        else:
            every_n = 15

    frame_index = 0
    saved = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_index % every_n == 0:
            out_path = output_dir / f"{args.prefix}_{frame_index:06d}.jpg"
            cv2.imwrite(str(out_path), frame)
            saved += 1

        frame_index += 1

    cap.release()
    print(f"Sampling every {every_n} frame(s)")
    print(f"Read {frame_index} frames")
    print(f"Saved {saved} frames to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
