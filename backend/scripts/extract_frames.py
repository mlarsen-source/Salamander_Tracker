import argparse
import random
from pathlib import Path

import cv2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--output-dir", default="data/frames/raw")
    parser.add_argument("--every-n", type=int, default=None)
    parser.add_argument("--target-frames", type=int, default=150)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--prefix", default="frame")
    parser.add_argument("--random", action="store_true", help="Sample random frames")
    parser.add_argument("--seed", type=int, default=None, help="Seed for random sampling")
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

    if args.start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    if args.random:
        saved = save_random_frames(
            cap=cap,
            output_dir=output_dir,
            prefix=args.prefix,
            frame_total=frame_total,
            start_frame=args.start_frame,
            target_frames=args.target_frames,
            seed=args.seed,
        )
        cap.release()
        print("Sampling random frames")
        print(f"Read {frame_total} frames")
        print(f"Saved {saved} frames to {output_dir.resolve()}")
        return

    frame_index = args.start_frame
    saved = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if (frame_index - args.start_frame) % every_n == 0:
            out_path = output_dir / f"{args.prefix}_{frame_index:06d}.jpg"
            if not cv2.imwrite(str(out_path), frame):
                raise SystemExit(f"Failed to write frame: {out_path}")
            saved += 1

        frame_index += 1

    cap.release()
    print(f"Sampling every {every_n} frame(s)")
    print(f"Read {frame_index} frames")
    print(f"Saved {saved} frames to {output_dir.resolve()}")


def save_random_frames(
    cap: cv2.VideoCapture,
    output_dir: Path,
    prefix: str,
    frame_total: int,
    start_frame: int,
    target_frames: int,
    seed: int | None,
) -> int:
    if frame_total <= 0:
        raise SystemExit("Random sampling requires a video with a known frame count")
    if start_frame >= frame_total:
        raise SystemExit("--start-frame must be less than the video frame count")

    available_frame_indices = [
        frame_index
        for frame_index in range(start_frame, frame_total)
        if not (output_dir / f"{prefix}_{frame_index:06d}.jpg").exists()
    ]
    if len(available_frame_indices) < target_frames:
        raise SystemExit(
            f"Only {len(available_frame_indices)} unsaved frames are available; "
            f"cannot save {target_frames}"
        )

    rng = random.Random(seed)
    selected_frame_indices = sorted(rng.sample(available_frame_indices, target_frames))
    selected_lookup = set(selected_frame_indices)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_index = start_frame
    saved = 0

    while frame_index < frame_total:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_index in selected_lookup:
            out_path = output_dir / f"{prefix}_{frame_index:06d}.jpg"
            if not cv2.imwrite(str(out_path), frame):
                raise SystemExit(f"Failed to write frame: {out_path}")
            saved += 1
            if saved == target_frames:
                break

        frame_index += 1

    return saved


if __name__ == "__main__":
    main()
