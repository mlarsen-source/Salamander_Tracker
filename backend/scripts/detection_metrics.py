"""Helpers for collecting detection + tracking metrics from YOLO results.

Both the batch processor (`infer_video.py`) and the live streamer
(`stream_video.py`) need to do the same bookkeeping per frame:

  * record how many salamanders were detected on this frame
  * accumulate how many seconds each tracked salamander has been on screen

Keeping that logic here means both call sites stay short and identical.
"""
from __future__ import annotations

from typing import Any


def build_frame_detection_record(
    frame_index: int, frames_per_second: float, detection_count: int
) -> dict[str, Any]:
    """Return one entry for the `detection_count_over_time` series."""
    return {
        "frame": frame_index,
        "timestamp_sec": frame_index / frames_per_second,
        "count": detection_count,
    }


def add_frame_to_track_durations(
    yolo_result: Any,
    frames_per_second: float,
    durations_by_track_id: dict[str, float],
) -> None:
    """Add one frame's worth of screen time to every tracked salamander.

    Mutates `durations_by_track_id` in place. Skips silently when the
    YOLO result has no tracking IDs (e.g. tracking was disabled or no
    boxes were detected on this frame).
    """
    if yolo_result.boxes.id is None:
        return

    seconds_per_frame = 1.0 / frames_per_second
    track_ids_on_frame = {
        str(int(raw_id))
        for raw_id in yolo_result.boxes.id.cpu().numpy().tolist()
    }
    for track_id in track_ids_on_frame:
        durations_by_track_id[track_id] = (
            durations_by_track_id.get(track_id, 0.0) + seconds_per_frame
        )


def round_track_durations(
    durations_by_track_id: dict[str, float],
) -> dict[str, float]:
    """Return a copy with durations rounded for clean JSON output."""
    return {
        track_id: round(seconds, 3)
        for track_id, seconds in durations_by_track_id.items()
    }
