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


def build_current_position_records(yolo_result: Any) -> list[dict[str, Any]]:
    """Return bounding-box center coordinates for detections on the current frame."""
    boxes = yolo_result.boxes
    if len(boxes) == 0:
        return []

    xyxy_boxes = boxes.xyxy.cpu().numpy().tolist()
    confidences = boxes.conf.cpu().numpy().tolist() if boxes.conf is not None else []
    track_ids = boxes.id.cpu().numpy().tolist() if boxes.id is not None else []

    records: list[dict[str, Any]] = []
    for index, (x1, y1, x2, y2) in enumerate(xyxy_boxes):
        width = x2 - x1
        height = y2 - y1
        record = {
            "track_id": str(int(track_ids[index])) if index < len(track_ids) else None,
            "x_center": round(x1 + width / 2, 1),
            "y_center": round(y1 + height / 2, 1),
            "width": round(width, 1),
            "height": round(height, 1),
        }
        if index < len(confidences):
            record["confidence"] = round(float(confidences[index]), 3)
        records.append(record)

    return records


def round_track_durations(
    durations_by_track_id: dict[str, float],
) -> dict[str, float]:
    """Return a copy with durations rounded for clean JSON output."""
    return {
        track_id: round(seconds, 3)
        for track_id, seconds in durations_by_track_id.items()
    }
