"""Stable per-id extraction from an Ultralytics BoT-SORT tracking result.

Kept separate from detector.py so the (track_id, bbox, keypoints)
shape this module hands back is independent of exactly how the
underlying `model.track(...)` call was made.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class TrackedPerson:
    track_id: int
    bbox: tuple  # (x1, y1, x2, y2) pixels
    keypoints: Optional[np.ndarray]  # (17, 3) x, y, conf -- None for plain-detect results
    det_conf: float


def extract_tracks(result) -> List[TrackedPerson]:
    """`result` is one Ultralytics `Results` object from `model.track(...,
    persist=True)`. Frames where BoT-SORT hasn't assigned an id yet
    (id is None) are dropped -- an id-less detection can't be attributed
    to a per-track state machine."""
    tracks: List[TrackedPerson] = []
    boxes = result.boxes
    if boxes is None or boxes.id is None:
        return tracks

    ids = boxes.id.cpu().numpy()
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    kps_all = None
    if getattr(result, "keypoints", None) is not None and result.keypoints.data is not None:
        kps_all = result.keypoints.data.cpu().numpy()  # (n, 17, 3)

    for i in range(len(ids)):
        tracks.append(
            TrackedPerson(
                track_id=int(ids[i]),
                bbox=tuple(float(v) for v in xyxy[i]),
                keypoints=kps_all[i] if kps_all is not None else None,
                det_conf=float(confs[i]),
            )
        )
    return tracks


def should_drop_track(ts: float, last_seen_ts: float, grace_sec: float) -> bool:
    """Field-verification fix: a single missed detection frame on real
    footage (motion blur, brief occlusion, a profile pose the detector
    only catches intermittently) used to be treated as "the person
    left", wiping the track's accumulated evidence/calibration/debounce
    state via on_track_lost(). Real gaps of a few hundred milliseconds
    to ~1 second are routine -- BoT-SORT itself is built to tolerate
    them -- so a track is only actually dropped once it's been missing
    continuously for TRACK_LOST_GRACE_SEC, not on the very first missed
    frame."""
    return (ts - last_seen_ts) >= grace_sec
