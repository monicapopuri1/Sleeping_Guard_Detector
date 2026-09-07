"""Writes the CLIP_DURATION_SEC MP4 clip centered on an alert.

runner.py keeps a rolling FrameRingBuffer of recent (timestamp, frame)
pairs; on alert it hands the relevant slice to `save_clip`, which is
the only place in this package that touches cv2.VideoWriter.
"""

from __future__ import annotations

import logging
from collections import deque
from pathlib import Path
from typing import Deque, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class FrameRingBuffer:
    """Keeps the last `keep_sec` seconds of frames so a clip can be
    built centered on an alert that is only confirmed after the fact."""

    def __init__(self, keep_sec: float):
        self.keep_sec = keep_sec
        self._frames: Deque[Tuple[float, np.ndarray]] = deque()

    def push(self, ts: float, frame: np.ndarray) -> None:
        self._frames.append((ts, frame.copy()))
        cutoff = ts - self.keep_sec
        while self._frames and self._frames[0][0] < cutoff:
            self._frames.popleft()

    def slice(self, center_ts: float, half_window_sec: float):
        lo = center_ts - half_window_sec
        hi = center_ts + half_window_sec
        return [(ts, f) for ts, f in self._frames if lo <= ts <= hi]


def save_clip(frames, out_path: str, fps: float) -> None:
    """`frames` is a list of (ts, BGR ndarray) tuples, oldest first."""
    if not frames:
        logger.warning("save_clip called with no frames, skipping: %s", out_path)
        return
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0][1].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    try:
        for _, frame in frames:
            writer.write(frame)
    finally:
        writer.release()
