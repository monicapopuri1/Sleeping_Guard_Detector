"""Optional debug/visualization output: draws bboxes, skeletons, and
per-track state onto frames and writes a continuous annotated MP4.

This is a dev/validation aid, not part of the alert pipeline -- it
never feeds back into detection logic, so it deliberately lives
outside guard_monitor/cv/, guard_monitor/state_machine/, and
guard_monitor/alert/ (the packages held to the no-magic-number rule).
Drawing constants here (colors, font scale, thickness) are cosmetic,
not detection thresholds.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from guard_monitor.cv import features as feat
from guard_monitor.state_machine.guard import GuardState

_SKELETON = [
    (feat.NOSE, feat.LEFT_EYE), (feat.NOSE, feat.RIGHT_EYE),
    (feat.LEFT_EYE, feat.LEFT_EAR), (feat.RIGHT_EYE, feat.RIGHT_EAR),
    (feat.LEFT_SHOULDER, feat.RIGHT_SHOULDER),
    (feat.LEFT_SHOULDER, feat.LEFT_ELBOW), (feat.LEFT_ELBOW, feat.LEFT_WRIST),
    (feat.RIGHT_SHOULDER, feat.RIGHT_ELBOW), (feat.RIGHT_ELBOW, feat.RIGHT_WRIST),
    (feat.LEFT_SHOULDER, feat.LEFT_HIP), (feat.RIGHT_SHOULDER, feat.RIGHT_HIP),
    (feat.LEFT_HIP, feat.RIGHT_HIP),
    (feat.LEFT_HIP, feat.LEFT_KNEE), (feat.LEFT_KNEE, feat.LEFT_ANKLE),
    (feat.RIGHT_HIP, feat.RIGHT_KNEE), (feat.RIGHT_KNEE, feat.RIGHT_ANKLE),
]

_STATE_COLOR = {
    GuardState.ABSENT: (128, 128, 128),
    GuardState.ACTIVE: (80, 200, 80),
    GuardState.STILL: (0, 210, 210),
    GuardState.SLEEPING_SUSPECT: (0, 140, 255),
    GuardState.ALERT: (0, 0, 255),
    GuardState.POST_ALERT: (0, 0, 180),
    GuardState.WOKE_UP: (255, 200, 0),
}

_OCCUPIED_COLOR = (80, 200, 80)
_VACANT_COLOR = (0, 0, 255)


_LABEL_FONT_SCALE = 0.6
_HUD_FONT_SCALE = 0.45


def _put_text(frame, text, org, color, font_scale, thickness):
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)


def draw_skeleton(frame: np.ndarray, kps: np.ndarray, min_conf: float, color) -> None:
    for a, b in _SKELETON:
        if kps[a, 2] >= min_conf and kps[b, 2] >= min_conf:
            pa = (int(kps[a, 0]), int(kps[a, 1]))
            pb = (int(kps[b, 0]), int(kps[b, 1]))
            cv2.line(frame, pa, pb, color, 2, cv2.LINE_AA)
    for idx in range(kps.shape[0]):
        if kps[idx, 2] >= min_conf:
            p = (int(kps[idx, 0]), int(kps[idx, 1]))
            cv2.circle(frame, p, 3, color, -1, cv2.LINE_AA)


def draw_person(
    frame: np.ndarray,
    track_id: int,
    bbox,
    keypoints: Optional[np.ndarray],
    state: GuardState,
    min_conf: float,
    hud_lines,
) -> None:
    color = _STATE_COLOR.get(state, (255, 255, 255))
    x1, y1, x2, y2 = (int(v) for v in bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    if keypoints is not None:
        draw_skeleton(frame, keypoints, min_conf, color)

    label = f"id{track_id} {state.value}"
    _put_text(frame, label, (x1, max(0, y1 - 8)), color, _LABEL_FONT_SCALE, 2)

    for i, line in enumerate(hud_lines):
        _put_text(frame, line, (x1, y2 + 18 + i * 16), color, _HUD_FONT_SCALE, 1)


def draw_roi(frame: np.ndarray, polygon, occupied: bool) -> None:
    color = _OCCUPIED_COLOR if occupied else _VACANT_COLOR
    pts = np.array([[int(x), int(y)] for x, y in polygon], dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)
    label = "OCCUPIED" if occupied else "VACANT"
    origin = tuple(pts[0, 0])
    _put_text(frame, label, origin, color, _LABEL_FONT_SCALE, 2)


class AnnotatedVideoWriter:
    """Lazily opens a cv2.VideoWriter on the first frame it sees, so the
    caller doesn't need to know frame size up front."""

    def __init__(self, out_path: str, fps: float):
        self.out_path = out_path
        self.fps = fps
        self._writer: Optional[cv2.VideoWriter] = None

    def write(self, frame: np.ndarray) -> None:
        if self._writer is None:
            Path(self.out_path).parent.mkdir(parents=True, exist_ok=True)
            height, width = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(self.out_path, fourcc, self.fps, (width, height))
        self._writer.write(frame)

    def release(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
