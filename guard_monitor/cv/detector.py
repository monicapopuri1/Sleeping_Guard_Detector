"""Ultralytics YOLO11n-pose / YOLO11n wrappers with BoT-SORT tracking.

Two separate pipelines, per the architecture:

    source -> YOLO11n-pose -> BoT-SORT -> per-id buffer   (sleep detection)
    source -> YOLO11n      -> BoT-SORT -> ROI              (vacant-post)

Both are pre-trained, off-the-shelf weights -- no training, no labels.
Each wrapper calls `model.track(..., tracker="botsort.yaml", persist=True)`
frame by frame so BoT-SORT keeps its own id continuity across the call
sequence, then hands the raw Ultralytics `Results` off to
guard_monitor.cv.tracker.extract_tracks for a plain-data view.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_BOTSORT_TRACKER_CONFIG = "botsort.yaml"


class _TrackingModel:
    def __init__(self, weights: str, device: str = "cpu"):
        from ultralytics import YOLO  # deferred: heavy import, keeps state_machine/tests GPU-free

        self.model = YOLO(weights)
        self.device = device

    def track(self, frame):
        results = self.model.track(
            source=frame,
            tracker=_BOTSORT_TRACKER_CONFIG,
            persist=True,
            device=self.device,
            verbose=False,
        )
        return results[0]


class PoseDetector(_TrackingModel):
    """YOLO11n-pose: person bbox + 17 COCO keypoints, tracked by id."""

    def __init__(self, weights: str = "yolo11n-pose.pt", device: str = "cpu"):
        super().__init__(weights, device)


class PersonDetector(_TrackingModel):
    """YOLO11n: person bbox only, tracked by id -- feeds vacant-post ROI logic."""

    def __init__(self, weights: str = "yolo11n.pt", device: str = "cpu"):
        super().__init__(weights, device)

    def track(self, frame):
        result = super().track(frame)
        if result.boxes is not None and result.boxes.cls is not None:
            person_cls_id = 0  # COCO class 0 == "person" in yolo11n's pretrained head
            keep = result.boxes.cls.cpu().numpy() == person_cls_id
            result.boxes = result.boxes[keep]
        return result
