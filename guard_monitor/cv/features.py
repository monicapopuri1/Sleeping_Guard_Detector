"""Size-free feature extraction from a single person's 17 COCO keypoints.

Every feature here is a ratio or an angle, never a raw pixel distance.
That is what lets one set of thresholds work across gate houses with
unknown camera height, unknown lens, and unknown mounting distance:
person_h (the detection bbox height, in pixels) is the *only* scale
anchor, and every linear feature is divided by it before being
compared to a threshold. Angles are naturally scale-free already, but
still get referenced to a per-camera baseline (see BaselineCalibrator)
so an oddly-mounted camera doesn't read as a permanently forward-leaning
guard.

Keypoint arrays are (17, 3): columns are (x, y, confidence), using the
standard COCO-pose keypoint order that Ultralytics YOLO11n-pose emits.

Fix 4 (field verification): every feature below has a one-sided
fallback. A guard turned ~90 degrees to camera only shows one shoulder
and one hip -- the original code required both, silently degrading to
"can't tell" (features read as upright/neutral) exactly when a
profile view is often the clearest angle to see a head-on-hand. Now a
single confident point stands in for the midpoint, and hand_at_head
checks every (wrist, head-keypoint) pair instead of requiring the nose
specifically. Callers get a `degraded` flag alongside these features so
the CSV log can record when a reading is one-sided.
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np

NOSE = 0
LEFT_EYE = 1
RIGHT_EYE = 2
LEFT_EAR = 3
RIGHT_EAR = 4
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16

HEAD_KEYPOINTS = (NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR)
TORSO_KEYPOINTS = (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP)

# Fix 1 item 3: joint-weighted motion. Wrists and elbows carry a hand
# gesture or a stand-up far more reliably than a nose/shoulder jitter
# of the same pixel magnitude, so they count double.
MOTION_JOINT_WEIGHTS = {
    NOSE: 1,
    LEFT_SHOULDER: 1,
    RIGHT_SHOULDER: 1,
    LEFT_ELBOW: 2,
    RIGHT_ELBOW: 2,
    LEFT_WRIST: 2,
    RIGHT_WRIST: 2,
    LEFT_HIP: 1,
    RIGHT_HIP: 1,
}


def person_height(bbox) -> float:
    """bbox = (x1, y1, x2, y2) in pixels. The pixel-scale anchor: every
    other linear feature is divided by this so features stay comparable
    regardless of camera distance or resolution."""
    x1, y1, x2, y2 = bbox
    return float(y2 - y1)


def _xy(kps: np.ndarray, idx: int) -> np.ndarray:
    return kps[idx, :2]


def _conf(kps: np.ndarray, idx: int) -> float:
    return float(kps[idx, 2])


def keypoints_visible(kps: np.ndarray, indices, min_conf: float) -> bool:
    return all(_conf(kps, i) >= min_conf for i in indices)


def body_visible(kps: np.ndarray, min_conf: float) -> bool:
    """Shoulders or hips confidently detected -- enough to know the
    torso pose even if the head is occluded (e.g. face-down on desk)."""
    shoulders = _conf(kps, LEFT_SHOULDER) >= min_conf or _conf(kps, RIGHT_SHOULDER) >= min_conf
    hips = _conf(kps, LEFT_HIP) >= min_conf or _conf(kps, RIGHT_HIP) >= min_conf
    return shoulders or hips


def head_conf_low(kps: np.ndarray, min_conf: float) -> bool:
    """True when no head keypoint is confidently detected -- the
    "head keypoints confidence < KP_MIN_CONF" suspect condition, which
    covers a guard slumped face-down where the pose model can't see a
    face at all."""
    return not any(_conf(kps, i) >= min_conf for i in HEAD_KEYPOINTS)


def _paired_point(kps: np.ndarray, left_idx: int, right_idx: int, min_conf: float) -> Optional[np.ndarray]:
    """Fix 4 item 1: midpoint if both sides are confident, the single
    confident side if only one is, None if neither is."""
    left_ok = _conf(kps, left_idx) >= min_conf
    right_ok = _conf(kps, right_idx) >= min_conf
    if left_ok and right_ok:
        return (_xy(kps, left_idx) + _xy(kps, right_idx)) / 2
    if left_ok:
        return _xy(kps, left_idx)
    if right_ok:
        return _xy(kps, right_idx)
    return None


def shoulder_point(kps: np.ndarray, min_conf: float) -> Optional[np.ndarray]:
    return _paired_point(kps, LEFT_SHOULDER, RIGHT_SHOULDER, min_conf)


def hip_point(kps: np.ndarray, min_conf: float) -> Optional[np.ndarray]:
    return _paired_point(kps, LEFT_HIP, RIGHT_HIP, min_conf)


def shoulder_mid(kps: np.ndarray) -> np.ndarray:
    """Unconditional midpoint, ignoring confidence. Kept for callers
    (e.g. calibration) that already gated on both sides being visible."""
    return (_xy(kps, LEFT_SHOULDER) + _xy(kps, RIGHT_SHOULDER)) / 2


def hip_mid(kps: np.ndarray) -> np.ndarray:
    return (_xy(kps, LEFT_HIP) + _xy(kps, RIGHT_HIP)) / 2


def torso_len(kps: np.ndarray, person_h: float) -> float:
    """norm(shoulder_mid - hip_mid) / person_h -- torso length as a
    fraction of bbox height, independent of camera distance."""
    if person_h <= 0:
        return 0
    return float(np.linalg.norm(shoulder_mid(kps) - hip_mid(kps)) / person_h)


def angle_from_vertical_deg(vec: np.ndarray) -> float:
    """Signed angle, in degrees, between `vec` (image coords, y grows
    downward) and the upright vertical axis. 0 = perfectly upright;
    magnitude grows as the vector tilts toward horizontal. Sign
    indicates left/right lean and is what lets calibration subtract out
    a fixed camera-mounting bias before we look at the magnitude."""
    dx, dy = float(vec[0]), float(vec[1])
    return float(np.degrees(np.arctan2(dx, -dy)))


def head_tilt_deg(kps: np.ndarray, baseline_deg: float, min_conf: float) -> "tuple[float, bool]":
    """angle(head_point, shoulder_point) - camera_baseline, in degrees.
    Returns (value, degraded) -- degraded is True when either point had
    to fall back to a single-sided keypoint (Fix 4 item 1) rather than a
    true midpoint, or False-with-value-0 when neither side of the torso
    is visible at all (caller should treat that as "no reading")."""
    shoulder = shoulder_point(kps, min_conf)
    if shoulder is None:
        return 0, True
    head_pts = [kps[i] for i in HEAD_KEYPOINTS if kps[i, 2] >= min_conf]
    if not head_pts:
        return 0, True
    head_xy = np.mean([p[:2] for p in head_pts], axis=0)
    # "degraded" tracks the shoulder-pair fallback, not the head-point
    # count: a single confident nose is the normal case, not a fallback.
    degraded = not (_conf(kps, LEFT_SHOULDER) >= min_conf and _conf(kps, RIGHT_SHOULDER) >= min_conf)
    vec = head_xy - shoulder
    return angle_from_vertical_deg(vec) - baseline_deg, degraded


def spine_lean_deg(kps: np.ndarray, baseline_deg: float, min_conf: float) -> "tuple[float, bool]":
    """angle(shoulder_point, hip_point) - camera_baseline, in degrees."""
    shoulder = shoulder_point(kps, min_conf)
    hip = hip_point(kps, min_conf)
    if shoulder is None or hip is None:
        return 0, True
    degraded = not (
        _conf(kps, LEFT_SHOULDER) >= min_conf and _conf(kps, RIGHT_SHOULDER) >= min_conf
        and _conf(kps, LEFT_HIP) >= min_conf and _conf(kps, RIGHT_HIP) >= min_conf
    )
    vec = shoulder - hip
    return angle_from_vertical_deg(vec) - baseline_deg, degraded


def hand_at_head_ratio(kps: np.ndarray, person_h: float, min_conf: float) -> "tuple[Optional[float], bool]":
    """Fix 4 item 1: min(norm(wrist - head_kp) / person_h) over every
    confident (wrist, head-keypoint) pair -- not just the nose, and not
    requiring both wrists. Returns (ratio_or_None, degraded)."""
    if person_h <= 0:
        return None, False
    wrists = [w for w in (LEFT_WRIST, RIGHT_WRIST) if _conf(kps, w) >= min_conf]
    heads = [h for h in HEAD_KEYPOINTS if _conf(kps, h) >= min_conf]
    if not wrists or not heads:
        return None, False
    ratios = [
        float(np.linalg.norm(_xy(kps, w) - _xy(kps, h)) / person_h)
        for w in wrists
        for h in heads
    ]
    degraded = len(wrists) < 2 or len(heads) < 2
    return min(ratios), degraded


def hand_at_head(kps: np.ndarray, person_h: float, min_conf: float, thresh: float) -> bool:
    ratio, _ = hand_at_head_ratio(kps, person_h, min_conf)
    return ratio is not None and ratio < thresh


def motion_score(
    prev_kps: np.ndarray,
    curr_kps: np.ndarray,
    person_h: float,
    min_conf: float,
    dt: float,
) -> float:
    """Weighted mean joint displacement / person_h / dt across the
    joints confidently detected in both frames -- a size-free RATE of
    how fast the person is moving (person-heights per second), not a
    raw per-tick displacement. Wrists/elbows are weighted double
    (MOTION_JOINT_WEIGHTS) since hand and arm movement is the more
    reliable "the person is not asleep" signal (Fix 1 item 3).

    Dividing by dt (rather than treating "one tick" as a fixed unit of
    time) is what makes this comparable across different processing
    frame rates -- e.g. a deployment sampling at 3 fps instead of the
    native 20 fps to cut inference cost sees the same real motion as
    the same score, not something ~7x larger just because more real
    time elapsed between two processed frames. STILL_THRESH /
    WAKE_LONG_THRESH / WAKE_SHORT_THRESH / CALIB_MAX_MOTION are
    calibrated in these per-second units."""
    if person_h <= 0 or dt <= 0:
        return 0
    weighted_displacements = []
    weights = []
    for idx, weight in MOTION_JOINT_WEIGHTS.items():
        if _conf(prev_kps, idx) >= min_conf and _conf(curr_kps, idx) >= min_conf:
            d = np.linalg.norm(_xy(curr_kps, idx) - _xy(prev_kps, idx))
            weighted_displacements.append(d * weight)
            weights.append(weight)
    if not weighted_displacements:
        return 0
    return float(np.sum(weighted_displacements) / np.sum(weights) / person_h / dt)


class SlidingWindowSmoother:
    """Rolling mean over the last `window_sec` seconds. Used both for the
    long-window (SLIDING_WINDOW) damping of single-frame jitter, and for
    the short-window (WAKE_SHORT_WINDOW_SEC) burst detector -- same
    mechanism, different window length."""

    def __init__(self, window_sec: float):
        self.window_sec = window_sec
        self._samples: deque = deque()

    def push(self, ts: float, value: float) -> float:
        self._samples.append((ts, value))
        cutoff = ts - self.window_sec
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()
        return self.mean()

    def mean(self) -> float:
        if not self._samples:
            return 0
        return float(np.mean([v for _, v in self._samples]))


class BboxMotionTracker:
    """Fix 1 item 2: a bbox-geometry wake backstop that keypoint noise cannot
    fake. Standing up roughly doubles bbox height and moves its center
    by a large fraction of the person's own height; jittery keypoints on
    an otherwise-static bbox never do either. Keeps a short history
    (bbox_wake_window_sec) and compares the current bbox against the
    oldest sample still inside that window."""

    def __init__(self, window_sec: float, height_wake_frac: float, center_wake_frac: float):
        self.window_sec = window_sec
        self.height_wake_frac = height_wake_frac
        self.center_wake_frac = center_wake_frac
        self._history: deque = deque()  # (ts, height, center_x, center_y)

    def reset(self) -> None:
        """Called after a detection gap larger than MOTION_GAP_RESET_SEC
        so the next bbox isn't compared against a stale pre-gap reference
        (which could look like a spurious height/center jump over an
        unknown multi-frame gap and misfire the wake backstop)."""
        self._history.clear()

    def push(self, ts: float, bbox, person_h: float) -> bool:
        x1, y1, x2, y2 = bbox
        height = y2 - y1
        center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
        self._history.append((ts, height, center))
        cutoff = ts - self.window_sec
        while len(self._history) > 1 and self._history[0][0] < cutoff:
            self._history.popleft()

        ref_ts, ref_height, ref_center = self._history[0]
        if ref_ts == ts or ref_height <= 0 or person_h <= 0:
            return False

        height_change_frac = abs(height - ref_height) / ref_height
        center_move_frac = float(np.linalg.norm(center - ref_center)) / person_h

        return height_change_frac > self.height_wake_frac or center_move_frac > self.center_wake_frac


class BaselineCalibrator:
    """First CALIBRATION_SEC seconds of a track's footage: record the
    median raw (un-referenced) spine_lean angle while the guard looks
    upright, i.e. normal shift-start posture. From CALIBRATION_SEC
    onward every angle is measured relative to that median, so a camera
    mounted at an odd angle doesn't read as a permanently slouching
    guard.

    Fix 3 (field verification): a video that starts with the guard
    already asleep polluted the whole baseline with a sleeping-posture
    reading, making every later upright frame look "tilted" by
    comparison. Now a frame only counts toward the baseline if it looks
    like a plausible upright posture in its own right (small raw
    head_tilt/spine_lean, low motion, head visible); if too few clean
    seconds accumulate, DEFAULT_BASELINE_SPINE is used instead of
    blocking startup on calibration. A slow, gated re-baseline nudges
    the baseline later if the guard settles into a slightly different
    upright posture than the calibration window happened to catch.
    """

    def __init__(
        self,
        calibration_sec: float,
        default_baseline: float,
        max_head_tilt: float,
        max_spine_lean: float,
        max_motion: float,
        min_clean_sec: float,
    ):
        self.calibration_sec = calibration_sec
        self.default_baseline = default_baseline
        self.max_head_tilt = max_head_tilt
        self.max_spine_lean = max_spine_lean
        self.max_motion = max_motion
        self.min_clean_sec = min_clean_sec

        self._start_ts: Optional[float] = None
        self._last_observe_ts: Optional[float] = None
        self._clean_samples: list = []
        self._clean_seconds: float = 0
        self.baseline_deg: float = 0
        self._done = False
        self.used_default = False
        self._active_low_tilt_since: Optional[float] = None

    def active(self, ts: float) -> bool:
        if self._done:
            return False
        if self._start_ts is None:
            self._start_ts = ts
        return (ts - self._start_ts) < self.calibration_sec

    def observe(
        self,
        ts: float,
        head_visible: bool,
        head_tilt_raw: float,
        spine_lean_raw: float,
        motion_long: float,
    ) -> None:
        if not self.active(ts):
            return
        dt = (ts - self._last_observe_ts) if self._last_observe_ts is not None else 0
        self._last_observe_ts = ts

        clean = (
            head_visible
            and abs(head_tilt_raw) < self.max_head_tilt
            and abs(spine_lean_raw) < self.max_spine_lean
            and motion_long < self.max_motion
        )
        if clean:
            self._clean_samples.append(spine_lean_raw)
            self._clean_seconds += dt

    def finalize_if_ready(self, ts: float) -> float:
        if not self._done and not self.active(ts):
            if self._clean_seconds >= self.min_clean_sec and self._clean_samples:
                self.baseline_deg = float(np.median(self._clean_samples))
            else:
                self.baseline_deg = self.default_baseline
                self.used_default = True
            self._done = True
        return self.baseline_deg

    def maybe_rebaseline(
        self,
        ts: float,
        is_active: bool,
        head_tilt_raw: float,
        spine_lean_raw: float,
        min_active_sec: float,
        max_head_tilt: float,
        max_shift_deg: float,
        blend_old: float,
    ) -> None:
        """Fix 3 item 3: silent re-baselining. Only runs after calibration has
        finalized; blends a slowly-confirmed upright reading into the
        existing baseline rather than replacing it outright."""
        if not self._done:
            return
        if not (is_active and abs(head_tilt_raw) < max_head_tilt):
            self._active_low_tilt_since = None
            return
        if self._active_low_tilt_since is None:
            self._active_low_tilt_since = ts
            return
        if ts - self._active_low_tilt_since < min_active_sec:
            return
        shift = abs(spine_lean_raw - self.baseline_deg)
        if shift < max_shift_deg:
            self.baseline_deg = blend_old * self.baseline_deg + (1 - blend_old) * spine_lean_raw
        self._active_low_tilt_since = ts
