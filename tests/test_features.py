"""Synthetic-keypoint tests for the size-free feature functions.

Uses the actual COCO keypoint indices: 0=nose, 5/6=shoulders,
9/10=wrists, 11/12=hips.
"""

import numpy as np
import pytest

from guard_monitor.cv import features as feat

MIN_CONF = 0.35


def make_keypoints(points):
    """points: {index: (x, y, conf)}, everything else defaults to (0,0,0)."""
    kps = np.zeros((17, 3))
    for idx, (x, y, c) in points.items():
        kps[idx] = [x, y, c]
    return kps


def test_person_height():
    assert feat.person_height((0, 0, 10, 50)) == 50


def test_shoulder_and_hip_midpoints():
    kps = make_keypoints({
        feat.LEFT_SHOULDER: (0, 0, 1),
        feat.RIGHT_SHOULDER: (10, 0, 1),
        feat.LEFT_HIP: (0, 20, 1),
        feat.RIGHT_HIP: (10, 20, 1),
    })
    np.testing.assert_allclose(feat.shoulder_mid(kps), [5, 0])
    np.testing.assert_allclose(feat.hip_mid(kps), [5, 20])


def test_torso_len_is_size_free():
    small = make_keypoints({
        feat.LEFT_SHOULDER: (0, 0, 1), feat.RIGHT_SHOULDER: (10, 0, 1),
        feat.LEFT_HIP: (0, 20, 1), feat.RIGHT_HIP: (10, 20, 1),
    })
    big = make_keypoints({
        feat.LEFT_SHOULDER: (0, 0, 1), feat.RIGHT_SHOULDER: (20, 0, 1),
        feat.LEFT_HIP: (0, 40, 1), feat.RIGHT_HIP: (20, 40, 1),
    })
    ratio_small = feat.torso_len(small, person_h=50)
    ratio_big = feat.torso_len(big, person_h=100)
    assert ratio_small == pytest.approx(ratio_big)


def test_angle_from_vertical_straight_up_is_zero():
    assert feat.angle_from_vertical_deg(np.array([0, -10])) == pytest.approx(0)


def test_angle_from_vertical_45_degrees():
    assert feat.angle_from_vertical_deg(np.array([10, -10])) == pytest.approx(45)


def test_head_tilt_upright_is_zero():
    kps = make_keypoints({
        feat.NOSE: (5, -10, 1),
        feat.LEFT_SHOULDER: (0, 0, 1), feat.RIGHT_SHOULDER: (10, 0, 1),
    })
    value, degraded = feat.head_tilt_deg(kps, baseline_deg=0, min_conf=MIN_CONF)
    assert value == pytest.approx(0)
    assert degraded is False


def test_head_tilt_forward_lean_detected():
    # nose offset sideways from shoulder_mid by the same distance it is
    # above it -> 45 degree tilt.
    kps = make_keypoints({
        feat.NOSE: (15, -10, 1),
        feat.LEFT_SHOULDER: (0, 0, 1), feat.RIGHT_SHOULDER: (10, 0, 1),
    })
    value, _ = feat.head_tilt_deg(kps, baseline_deg=0, min_conf=MIN_CONF)
    assert value == pytest.approx(45)


def test_head_tilt_baseline_subtraction_cancels_camera_bias():
    kps = make_keypoints({
        feat.NOSE: (15, -10, 1),
        feat.LEFT_SHOULDER: (0, 0, 1), feat.RIGHT_SHOULDER: (10, 0, 1),
    })
    raw, _ = feat.head_tilt_deg(kps, baseline_deg=0, min_conf=MIN_CONF)
    corrected, _ = feat.head_tilt_deg(kps, baseline_deg=raw, min_conf=MIN_CONF)
    assert corrected == pytest.approx(0)


def test_spine_lean_upright_is_zero():
    kps = make_keypoints({
        feat.LEFT_SHOULDER: (0, 0, 1), feat.RIGHT_SHOULDER: (10, 0, 1),
        feat.LEFT_HIP: (0, 20, 1), feat.RIGHT_HIP: (10, 20, 1),
    })
    value, degraded = feat.spine_lean_deg(kps, baseline_deg=0, min_conf=MIN_CONF)
    assert value == pytest.approx(0)
    assert degraded is False


def test_hand_at_head_true_when_close():
    kps = make_keypoints({
        feat.NOSE: (0, 0, 1),
        feat.LEFT_WRIST: (1, 0, 1),
        feat.RIGHT_WRIST: (100, 100, 1),
    })
    assert feat.hand_at_head(kps, person_h=100, min_conf=MIN_CONF, thresh=0.45) is True


def test_hand_at_head_false_when_far():
    kps = make_keypoints({
        feat.NOSE: (0, 0, 1),
        feat.LEFT_WRIST: (100, 100, 1),
        feat.RIGHT_WRIST: (100, 100, 1),
    })
    assert feat.hand_at_head(kps, person_h=100, min_conf=MIN_CONF, thresh=0.45) is False


def test_hand_at_head_ratio_none_when_no_wrist_visible():
    kps = make_keypoints({feat.NOSE: (0, 0, 1)})
    ratio, degraded = feat.hand_at_head_ratio(kps, person_h=100, min_conf=MIN_CONF)
    assert ratio is None
    assert degraded is False


# --------------------------------------------------------------------
# Fix 4: one-sided fallbacks -- a guard turned ~90deg to camera only
# shows one shoulder/hip/wrist, and the original code silently read
# that as "no signal" instead of falling back to what IS visible.
# --------------------------------------------------------------------

def test_hand_at_head_true_with_only_right_side_visible():
    """The required regression test: every LEFT-side keypoint has
    conf 0.0, only the RIGHT side is confident. hand_at_head must still
    fire when the (visible) right wrist is near the nose."""
    kps = make_keypoints({
        feat.NOSE: (0, 0, 1),
        feat.RIGHT_WRIST: (1, 0, 1),
        # left wrist / left eye / left ear left at conf 0 implicitly
    })
    assert kps[feat.LEFT_WRIST, 2] == 0
    assert kps[feat.LEFT_EYE, 2] == 0
    assert kps[feat.LEFT_EAR, 2] == 0
    assert feat.hand_at_head(kps, person_h=100, min_conf=MIN_CONF, thresh=0.45) is True


def test_hand_at_head_uses_eye_or_ear_when_nose_not_visible():
    kps = make_keypoints({
        feat.RIGHT_EAR: (0, 0, 1),
        feat.RIGHT_WRIST: (1, 0, 1),
    })
    assert feat.hand_at_head(kps, person_h=100, min_conf=MIN_CONF, thresh=0.45) is True


def test_shoulder_point_falls_back_to_single_side():
    kps = make_keypoints({feat.RIGHT_SHOULDER: (10, 0, 1)})
    point = feat.shoulder_point(kps, MIN_CONF)
    np.testing.assert_allclose(point, [10, 0])


def test_shoulder_point_none_when_neither_side_visible():
    kps = make_keypoints({})
    assert feat.shoulder_point(kps, MIN_CONF) is None


def test_head_tilt_degraded_but_computed_with_one_shoulder():
    kps = make_keypoints({
        feat.NOSE: (10, -10, 1),
        feat.RIGHT_SHOULDER: (10, 0, 1),
        # left shoulder not visible
    })
    value, degraded = feat.head_tilt_deg(kps, baseline_deg=0, min_conf=MIN_CONF)
    assert value == pytest.approx(0)  # nose directly above the one visible shoulder
    assert degraded is True


def test_spine_lean_degraded_but_computed_with_one_hip_one_shoulder():
    kps = make_keypoints({
        feat.RIGHT_SHOULDER: (5, -20, 1),
        feat.RIGHT_HIP: (5, 0, 1),
    })
    value, degraded = feat.spine_lean_deg(kps, baseline_deg=0, min_conf=MIN_CONF)
    assert value == pytest.approx(0)
    assert degraded is True


def test_head_tilt_no_reading_when_no_torso_visible():
    kps = make_keypoints({feat.NOSE: (0, 0, 1)})
    value, degraded = feat.head_tilt_deg(kps, baseline_deg=0, min_conf=MIN_CONF)
    assert value == pytest.approx(0)
    assert degraded is True


# --------------------------------------------------------------------
# Fix 1.3: joint-weighted motion
# --------------------------------------------------------------------

def test_motion_score_zero_when_static():
    kps = make_keypoints({
        feat.NOSE: (5, 5, 1),
        feat.LEFT_SHOULDER: (0, 0, 1), feat.RIGHT_SHOULDER: (10, 0, 1),
    })
    assert feat.motion_score(kps, kps, person_h=100, min_conf=MIN_CONF, dt=1) == pytest.approx(0)


def test_motion_score_scales_with_displacement_over_person_h():
    prev = make_keypoints({feat.NOSE: (0, 0, 1)})
    curr = make_keypoints({feat.NOSE: (10, 0, 1)})
    assert feat.motion_score(prev, curr, person_h=100, min_conf=MIN_CONF, dt=1) == pytest.approx(0.1)


def test_motion_score_ignores_low_confidence_joints():
    prev = make_keypoints({feat.NOSE: (0, 0, 1), feat.LEFT_SHOULDER: (0, 0, 0.1)})
    curr = make_keypoints({feat.NOSE: (0, 0, 1), feat.LEFT_SHOULDER: (1000, 1000, 0.1)})
    assert feat.motion_score(prev, curr, person_h=100, min_conf=MIN_CONF, dt=1) == pytest.approx(0)


def test_motion_score_weights_wrist_double_shoulder():
    # Same displacement (10px over person_h=100 -> 0.1 raw ratio each),
    # but a wrist move should pull the weighted mean further toward 0.1
    # than an equal-magnitude nose-only move would, when mixed with an
    # unweighted joint at zero displacement.
    prev = make_keypoints({
        feat.NOSE: (0, 0, 1),
        feat.LEFT_WRIST: (0, 0, 1),
    })
    curr_nose_moves = make_keypoints({
        feat.NOSE: (10, 0, 1),
        feat.LEFT_WRIST: (0, 0, 1),
    })
    curr_wrist_moves = make_keypoints({
        feat.NOSE: (0, 0, 1),
        feat.LEFT_WRIST: (10, 0, 1),
    })
    score_nose = feat.motion_score(prev, curr_nose_moves, person_h=100, min_conf=MIN_CONF, dt=1)
    score_wrist = feat.motion_score(prev, curr_wrist_moves, person_h=100, min_conf=MIN_CONF, dt=1)
    assert score_wrist > score_nose


def test_motion_score_divides_by_dt():
    """Fix (throughput scaling): motion_score is a rate, not a raw
    per-tick displacement -- the same displacement over a longer dt
    must read as proportionally less motion."""
    prev = make_keypoints({feat.NOSE: (0, 0, 1)})
    curr = make_keypoints({feat.NOSE: (10, 0, 1)})
    fast = feat.motion_score(prev, curr, person_h=100, min_conf=MIN_CONF, dt=0.05)
    slow = feat.motion_score(prev, curr, person_h=100, min_conf=MIN_CONF, dt=0.5)
    assert fast == pytest.approx(slow * 10)


def test_motion_score_zero_for_nonpositive_dt():
    prev = make_keypoints({feat.NOSE: (0, 0, 1)})
    curr = make_keypoints({feat.NOSE: (10, 0, 1)})
    assert feat.motion_score(prev, curr, person_h=100, min_conf=MIN_CONF, dt=0) == 0


def test_body_visible_true_with_shoulders():
    kps = make_keypoints({feat.LEFT_SHOULDER: (0, 0, 1), feat.RIGHT_SHOULDER: (10, 0, 1)})
    assert feat.body_visible(kps, min_conf=MIN_CONF) is True


def test_body_visible_false_when_nothing_confident():
    kps = make_keypoints({})
    assert feat.body_visible(kps, min_conf=MIN_CONF) is False


def test_head_conf_low_true_when_no_head_keypoint_visible():
    kps = make_keypoints({feat.LEFT_SHOULDER: (0, 0, 1), feat.RIGHT_SHOULDER: (10, 0, 1)})
    assert feat.head_conf_low(kps, min_conf=MIN_CONF) is True


def test_head_conf_low_false_when_nose_visible():
    kps = make_keypoints({feat.NOSE: (5, 5, 1)})
    assert feat.head_conf_low(kps, min_conf=MIN_CONF) is False


def test_sliding_window_smoother_drops_old_samples():
    smoother = feat.SlidingWindowSmoother(window_sec=5)
    smoother.push(0, 10)
    mean_at_1 = smoother.push(1, 0)
    assert mean_at_1 == pytest.approx(5)
    mean_at_100 = smoother.push(100, 2)
    assert mean_at_100 == pytest.approx(2)


# --------------------------------------------------------------------
# Fix 1.2: bbox geometry wake backstop
# --------------------------------------------------------------------

def test_bbox_tracker_no_wake_when_static():
    tracker = feat.BboxMotionTracker(window_sec=2, height_wake_frac=0.25, center_wake_frac=0.30)
    bbox = (100, 100, 150, 300)  # height 200
    assert tracker.push(0, bbox, person_h=200) is False
    assert tracker.push(1, bbox, person_h=200) is False


def test_bbox_tracker_wakes_on_height_change():
    tracker = feat.BboxMotionTracker(window_sec=2, height_wake_frac=0.25, center_wake_frac=0.30)
    tracker.push(0, (100, 100, 150, 300), person_h=200)  # height 200
    # standing up: bbox height roughly doubles
    woke = tracker.push(1, (100, 50, 150, 450), person_h=200)  # height 400
    assert woke is True


def test_bbox_tracker_wakes_on_center_move():
    tracker = feat.BboxMotionTracker(window_sec=2, height_wake_frac=0.25, center_wake_frac=0.30)
    tracker.push(0, (100, 100, 150, 300), person_h=200)
    # center moves ~200px, well over 30% of person_h=200
    woke = tracker.push(1, (300, 100, 350, 300), person_h=200)
    assert woke is True


def test_bbox_tracker_ignores_small_jitter():
    tracker = feat.BboxMotionTracker(window_sec=2, height_wake_frac=0.25, center_wake_frac=0.30)
    tracker.push(0, (100, 100, 150, 300), person_h=200)
    woke = tracker.push(1, (102, 101, 152, 299), person_h=200)
    assert woke is False


def test_bbox_tracker_reset_clears_stale_reference():
    """Track-continuity fix: after a detection gap, the next bbox must
    not be compared against a pre-gap reference -- that comparison can
    look like a big jump purely from elapsed time, not real movement."""
    tracker = feat.BboxMotionTracker(window_sec=2, height_wake_frac=0.25, center_wake_frac=0.30)
    tracker.push(0, (100, 100, 150, 300), person_h=200)
    tracker.reset()
    # same bbox reappears after a gap -- with history cleared this is
    # treated as the first-ever sample, so it can't possibly "wake"
    woke = tracker.push(5, (100, 100, 150, 300), person_h=200)
    assert woke is False
