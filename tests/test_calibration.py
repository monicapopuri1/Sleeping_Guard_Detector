"""Spins the camera-baseline calibration phase through a synthetic
stream and confirms the recorded baseline matches the median upright
spine_lean reading.

Fix 3 (field verification): a video that starts with the guard already
asleep must not let that sleeping posture pollute the baseline. These
tests cover the gating (only "plausible upright" frames count), the
DEFAULT_BASELINE_SPINE fallback when too few clean frames accumulate,
and the slow gated re-baseline.
"""

import numpy as np
import pytest

from guard_monitor.cv import features as feat

CALIBRATION_SEC = 10
DEFAULT_BASELINE = 10.0
MAX_HEAD_TILT = 15.0
MAX_SPINE_LEAN = 25.0
MAX_MOTION = 0.15
MIN_CLEAN_SEC = 5.0


def make_calibrator(**overrides):
    kwargs = dict(
        calibration_sec=CALIBRATION_SEC,
        default_baseline=DEFAULT_BASELINE,
        max_head_tilt=MAX_HEAD_TILT,
        max_spine_lean=MAX_SPINE_LEAN,
        max_motion=MAX_MOTION,
        min_clean_sec=MIN_CLEAN_SEC,
    )
    kwargs.update(overrides)
    return feat.BaselineCalibrator(**kwargs)


def test_calibrator_active_only_within_calibration_window():
    cal = make_calibrator()
    assert cal.active(0) is True
    assert cal.active(5) is True
    assert cal.active(10) is False


def test_calibrator_baseline_is_median_of_clean_upright_samples():
    cal = make_calibrator()
    for ts in range(0, 10):
        cal.observe(ts, head_visible=True, head_tilt_raw=2, spine_lean_raw=7, motion_long=0)
    baseline = cal.finalize_if_ready(10)
    assert baseline == pytest.approx(7)
    assert cal.used_default is False


def test_calibrator_rejects_frames_that_look_asleep():
    """The Fix 3 regression: a sleeping-posture frame (large spine_lean)
    during calibration must not be averaged into the baseline."""
    cal = make_calibrator()
    for ts in range(0, 5):
        # looks asleep: spine_lean way past the gate
        cal.observe(ts, head_visible=True, head_tilt_raw=2, spine_lean_raw=60, motion_long=0)
    for ts in range(5, 10):
        cal.observe(ts, head_visible=True, head_tilt_raw=2, spine_lean_raw=8, motion_long=0)
    baseline = cal.finalize_if_ready(10)
    # only the 5 clean (upright-looking) seconds count, median of those is 8
    assert baseline == pytest.approx(8)


def test_calibrator_uses_default_when_too_few_clean_seconds():
    cal = make_calibrator()
    # guard looks asleep for the whole calibration window -- no clean samples
    for ts in range(0, 10):
        cal.observe(ts, head_visible=True, head_tilt_raw=2, spine_lean_raw=60, motion_long=0)
    baseline = cal.finalize_if_ready(10)
    assert baseline == pytest.approx(DEFAULT_BASELINE)
    assert cal.used_default is True


def test_calibrator_rejects_high_motion_frames():
    cal = make_calibrator()
    for ts in range(0, 10):
        cal.observe(ts, head_visible=True, head_tilt_raw=2, spine_lean_raw=8, motion_long=MAX_MOTION + 1)
    baseline = cal.finalize_if_ready(10)
    assert cal.used_default is True


def test_calibrator_rejects_head_not_visible():
    cal = make_calibrator()
    for ts in range(0, 10):
        cal.observe(ts, head_visible=False, head_tilt_raw=2, spine_lean_raw=8, motion_long=0)
    baseline = cal.finalize_if_ready(10)
    assert cal.used_default is True


def test_calibrator_does_not_finalize_before_window_ends():
    cal = make_calibrator()
    cal.observe(0, head_visible=True, head_tilt_raw=2, spine_lean_raw=8, motion_long=0)
    baseline = cal.finalize_if_ready(5)
    assert baseline == pytest.approx(0)  # default float, window not over yet
    assert cal.used_default is False  # not decided yet either


def test_calibrator_baseline_then_used_to_zero_out_bias():
    cal = make_calibrator()
    for ts in range(0, 10):
        cal.observe(ts, head_visible=True, head_tilt_raw=2, spine_lean_raw=12, motion_long=0)
    baseline = cal.finalize_if_ready(10)

    kps = np.zeros((17, 3))
    kps[feat.LEFT_SHOULDER] = [0, -12, 1]
    kps[feat.RIGHT_SHOULDER] = [0, -12, 1]
    kps[feat.LEFT_HIP] = [0, 0, 1]
    kps[feat.RIGHT_HIP] = [0, 0, 1]
    # this frame's raw spine_lean is 0 (perfectly vertical vector), so
    # after subtracting a nonzero baseline it should read as *negative*
    # baseline -- the meaningful check is that subtraction is applied at
    # all and matches the calibrated value exactly.
    corrected, _ = feat.spine_lean_deg(kps, baseline_deg=baseline, min_conf=0.35)
    assert corrected == pytest.approx(-baseline)


# --------------------------------------------------------------------
# Fix 3.3: silent re-baselining
# --------------------------------------------------------------------

def test_rebaseline_does_nothing_before_calibration_finalized():
    cal = make_calibrator()
    cal.maybe_rebaseline(
        ts=0, is_active=True, head_tilt_raw=0, spine_lean_raw=20,
        min_active_sec=1, max_head_tilt=12, max_shift_deg=25, blend_old=0.7,
    )
    assert cal.baseline_deg == pytest.approx(0)


def test_rebaseline_blends_after_sustained_active_low_tilt():
    cal = make_calibrator()
    for ts in range(0, 10):
        cal.observe(ts, head_visible=True, head_tilt_raw=2, spine_lean_raw=10, motion_long=0)
    cal.finalize_if_ready(10)
    assert cal.baseline_deg == pytest.approx(10)

    # guard sits with a slightly different upright spine_lean (14) while
    # ACTIVE and low head_tilt, sustained for min_active_sec=5
    for ts in range(10, 16):
        cal.maybe_rebaseline(
            ts=ts, is_active=True, head_tilt_raw=2, spine_lean_raw=14,
            min_active_sec=5, max_head_tilt=12, max_shift_deg=25, blend_old=0.7,
        )
    # 0.7*10 + 0.3*14 = 11.2
    assert cal.baseline_deg == pytest.approx(11.2)


def test_rebaseline_skipped_when_shift_too_large():
    cal = make_calibrator()
    for ts in range(0, 10):
        cal.observe(ts, head_visible=True, head_tilt_raw=2, spine_lean_raw=10, motion_long=0)
    cal.finalize_if_ready(10)

    for ts in range(10, 16):
        cal.maybe_rebaseline(
            ts=ts, is_active=True, head_tilt_raw=2, spine_lean_raw=80,
            min_active_sec=5, max_head_tilt=12, max_shift_deg=25, blend_old=0.7,
        )
    assert cal.baseline_deg == pytest.approx(10)


def test_rebaseline_resets_when_not_active_or_high_tilt():
    cal = make_calibrator()
    for ts in range(0, 10):
        cal.observe(ts, head_visible=True, head_tilt_raw=2, spine_lean_raw=10, motion_long=0)
    cal.finalize_if_ready(10)

    cal.maybe_rebaseline(ts=10, is_active=True, head_tilt_raw=2, spine_lean_raw=14, min_active_sec=5, max_head_tilt=12, max_shift_deg=25, blend_old=0.7)
    cal.maybe_rebaseline(ts=11, is_active=False, head_tilt_raw=2, spine_lean_raw=14, min_active_sec=5, max_head_tilt=12, max_shift_deg=25, blend_old=0.7)
    # active-low-tilt streak was interrupted; needs another full
    # min_active_sec from scratch, so no blend has happened yet
    cal.maybe_rebaseline(ts=13, is_active=True, head_tilt_raw=2, spine_lean_raw=14, min_active_sec=5, max_head_tilt=12, max_shift_deg=25, blend_old=0.7)
    assert cal.baseline_deg == pytest.approx(10)
