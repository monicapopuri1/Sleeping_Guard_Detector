"""Track-continuity fix: should_drop_track is a pure decision function,
no cv2/ultralytics involved, so it's directly unit-testable."""

import pytest

from guard_monitor.cv.tracker import should_drop_track


def test_not_dropped_immediately_after_a_missed_frame():
    # last seen 0.3s ago, grace period is 2s -- routine gap, keep it
    assert should_drop_track(ts=10.3, last_seen_ts=10.0, grace_sec=2) is False


def test_dropped_once_grace_period_elapses():
    assert should_drop_track(ts=12.0, last_seen_ts=10.0, grace_sec=2) is True


def test_boundary_is_inclusive():
    assert should_drop_track(ts=12.0, last_seen_ts=10.0, grace_sec=2) is True
    assert should_drop_track(ts=11.999, last_seen_ts=10.0, grace_sec=2) is False


@pytest.mark.parametrize("grace_sec", [0, 0.5, 1, 5])
def test_zero_gap_never_drops(grace_sec):
    assert should_drop_track(ts=10.0, last_seen_ts=10.0, grace_sec=grace_sec) is (grace_sec == 0)
