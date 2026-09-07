"""Debounce filter for boolean posture signals (Fix 6).

A raw signal (e.g. head_tilt > HEAD_TILT_FWD) can flicker frame to
frame near the threshold. DebouncedFlag only flips its reported state
after the raw value has held steadily different for a hold period --
longer to assert True than to clear back to False, since a false
positive costs an alert but a false negative just costs a little
delay.

Pure stdlib, no dependency on numpy/cv2/ultralytics.
"""

from __future__ import annotations

from typing import Optional


class DebouncedFlag:
    def __init__(self, true_hold_sec: float, false_hold_sec: float):
        self.true_hold_sec = true_hold_sec
        self.false_hold_sec = false_hold_sec
        self.state: bool = False
        self._pending_since: Optional[float] = None

    def update(self, ts: float, raw_value: bool) -> bool:
        if raw_value == self.state:
            self._pending_since = None
            return self.state
        if self._pending_since is None:
            self._pending_since = ts
        hold = self.true_hold_sec if raw_value else self.false_hold_sec
        if ts - self._pending_since >= hold:
            self.state = raw_value
            self._pending_since = None
        return self.state

    def reset(self) -> None:
        self.state = False
        self._pending_since = None
