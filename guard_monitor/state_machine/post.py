"""Pure per-gate vacant-post state machine.

Same purity constraint as guard.py: stdlib only, no OpenCV/Ultralytics.

    OCCUPIED -> EMPTY         no tracked id inside the ROI this tick
    EMPTY    -> OCCUPIED      a tracked id is inside the ROI
    EMPTY    -> VACANT_ALERT  empty sustained for VACANT_DWELL_MIN
                               (VACANT_DWELL_SEC, derived in config.py)

VACANT_ALERT has no rule-driven outgoing transition; the caller
dispatches the alert and calls acknowledge_alert() once handled.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class PostState(str, Enum):
    UNKNOWN = "UNKNOWN"
    OCCUPIED = "OCCUPIED"
    EMPTY = "EMPTY"
    VACANT_ALERT = "VACANT_ALERT"


@dataclass
class PostTick:
    ts: float
    occupied: bool  # True if any tracked id currently sits inside the ROI
    quiet_window: bool


@dataclass
class PostResult:
    state: PostState
    alert_fired: bool


class PostStateMachine:
    def __init__(self, config):
        self.config = config
        self.state = PostState.UNKNOWN
        self._empty_since: Optional[float] = None

    def acknowledge_alert(self) -> None:
        if self.state is PostState.VACANT_ALERT:
            self.state = PostState.EMPTY
        self._empty_since = None

    def tick(self, t: PostTick) -> PostResult:
        if t.occupied:
            self._empty_since = None
            self.state = PostState.OCCUPIED
            return PostResult(self.state, False)

        if self.state is not PostState.VACANT_ALERT:
            self.state = PostState.EMPTY
        if self._empty_since is None:
            self._empty_since = t.ts
            return PostResult(self.state, False)

        elapsed = t.ts - self._empty_since
        if self.state is PostState.EMPTY and elapsed >= self.config.vacant_dwell_sec:
            if not t.quiet_window:
                self.state = PostState.VACANT_ALERT
                return PostResult(self.state, True)
        return PostResult(self.state, False)
