"""Pure per-track sleeping-guard state machine.

No dependency on OpenCV, Ultralytics, or any I/O -- this module only
imports the stdlib. That is intentional: it must be unit-testable with
synthetic ticks and importable with no GPU / no camera driver present.

State table, revised after field verification against two 5-minute
CCTV clips (see the fix rationale in each handler's docstring for what
broke and why):

    ABSENT            -> ACTIVE            when a track appears
    ACTIVE            -> STILL             motion_score < STILL_THRESH,
                                            sustained STILL_SUSTAIN_SEC,
                                            track older than TRACK_MIN_AGE
    STILL             -> SLEEPING_SUSPECT  any debounced suspect reason
                                            sustained DWELL_TO_SUSPECT,
                                            and not inside a post-wake
                                            "prove it" window
    SLEEPING_SUSPECT  -> ALERT             sleep_evidence_seconds >=
                                            DWELL_TO_ALERT
    ALERT             -> POST_ALERT        immediately (one-tick pulse;
                                            the runner dispatches on the
                                            alert_fired=True tick)
    POST_ALERT        -> POST_ALERT        re-fires alert_fired=True
                                            every REALERT_COOLDOWN_SEC
                                            for as long as sleep persists
    WOKE_UP           -> ACTIVE            after WOKE_UP_HOLD_SEC

    (ANY state except ABSENT) -> WOKE_UP   wake condition holds -- see
                                            _wake_condition. This is
                                            evaluated before the
                                            per-state logic above, so it
                                            preempts every other
                                            transition including out of
                                            ALERT/POST_ALERT.

Track loss and manual acknowledgement are external lifecycle events
(like "a track appears"), handled by on_track_lost() / acknowledge_alert().
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from guard_monitor.state_machine.debounce import DebouncedFlag


class GuardState(str, Enum):
    ABSENT = "ABSENT"
    ACTIVE = "ACTIVE"
    STILL = "STILL"
    SLEEPING_SUSPECT = "SLEEPING_SUSPECT"
    ALERT = "ALERT"
    POST_ALERT = "POST_ALERT"
    WOKE_UP = "WOKE_UP"


@dataclass
class GuardTick:
    """One frame's worth of size-free features for a single track."""

    ts: float  # seconds, monotonically increasing per track
    motion_score: float  # long-window (SLIDING_WINDOW) smoothed, joint-weighted
    motion_short: float  # short-window (WAKE_SHORT_WINDOW_SEC) smoothed
    bbox_wake: bool  # bbox height/center geometry backstop, computed upstream
    head_tilt: float
    spine_lean: float
    hand_at_head: bool
    head_conf_low: bool  # head keypoints below KP_MIN_CONF
    body_visible: bool
    vlm_asleep: bool  # cached VLM second-opinion verdict, default False
    track_age: float  # seconds since this track id first appeared
    quiet_window: bool  # True while inside the start-of-shift quiet window


@dataclass
class GuardResult:
    state: GuardState
    alert_fired: bool  # True on every tick the runner should dispatch an alert
    reasons: tuple  # which suspect reason(s) most recently triggered


_SUSPECT_REASONS = ("head_tilt", "spine_lean", "hand_at_head", "head_conf_low", "vlm_asleep")
_DEBOUNCED_REASONS = ("head_tilt", "spine_lean", "hand_at_head", "head_conf_low")


class GuardStateMachine:
    def __init__(self, config):
        self.config = config
        self.state = GuardState.ABSENT
        self._last_ts: Optional[float] = None
        self._low_motion_since: Optional[float] = None
        self._long_wake_since: Optional[float] = None
        self._suspect_cond_since = {k: None for k in _SUSPECT_REASONS}
        self._suspect_triggered_reasons: tuple = ()
        self._woke_up_since: Optional[float] = None
        self._redoze_block_until: Optional[float] = None
        self._last_alert_ts: Optional[float] = None
        self.sleep_evidence_seconds: float = 0
        self._debounced = {
            name: DebouncedFlag(config.signal_debounce_true_sec, config.signal_debounce_false_sec)
            for name in _DEBOUNCED_REASONS
        }

    def on_track_lost(self) -> None:
        """External lifecycle event: the tracker dropped this id."""
        self.__init__(self.config)

    def acknowledge_alert(self) -> None:
        """Explicit manual reset back to ACTIVE. Not called by the normal
        alert path any more -- POST_ALERT self-manages via wake detection
        and the re-alert cooldown (Fix 2). Kept for tests / operator
        override."""
        if self.state in (GuardState.ALERT, GuardState.POST_ALERT):
            self.state = GuardState.ACTIVE
        self.sleep_evidence_seconds = 0
        self._low_motion_since = None
        self._suspect_cond_since = {k: None for k in _SUSPECT_REASONS}
        for flag in self._debounced.values():
            flag.reset()

    def _wake_condition(self, t: GuardTick) -> bool:
        """Any ONE of three independent signals counts as a wake, because
        each fails differently: keypoint-jitter noise can fake a motion
        score but not a doubling bbox height; a brief-but-real stand-up
        can be smoothed away by the long sliding window but not by the
        short one; a genuinely slow, deliberate wake might not clear the
        short-window bar but will clear the long one if sustained."""
        if t.bbox_wake:
            return True
        if t.motion_short > self.config.wake_short_thresh:
            return True
        if t.motion_score > self.config.wake_long_thresh:
            if self._long_wake_since is None:
                self._long_wake_since = t.ts
            if t.ts - self._long_wake_since >= self.config.wake_sustain_sec:
                return True
        else:
            self._long_wake_since = None
        return False

    def _enter_woke_up(self, ts: float) -> None:
        self.state = GuardState.WOKE_UP
        self._woke_up_since = ts
        self._long_wake_since = None
        self._low_motion_since = None
        self._suspect_cond_since = {k: None for k in _SUSPECT_REASONS}
        self._last_alert_ts = None
        # Fix 2 item 5: WOKE_UP is the one transition allowed to hard-reset
        # the evidence counter; everywhere else it only decays.
        self.sleep_evidence_seconds = 0
        for flag in self._debounced.values():
            flag.reset()

    def tick(self, t: GuardTick) -> GuardResult:
        if self.state is GuardState.ABSENT:
            self.state = GuardState.ACTIVE
            self._last_ts = t.ts
            return GuardResult(self.state, False, ())

        dt = max(0, t.ts - self._last_ts) if self._last_ts is not None else 0
        self._last_ts = t.ts

        # Fix 1: evaluated before per-state logic, for every state except
        # ABSENT/WOKE_UP (WOKE_UP already *is* the wake state -- see its
        # own handler for the hold-then-ACTIVE timer).
        if self.state is not GuardState.WOKE_UP and self._wake_condition(t):
            self._enter_woke_up(t.ts)
            return GuardResult(self.state, False, self._suspect_triggered_reasons)

        if self.state is GuardState.ACTIVE:
            return self._tick_active(t, dt)
        if self.state is GuardState.STILL:
            return self._tick_still(t, dt)
        if self.state is GuardState.SLEEPING_SUSPECT:
            return self._tick_sleeping_suspect(t, dt)
        if self.state is GuardState.ALERT:
            return self._tick_alert(t)
        if self.state is GuardState.POST_ALERT:
            return self._tick_post_alert(t)
        return self._tick_woke_up(t)

    def _tick_active(self, t: GuardTick, dt: float) -> GuardResult:
        # Fix 2 item 5: fast decay while genuinely active.
        self.sleep_evidence_seconds = max(0, self.sleep_evidence_seconds - self.config.evidence_decay_active * dt)

        eligible = t.track_age >= self.config.track_min_age
        if eligible and t.motion_score < self.config.still_thresh:
            if self._low_motion_since is None:
                self._low_motion_since = t.ts
            elif t.ts - self._low_motion_since >= self.config.still_sustain_sec:
                self.state = GuardState.STILL
                self._low_motion_since = None
        else:
            self._low_motion_since = None
        return GuardResult(self.state, False, ())

    def _tick_still(self, t: GuardTick, dt: float) -> GuardResult:
        raw = {
            "head_tilt": t.head_tilt > self.config.head_tilt_fwd,
            "spine_lean": t.spine_lean > self.config.spine_lean_fwd,
            "hand_at_head": t.hand_at_head,
            "head_conf_low": t.head_conf_low and t.body_visible,
        }
        debounced = {name: self._debounced[name].update(t.ts, val) for name, val in raw.items()}
        # vlm_asleep is already stable-for-a-window by construction (the
        # runner only refreshes it every VLM_RECHECK_SEC), so it doesn't
        # need a second debounce layer on top.
        debounced["vlm_asleep"] = t.vlm_asleep

        if any(debounced.values()):
            self.sleep_evidence_seconds += dt
        else:
            self.sleep_evidence_seconds = max(0, self.sleep_evidence_seconds - self.config.evidence_decay_slow * dt)

        triggered = []
        for name, val in debounced.items():
            if val:
                if self._suspect_cond_since[name] is None:
                    self._suspect_cond_since[name] = t.ts
                elif t.ts - self._suspect_cond_since[name] >= self.config.dwell_to_suspect:
                    triggered.append(name)
            else:
                self._suspect_cond_since[name] = None

        blocked = self._redoze_block_until is not None and t.ts < self._redoze_block_until
        if triggered and not blocked:
            self.state = GuardState.SLEEPING_SUSPECT
            self._suspect_triggered_reasons = tuple(triggered)

        return GuardResult(self.state, False, tuple(triggered))

    def _tick_sleeping_suspect(self, t: GuardTick, dt: float) -> GuardResult:
        # Fix 2 items 1 and 5: evidence keeps accumulating (never a fresh
        # "since" timestamp that a reset could zero out from under us).
        self.sleep_evidence_seconds += dt

        if self.sleep_evidence_seconds >= self.config.dwell_to_alert:
            suppressed = t.quiet_window or t.track_age < self.config.track_min_age
            if not suppressed:
                self.state = GuardState.ALERT
                self._last_alert_ts = t.ts
                return GuardResult(self.state, True, self._suspect_triggered_reasons)
        return GuardResult(self.state, False, self._suspect_triggered_reasons)

    def _tick_alert(self, t: GuardTick) -> GuardResult:
        # Reached only when _wake_condition was False this tick. ALERT
        # is a one-tick pulse: the caller dispatches on the tick that
        # returned alert_fired=True (from _tick_sleeping_suspect); this
        # handler's only job is to hand off to POST_ALERT so evidence
        # keeps being tracked instead of the machine going silent, which
        # was the original bug (ALERT had no wake check at all).
        self.state = GuardState.POST_ALERT
        return GuardResult(self.state, False, self._suspect_triggered_reasons)

    def _tick_post_alert(self, t: GuardTick) -> GuardResult:
        # Fix 2 item 2: sleep_evidence_seconds is frozen here -- no increment,
        # no decay. We already have our evidence; we're only waiting on
        # the wake check (run before this, in tick()) or the re-alert
        # cooldown below.
        if t.ts - self._last_alert_ts >= self.config.realert_cooldown_sec:
            suppressed = t.quiet_window or t.track_age < self.config.track_min_age
            if not suppressed:
                self._last_alert_ts = t.ts
                return GuardResult(self.state, True, self._suspect_triggered_reasons)
        return GuardResult(self.state, False, self._suspect_triggered_reasons)

    def _tick_woke_up(self, t: GuardTick) -> GuardResult:
        if t.ts - self._woke_up_since >= self.config.woke_up_hold_sec:
            self.state = GuardState.ACTIVE
            # Fix 2 item 4: guard must prove REDOZE_MIN_ZERO_SEC of real
            # stillness before SLEEPING_SUSPECT can be re-entered --
            # blocks an immediate redoze read as a continuation.
            self._redoze_block_until = t.ts + self.config.redoze_min_zero_sec
            self._low_motion_since = None
            self._suspect_cond_since = {k: None for k in _SUSPECT_REASONS}
            for flag in self._debounced.values():
                flag.reset()
        return GuardResult(self.state, False, ())
