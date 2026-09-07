"""guard_monitor.config -- the single source of truth for every tunable.

Hard rule for this project: no threshold, dwell time, or magic number
may be written as a bare literal anywhere under guard_monitor/cv/,
guard_monitor/state_machine/, or guard_monitor/alert/. Every value used
by those packages must be read off a GuardMonitorConfig instance, and
every field on GuardMonitorConfig must be reachable from a CLI flag
(see build_arg_parser / config_from_args below).

Fields marked "(derived)" are not independently overridable -- they are
computed from another field so that unit conversion never needs a
literal inside the scanned packages.

wake_thresh and woke_up_sustain_sec from the original spec were
replaced outright (not aliased) by wake_long_thresh and
woke_up_hold_sec per the field-verification fix pass -- the single
long-window threshold wasn't reliable enough on its own (see
state_machine/guard.py's wake-condition docstring for why).

Throughput fix (scaling to many parallel streams): STILL_THRESH,
WAKE_LONG_THRESH, WAKE_SHORT_THRESH, and CALIB_MAX_MOTION were
rescaled x20 from their original spec values (0.05, 0.10, 0.15, 0.15)
to (1.0, 2.0, 3.0, 3.0). This is not a re-tuning -- it's a unit change.
cv/features.py's motion_score() used to return raw per-tick
displacement with no time component, implicitly assuming a fixed
~20fps processing rate; it now divides by dt, returning a per-second
rate, so the same real motion reads the same regardless of whether
INFER_FPS samples at the source's native rate or a reduced one. The
x20 factor exactly cancels out at the ~20fps (dt~=0.05s) the two
acceptance-test clips were captured and verified at, so verified
behavior at that frame rate is unchanged; only the ability to run
lower INFER_FPS without the numbers meaning something different is new.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, fields, MISSING


@dataclass
class GuardMonitorConfig:
    # --- sleeping-guard detection tunables (spec defaults) -----------
    # still_thresh is in motion_score's per-second units (see module
    # docstring's "Throughput fix" note) -- x20 of the original 0.05
    # spec value, which cancels out exactly at ~20fps.
    still_thresh: float = 1.0
    head_tilt_fwd: float = 25.0
    spine_lean_fwd: float = 35.0
    hand_head_dist: float = 0.45
    kp_min_conf: float = 0.35
    sliding_window: float = 5.0
    dwell_to_suspect: float = 30.0
    dwell_to_alert: float = 90.0
    track_min_age: float = 3.0
    calibration_sec: float = 300.0
    vacant_dwell_min: float = 5.0
    starts_of_shift_quiet_min: float = 10.0

    # --- supplementary tunables ---------------------------------------
    # The spec's state table names sustained durations ("sustained 60
    # sec", "sustained 5 sec") that are not among the 13 named constants
    # above. They are still thresholds that drive alerting, so per the
    # no-magic-number rule they get their own tunables and flags rather
    # than being written as literals inside state_machine/guard.py.
    still_sustain_sec: float = 60.0
    wake_sustain_sec: float = 5.0

    # 10-second alert clip window (alert/clip_saver.py) and the RTSP
    # reconnect backoff (runner.py) are likewise operator-tunable
    # rather than hardcoded.
    clip_duration_sec: float = 10.0
    reconnect_interval_sec: float = 5.0
    webhook_timeout_sec: float = 5.0

    # Shift clock the quiet window is measured from, "HH:MM" 24h.
    quiet_window_start: str = "00:00"

    # --- Fix 1: motion must be checked in every state -----------------
    # A single long-window threshold (the old WAKE_THRESH) can be diluted
    # by SLIDING_WINDOW averaging on a brief-but-real wake event (a guard
    # standing up for a few seconds). Three independent, cheaper-to-fool
    # -differently signals replace it; any one of them is a wake:
    wake_long_thresh: float = 2.0           # per-second units; replaces WAKE_THRESH (was 0.18)
    wake_short_thresh: float = 3.0          # per-second units; immediate, no sustain
    wake_short_window_sec: float = 0.5
    bbox_height_wake_frac: float = 0.25     # bbox height self-relative change
    bbox_center_wake_frac: float = 0.30     # bbox center move / person_h
    bbox_wake_window_sec: float = 2.0
    woke_up_hold_sec: float = 3.0           # replaces WOKE_UP_SUSTAIN_SEC (was 10)

    # --- Fix 2: alert must not reset the evidence ----------------------
    realert_cooldown_sec: float = 180.0
    redoze_min_zero_sec: float = 15.0
    evidence_decay_active: float = 1.5      # sleep_evidence_seconds -= this * dt in ACTIVE
    evidence_decay_slow: float = 0.1        # decay elsewhere when no signal present

    # --- Fix 3: calibration must reject sleeping postures ---------------
    default_baseline_spine: float = 10.0
    calib_max_head_tilt: float = 15.0
    calib_max_spine_lean: float = 25.0
    calib_max_motion: float = 3.0           # per-second units, see still_thresh
    calib_min_clean_sec: float = 30.0
    rebaseline_min_active_sec: float = 60.0
    rebaseline_max_head_tilt: float = 12.0
    rebaseline_max_shift_deg: float = 25.0
    rebaseline_blend_old: float = 0.7       # new_baseline = old*this + observed*(1-this)

    # --- Fix 5: long-stillness-with-no-signal VLM second opinion --------
    long_still_sec: float = 180.0
    vlm_recheck_sec: float = 120.0
    vlm_max_calls_per_hour: float = 60.0
    vlm_crop_margin_frac: float = 0.20

    # --- Fix 6: debounce on the four (now five, with VLM) signals -------
    signal_debounce_true_sec: float = 8.0
    signal_debounce_false_sec: float = 5.0

    # --- track-continuity fix (post-fix-pack field verification) --------
    # A track is only dropped after this long with no detection at all --
    # bridges the routine sub-second detection gaps real footage produces
    # (motion blur, brief occlusion, a profile pose caught intermittently)
    # without silently wiping accumulated evidence/calibration/debounce
    # state on the very first missed frame.
    track_lost_grace_sec: float = 2.0
    # If a track's detection gap exceeds this (shorter than the grace
    # period above), the next frame's motion/bbox-delta comparisons are
    # reset rather than computed across the gap -- otherwise "how far did
    # the bbox move" over an unknown multi-frame gap can look like a
    # large single-frame motion spike and trigger a false wake right at
    # reconnection.
    motion_gap_reset_sec: float = 0.5

    # --- throughput fix (scaling to many parallel streams) --------------
    # Sleep-detection dwell times (STILL_SUSTAIN_SEC, DWELL_TO_SUSPECT,
    # etc.) operate on tens-of-seconds timescales, so running pose
    # inference on every camera frame is wasted compute -- 0 (default)
    # processes every frame (unchanged behavior); a positive value caps
    # inference to roughly that many times per second, skipping the
    # detector (and all per-track processing) on frames in between.
    # Recorded clips and the annotated video still write every source
    # frame at full quality -- only the inference cadence drops.
    # If set below ~1 / MOTION_GAP_RESET_SEC, raise
    # motion_gap_reset_sec too, or every sampled tick will look like a
    # reconnection and motion will never accumulate.
    infer_fps: float = 0

    @property
    def vacant_dwell_sec(self) -> float:
        """(derived) VACANT_DWELL_MIN expressed in seconds."""
        return self.vacant_dwell_min * 60.0

    @property
    def quiet_window_sec(self) -> float:
        """(derived) STARTS_OF_SHIFT_QUIET_MIN expressed in seconds."""
        return self.starts_of_shift_quiet_min * 60.0


def _flag_name(field_name: str) -> str:
    return "--" + field_name.replace("_", "-")


def add_config_args(parser: argparse.ArgumentParser) -> None:
    """Register one CLI flag per GuardMonitorConfig field."""
    defaults = GuardMonitorConfig()
    group = parser.add_argument_group("tunables (see guard_monitor/config.py)")
    for f in fields(GuardMonitorConfig):
        default = getattr(defaults, f.name)
        arg_type = str if isinstance(default, str) else float
        group.add_argument(
            _flag_name(f.name),
            type=arg_type,
            default=default,
            dest=f.name,
            help=f"default={default}",
        )


def config_from_args(args: argparse.Namespace) -> GuardMonitorConfig:
    kwargs = {}
    for f in fields(GuardMonitorConfig):
        if f.default is not MISSING and hasattr(args, f.name):
            kwargs[f.name] = getattr(args, f.name)
    return GuardMonitorConfig(**kwargs)
