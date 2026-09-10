"""Pure state-machine transition tests -- synthetic ticks, no CV/IO.

Covers every state and every transition in the (revised) spec: the
quiet-window alert guard, the dwell-to-alert evidence accumulator, the
Fix-1 wake condition from every non-ABSENT state (including the
formerly-frozen ALERT state), the Fix-2 POST_ALERT re-alert cooldown
and redoze block, and debounced suspect signals (Fix 6).
"""

from dataclasses import replace

import pytest

from guard_monitor.config import GuardMonitorConfig
from guard_monitor.state_machine.guard import GuardState, GuardStateMachine, GuardTick


def make_tick(ts, **overrides):
    defaults = dict(
        motion_score=0,
        motion_short=0,
        bbox_wake=False,
        head_tilt=0,
        spine_lean=0,
        hand_at_head=False,
        head_conf_low=False,
        body_visible=True,
        vlm_asleep=False,
        track_age=100,
        quiet_window=False,
    )
    defaults.update(overrides)
    return GuardTick(ts=ts, **defaults)


@pytest.fixture
def config():
    return GuardMonitorConfig()


def drive_to_still(sm, config, start_ts=0):
    """ABSENT -> ACTIVE -> STILL, returns the ts of the last tick."""
    sm.tick(make_tick(start_ts))  # ABSENT -> ACTIVE
    assert sm.state is GuardState.ACTIVE
    ts = start_ts + 1
    sm.tick(make_tick(ts, motion_score=0))
    ts += config.still_sustain_sec
    result = sm.tick(make_tick(ts, motion_score=0))
    assert result.state is GuardState.STILL
    return ts


def drive_to_suspect(sm, config, start_ts=0, reason_kwargs=None):
    """STILL -> SLEEPING_SUSPECT via one debounced reason (default
    head_tilt), accounting for both the debounce hold and the dwell."""
    reason_kwargs = reason_kwargs or {"head_tilt": config.head_tilt_fwd + 1}
    ts = drive_to_still(sm, config, start_ts)
    ts += 1
    sm.tick(make_tick(ts, motion_score=0, **reason_kwargs))
    if "vlm_asleep" not in reason_kwargs:
        ts += config.signal_debounce_true_sec
        sm.tick(make_tick(ts, motion_score=0, **reason_kwargs))
    ts += config.dwell_to_suspect
    result = sm.tick(make_tick(ts, motion_score=0, **reason_kwargs))
    assert result.state is GuardState.SLEEPING_SUSPECT
    return ts


def drive_to_alert(sm, config, start_ts=0):
    ts = drive_to_suspect(sm, config, start_ts)
    ts += config.dwell_to_alert
    result = sm.tick(make_tick(ts, motion_score=0, head_tilt=config.head_tilt_fwd + 1))
    assert result.state is GuardState.ALERT
    assert result.alert_fired is True
    return ts


# --------------------------------------------------------------------
# ABSENT / ACTIVE / STILL
# --------------------------------------------------------------------

def test_absent_to_active(config):
    sm = GuardStateMachine(config)
    assert sm.state is GuardState.ABSENT
    result = sm.tick(make_tick(0))
    assert result.state is GuardState.ACTIVE


def test_active_to_still_requires_sustained_low_motion(config):
    sm = GuardStateMachine(config)
    sm.tick(make_tick(0))
    ts = 1
    sm.tick(make_tick(ts, motion_score=0))
    result = sm.tick(make_tick(ts + config.still_sustain_sec / 2, motion_score=0))
    assert result.state is GuardState.ACTIVE
    result = sm.tick(make_tick(ts + config.still_sustain_sec, motion_score=0))
    assert result.state is GuardState.STILL


def test_active_to_still_blocked_by_track_min_age(config):
    sm = GuardStateMachine(config)
    sm.tick(make_tick(0, track_age=0))
    ts = 1
    sm.tick(make_tick(ts, motion_score=0, track_age=0))
    result = sm.tick(make_tick(ts + config.still_sustain_sec, motion_score=0, track_age=0))
    assert result.state is GuardState.ACTIVE


def test_active_low_motion_interrupted_resets_timer(config):
    sm = GuardStateMachine(config)
    sm.tick(make_tick(0))
    ts = 1
    sm.tick(make_tick(ts, motion_score=0))
    sm.tick(make_tick(ts + config.still_sustain_sec / 2, motion_score=1))
    result = sm.tick(make_tick(ts + config.still_sustain_sec, motion_score=0))
    assert result.state is GuardState.ACTIVE


# --------------------------------------------------------------------
# Fix 1: wake condition, evaluated before per-state logic everywhere
# --------------------------------------------------------------------

def test_wake_via_bbox_backstop_is_immediate_from_still(config):
    sm = GuardStateMachine(config)
    ts = drive_to_still(sm, config)
    result = sm.tick(make_tick(ts + 1, bbox_wake=True))
    assert result.state is GuardState.WOKE_UP


def test_soft_wake_short_burst_is_a_stir_not_instant(config):
    """F7: the two motion-based signals are ambiguous (could be tossing
    in sleep), so unlike bbox_wake they do NOT reset instantly -- they
    must be confirmed for AWAKE_CONFIRM_SEC first."""
    sm = GuardStateMachine(config)
    ts = drive_to_still(sm, config)
    result = sm.tick(make_tick(ts + 1, motion_short=config.wake_short_thresh + 1))
    assert result.state is GuardState.STILL


def test_soft_wake_confirmed_after_awake_confirm_sec(config):
    sm = GuardStateMachine(config)
    ts = drive_to_still(sm, config)
    ts += 1
    sm.tick(make_tick(ts, motion_short=config.wake_short_thresh + 1))
    result = sm.tick(make_tick(ts + config.awake_confirm_sec, motion_short=config.wake_short_thresh + 1))
    assert result.state is GuardState.WOKE_UP


def test_wake_via_long_threshold_requires_sustain_then_confirm(config):
    sm = GuardStateMachine(config)
    ts = drive_to_still(sm, config)
    ts += 1
    sm.tick(make_tick(ts, motion_score=config.wake_long_thresh + 1))
    result = sm.tick(make_tick(ts + config.wake_sustain_sec / 2, motion_score=config.wake_long_thresh + 1))
    assert result.state is GuardState.STILL
    # long-threshold sustain is cleared, but F7's awake-confirm timer
    # only starts counting from when the signal first became "soft
    # wake" true (i.e. once wake_sustain_sec elapses) -- still a stir,
    # not an instant transition, until AWAKE_CONFIRM_SEC passes.
    result = sm.tick(make_tick(ts + config.wake_sustain_sec, motion_score=config.wake_long_thresh + 1))
    assert result.state is GuardState.STILL
    result = sm.tick(make_tick(ts + config.wake_sustain_sec + config.awake_confirm_sec, motion_score=config.wake_long_thresh + 1))
    assert result.state is GuardState.WOKE_UP


# --------------------------------------------------------------------
# F7: stir vs wake-up -- evidence pauses, drains, then fully clears
# --------------------------------------------------------------------

def test_stir_freezes_evidence_without_resetting(config):
    sm = GuardStateMachine(config)
    ts = drive_to_suspect(sm, config)
    evidence_before = sm.sleep_evidence_seconds
    assert evidence_before > 0
    ts += 1
    result = sm.tick(make_tick(ts, motion_short=config.wake_short_thresh + 1))
    assert result.state is GuardState.SLEEPING_SUSPECT
    assert sm.sleep_evidence_seconds == pytest.approx(evidence_before)
    # still frozen partway through the stir window
    result = sm.tick(make_tick(ts + config.stir_max_sec / 2, motion_short=config.wake_short_thresh + 1))
    assert result.state is GuardState.SLEEPING_SUSPECT
    assert sm.sleep_evidence_seconds == pytest.approx(evidence_before)


def test_stir_resolves_and_evidence_resumes_growing(config):
    sm = GuardStateMachine(config)
    ts = drive_to_suspect(sm, config)
    evidence_before = sm.sleep_evidence_seconds
    ts += 1
    sm.tick(make_tick(ts, motion_short=config.wake_short_thresh + 1))
    ts += 1
    # motion drops back below threshold well before stir_max_sec -- the
    # guard settles back down, nothing lost
    result = sm.tick(make_tick(ts, motion_score=0, motion_short=0))
    assert result.state is GuardState.SLEEPING_SUSPECT
    assert sm.sleep_evidence_seconds >= evidence_before


def test_drain_zone_slowly_reduces_evidence(config):
    sm = GuardStateMachine(config)
    ts = drive_to_suspect(sm, config)
    evidence_before = sm.sleep_evidence_seconds
    ts += 1
    stir_start = ts
    sm.tick(make_tick(ts, motion_short=config.wake_short_thresh + 1))
    # into the drain zone (past stir_max_sec, short of awake_confirm_sec)
    drain_ts = stir_start + config.stir_max_sec + 1
    result = sm.tick(make_tick(drain_ts, motion_short=config.wake_short_thresh + 1))
    assert result.state is GuardState.SLEEPING_SUSPECT
    assert sm.sleep_evidence_seconds < evidence_before


def test_awake_confirm_sec_fully_clears_evidence(config):
    sm = GuardStateMachine(config)
    ts = drive_to_suspect(sm, config)
    ts += 1
    stir_start = ts
    sm.tick(make_tick(ts, motion_short=config.wake_short_thresh + 1))
    result = sm.tick(make_tick(stir_start + config.awake_confirm_sec, motion_short=config.wake_short_thresh + 1))
    assert result.state is GuardState.WOKE_UP
    assert sm.sleep_evidence_seconds == 0


def test_episode_closed_reported_after_alert_and_confirmed_wake(config):
    sm = GuardStateMachine(config)
    alert_ts = drive_to_alert(sm, config)
    sm.tick(make_tick(alert_ts + 1, motion_score=0, head_tilt=config.head_tilt_fwd + 1))  # -> POST_ALERT
    wake_start = alert_ts + 2
    sm.tick(make_tick(wake_start, motion_short=config.wake_short_thresh + 1))
    result = sm.tick(make_tick(wake_start + config.awake_confirm_sec, motion_short=config.wake_short_thresh + 1))
    assert result.state is GuardState.WOKE_UP
    assert result.episode_closed is True
    assert result.episode_alert_ts == pytest.approx(alert_ts)
    assert result.episode_wake_started_ts == pytest.approx(wake_start)


def test_episode_not_closed_when_no_alert_had_fired(config):
    """A stir/confirm cycle from plain STILL (never reached ALERT) isn't
    a closed "episode" in the alert-lifecycle sense -- nothing to log."""
    sm = GuardStateMachine(config)
    ts = drive_to_still(sm, config)
    ts += 1
    sm.tick(make_tick(ts, motion_short=config.wake_short_thresh + 1))
    result = sm.tick(make_tick(ts + config.awake_confirm_sec, motion_short=config.wake_short_thresh + 1))
    assert result.state is GuardState.WOKE_UP
    assert result.episode_closed is False


def test_hard_wake_bbox_still_instant_even_with_open_episode(config):
    """The bbox backstop (a real stand-up) is unambiguous and bypasses
    the stir/drain/confirm tiering entirely -- it still closes an open
    episode immediately, same as before F7."""
    sm = GuardStateMachine(config)
    alert_ts = drive_to_alert(sm, config)
    sm.tick(make_tick(alert_ts + 1, motion_score=0, head_tilt=config.head_tilt_fwd + 1))  # -> POST_ALERT
    result = sm.tick(make_tick(alert_ts + 2, bbox_wake=True))
    assert result.state is GuardState.WOKE_UP
    assert result.episode_closed is True
    # Regression check: a hard wake never necessarily ran through the
    # soft-wake duration tracker, so episode_wake_started_ts must not be
    # None (the runner does ts - episode_wake_started_ts unconditionally
    # once episode_closed is True) -- for an instant wake, "started" and
    # "now" are the same tick.
    assert result.episode_wake_started_ts == pytest.approx(alert_ts + 2)
    assert result.episode_closed is True


def test_wake_from_sleeping_suspect(config):
    sm = GuardStateMachine(config)
    ts = drive_to_suspect(sm, config)
    result = sm.tick(make_tick(ts + 1, bbox_wake=True))
    assert result.state is GuardState.WOKE_UP


def test_wake_from_active_bounces_through_woke_up(config):
    sm = GuardStateMachine(config)
    sm.tick(make_tick(0))
    result = sm.tick(make_tick(1, bbox_wake=True))
    assert result.state is GuardState.WOKE_UP


def test_wake_overrides_alert_state_directly(config):
    """This is the exact bug from field verification: a stand-up during
    ALERT/SLEEPING_SUSPECT must interrupt it. Previously ALERT had zero
    wake check and stayed ALERT forever until externally acknowledged.
    A stand-up is a hard wake (bbox), which stays instant under F7."""
    sm = GuardStateMachine(config)
    ts = drive_to_alert(sm, config)
    result = sm.tick(make_tick(ts + 1, bbox_wake=True))
    assert result.state is GuardState.WOKE_UP


def test_wake_from_post_alert(config):
    sm = GuardStateMachine(config)
    ts = drive_to_alert(sm, config)
    ts += 1
    result = sm.tick(make_tick(ts, motion_score=0, head_tilt=config.head_tilt_fwd + 1))
    assert result.state is GuardState.POST_ALERT
    ts += 1
    result = sm.tick(make_tick(ts, bbox_wake=True))
    assert result.state is GuardState.WOKE_UP


# --------------------------------------------------------------------
# STILL -> SLEEPING_SUSPECT, every debounced reason (Fix 6) + VLM (Fix 5)
# --------------------------------------------------------------------

@pytest.mark.parametrize(
    "reason_kwargs",
    [
        {"head_tilt": None},
        {"spine_lean": None},
        {"hand_at_head": None},
        {"head_conf_low": None},
        {"vlm_asleep": None},
    ],
)
def test_still_to_suspect_each_reason(config, reason_kwargs):
    key = next(iter(reason_kwargs))
    if key == "head_tilt":
        kwargs = {"head_tilt": config.head_tilt_fwd + 1}
    elif key == "spine_lean":
        kwargs = {"spine_lean": config.spine_lean_fwd + 1}
    elif key == "hand_at_head":
        kwargs = {"hand_at_head": True}
    elif key == "head_conf_low":
        kwargs = {"head_conf_low": True, "body_visible": True}
    else:
        kwargs = {"vlm_asleep": True}

    sm = GuardStateMachine(config)
    drive_to_suspect(sm, config, reason_kwargs=kwargs)
    assert sm.state is GuardState.SLEEPING_SUSPECT


def test_suspect_reason_must_survive_debounce(config):
    """A reason that flips true for less than SIGNAL_DEBOUNCE_TRUE_SEC
    must not start the dwell-to-suspect clock at all."""
    sm = GuardStateMachine(config)
    ts = drive_to_still(sm, config)
    ts += 1
    sm.tick(make_tick(ts, motion_score=0, head_tilt=config.head_tilt_fwd + 1))
    # condition drops before the debounce hold elapses
    sm.tick(make_tick(ts + config.signal_debounce_true_sec / 2, motion_score=0, head_tilt=0))
    result = sm.tick(make_tick(ts + config.signal_debounce_true_sec + config.dwell_to_suspect, motion_score=0, head_tilt=0))
    assert result.state is GuardState.STILL


def test_suspect_reason_interrupted_after_debounce_resets_dwell(config):
    sm = GuardStateMachine(config)
    ts = drive_to_still(sm, config)
    ts += 1
    true_kw = {"head_tilt": config.head_tilt_fwd + 1}
    false_kw = {"head_tilt": 0}
    sm.tick(make_tick(ts, motion_score=0, **true_kw))
    ts += config.signal_debounce_true_sec
    sm.tick(make_tick(ts, motion_score=0, **true_kw))  # debounced True, dwell timer starts
    ts += 1
    sm.tick(make_tick(ts, motion_score=0, **false_kw))
    ts += config.signal_debounce_false_sec
    sm.tick(make_tick(ts, motion_score=0, **false_kw))  # debounced False, dwell timer cleared
    ts += 1
    sm.tick(make_tick(ts, motion_score=0, **true_kw))
    ts += config.signal_debounce_true_sec
    sm.tick(make_tick(ts, motion_score=0, **true_kw))  # debounced True again, fresh dwell timer
    result = sm.tick(make_tick(ts + config.dwell_to_suspect - 1, motion_score=0, **true_kw))
    assert result.state is GuardState.STILL


# --------------------------------------------------------------------
# SLEEPING_SUSPECT -> ALERT, quiet window
# --------------------------------------------------------------------

def test_suspect_to_alert_after_dwell(config):
    sm = GuardStateMachine(config)
    drive_to_alert(sm, config)
    assert sm.state is GuardState.ALERT


def test_suspect_to_alert_suppressed_during_quiet_window(config):
    sm = GuardStateMachine(config)
    ts = drive_to_suspect(sm, config)
    ts += config.dwell_to_alert
    result = sm.tick(make_tick(ts, motion_score=0, head_tilt=config.head_tilt_fwd + 1, quiet_window=True))
    assert result.state is GuardState.SLEEPING_SUSPECT
    assert result.alert_fired is False

    result = sm.tick(make_tick(ts + 1, motion_score=0, head_tilt=config.head_tilt_fwd + 1, quiet_window=False))
    assert result.state is GuardState.ALERT
    assert result.alert_fired is True


# --------------------------------------------------------------------
# Fix 2: ALERT -> POST_ALERT, evidence frozen, re-alert cooldown
# --------------------------------------------------------------------

def test_alert_transitions_to_post_alert_next_tick(config):
    sm = GuardStateMachine(config)
    ts = drive_to_alert(sm, config)
    result = sm.tick(make_tick(ts + 1, motion_score=0, head_tilt=config.head_tilt_fwd + 1))
    assert result.state is GuardState.POST_ALERT
    assert result.alert_fired is False


def test_post_alert_freezes_evidence(config):
    sm = GuardStateMachine(config)
    ts = drive_to_alert(sm, config)
    sm.tick(make_tick(ts + 1, motion_score=0, head_tilt=config.head_tilt_fwd + 1))
    evidence_at_entry = sm.sleep_evidence_seconds
    sm.tick(make_tick(ts + 50, motion_score=0, head_tilt=config.head_tilt_fwd + 1))
    assert sm.sleep_evidence_seconds == evidence_at_entry


def test_post_alert_realerts_after_cooldown(config):
    sm = GuardStateMachine(config)
    alert_ts = drive_to_alert(sm, config)
    sm.tick(make_tick(alert_ts + 1, motion_score=0, head_tilt=config.head_tilt_fwd + 1))  # -> POST_ALERT
    before_cooldown = sm.tick(make_tick(alert_ts + config.realert_cooldown_sec - 1, motion_score=0, head_tilt=config.head_tilt_fwd + 1))
    assert before_cooldown.alert_fired is False
    result = sm.tick(make_tick(alert_ts + config.realert_cooldown_sec, motion_score=0, head_tilt=config.head_tilt_fwd + 1))
    assert result.state is GuardState.POST_ALERT
    assert result.alert_fired is True


# --------------------------------------------------------------------
# WOKE_UP hold + Fix 2 redoze block
# --------------------------------------------------------------------

def test_woke_up_to_active_after_hold(config):
    sm = GuardStateMachine(config)
    sm.tick(make_tick(0))
    sm.tick(make_tick(1, bbox_wake=True))
    assert sm.state is GuardState.WOKE_UP
    result = sm.tick(make_tick(1 + config.woke_up_hold_sec / 2, motion_score=0))
    assert result.state is GuardState.WOKE_UP
    result = sm.tick(make_tick(1 + config.woke_up_hold_sec, motion_score=0))
    assert result.state is GuardState.ACTIVE


def test_redoze_block_then_lifts(config):
    cfg = replace(config, still_sustain_sec=1, dwell_to_suspect=1, signal_debounce_true_sec=1)
    sm = GuardStateMachine(cfg)
    ts = drive_to_suspect(sm, cfg)
    sm.tick(make_tick(ts + 1, bbox_wake=True))
    assert sm.state is GuardState.WOKE_UP
    ts2 = ts + 1 + cfg.woke_up_hold_sec
    result = sm.tick(make_tick(ts2, motion_score=0))
    assert result.state is GuardState.ACTIVE

    ts3 = ts2 + 1
    sm.tick(make_tick(ts3, motion_score=0))
    ts3 += cfg.still_sustain_sec
    sm.tick(make_tick(ts3, motion_score=0))
    ts3 += 1
    sm.tick(make_tick(ts3, motion_score=0, head_tilt=cfg.head_tilt_fwd + 1))
    ts3 += cfg.signal_debounce_true_sec
    sm.tick(make_tick(ts3, motion_score=0, head_tilt=cfg.head_tilt_fwd + 1))
    ts3 += cfg.dwell_to_suspect
    result = sm.tick(make_tick(ts3, motion_score=0, head_tilt=cfg.head_tilt_fwd + 1))
    assert result.state is GuardState.STILL  # still inside the redoze block

    result = sm.tick(make_tick(ts3 + cfg.redoze_min_zero_sec, motion_score=0, head_tilt=cfg.head_tilt_fwd + 1))
    assert result.state is GuardState.SLEEPING_SUSPECT


# --------------------------------------------------------------------
# Lifecycle
# --------------------------------------------------------------------

def test_on_track_lost_resets_to_absent(config):
    sm = GuardStateMachine(config)
    sm.tick(make_tick(0))
    assert sm.state is GuardState.ACTIVE
    sm.on_track_lost()
    assert sm.state is GuardState.ABSENT


def test_acknowledge_alert_resets_to_active(config):
    sm = GuardStateMachine(config)
    drive_to_alert(sm, config)
    sm.acknowledge_alert()
    assert sm.state is GuardState.ACTIVE
    assert sm.sleep_evidence_seconds == 0
