"""Main loop: source -> detector -> tracker -> features -> state machine
-> alert. Owns RTSP reconnect, signal handling, and all I/O side
effects (CSV, clips, dispatch). Everything CPU/GPU-heavy or stateful
about *why* an alert fires lives in cv/ and state_machine/; this module
just wires those pieces to a live video source.
"""

from __future__ import annotations

import json
import logging
import signal
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

import cv2

from guard_monitor.alert.clip_saver import FrameRingBuffer, save_clip
from guard_monitor.alert.console import ConsoleDispatcher
from guard_monitor.alert.dispatcher import AlertEvent
from guard_monitor.alert.vlm import VLMGate, crop_with_margin, vlm_from_env
from guard_monitor.alert.webhook import WebhookDispatcher
from guard_monitor.cv import roi as roi_mod
from guard_monitor.cv.detector import PersonDetector, PoseDetector
from guard_monitor.cv.tracker import extract_tracks, should_drop_track
from guard_monitor.cv import features as feat
from guard_monitor.log.csv_writer import EventsCsvWriter, write_rows_csv
from guard_monitor.state_machine.guard import GuardState, GuardStateMachine, GuardTick
from guard_monitor.state_machine.post import PostStateMachine, PostTick
from guard_monitor.viz.annotate import AnnotatedVideoWriter, draw_person, draw_roi

logger = logging.getLogger(__name__)

_DEFAULT_FPS = 15  # fallback when the source can't report its own fps
_VLM_ASLEEP_VERDICTS = ("asleep", "dozing")


def _parse_hhmm(value: str) -> tuple:
    hh, mm = value.split(":")
    return int(hh), int(mm)


def is_quiet_window(now: datetime, config) -> bool:
    start_h, start_m = _parse_hhmm(config.quiet_window_start)
    window_start = now.replace(hour=start_h, minute=start_m, second=0, microsecond=0)
    if now < window_start:
        window_start -= timedelta(days=1)
    return (now - window_start).total_seconds() < config.quiet_window_sec


@dataclass
class _TrackContext:
    state_machine: GuardStateMachine
    calibrator: feat.BaselineCalibrator
    smoother: feat.SlidingWindowSmoother
    short_smoother: feat.SlidingWindowSmoother
    bbox_tracker: feat.BboxMotionTracker
    first_seen_ts: float
    last_seen_ts: float
    last_kps: Optional[object] = None
    still_no_signal_since: Optional[float] = None
    logged_default_baseline: bool = False
    pending_alert: Optional["_PendingClip"] = None


@dataclass
class _PendingClip:
    alert_ts: float
    ready_ts: float
    track_id: Optional[int]
    alert_type: str
    started_at: str
    detected_at: str


def _open_capture(source: str) -> cv2.VideoCapture:
    try:
        src = int(source)
    except ValueError:
        src = source
    return cv2.VideoCapture(src)


def _is_live_source(source: str) -> bool:
    return source.lower().startswith("rtsp://") or source == "0"


def build_dispatcher(args, config):
    if args.webhook_url:
        return WebhookDispatcher(args.webhook_url, config)
    return ConsoleDispatcher()


def _make_track_context(config, ts: float) -> _TrackContext:
    return _TrackContext(
        state_machine=GuardStateMachine(config),
        calibrator=feat.BaselineCalibrator(
            config.calibration_sec,
            config.default_baseline_spine,
            config.calib_max_head_tilt,
            config.calib_max_spine_lean,
            config.calib_max_motion,
            config.calib_min_clean_sec,
        ),
        smoother=feat.SlidingWindowSmoother(config.sliding_window),
        short_smoother=feat.SlidingWindowSmoother(config.wake_short_window_sec),
        bbox_tracker=feat.BboxMotionTracker(
            config.bbox_wake_window_sec, config.bbox_height_wake_frac, config.bbox_center_wake_frac
        ),
        first_seen_ts=ts,
        last_seen_ts=ts,
    )


def run(args, config) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    events_csv = EventsCsvWriter(str(out_dir))
    dispatcher = build_dispatcher(args, config)
    vlm_gate = VLMGate(vlm_from_env(config.webhook_timeout_sec), config.vlm_recheck_sec, config.vlm_max_calls_per_hour)

    gates = {}
    if args.roi_config:
        gates = roi_mod.load_roi_config(args.roi_config)
    polygon = roi_mod.get_gate_polygon(gates, args.gate_id) if gates else None
    if args.roi_config and polygon is None:
        logger.warning("no ROI polygon for gate_id=%s in %s -- vacant-post logic disabled", args.gate_id, args.roi_config)
    elif not args.roi_config:
        logger.warning("no --roi-config given -- vacant-post logic disabled, sleep detection still runs")

    stop_requested = {"flag": False}

    def _handle_signal(signum, _frame):
        logger.info("received signal %s, shutting down", signum)
        stop_requested["flag"] = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    pose_detector = PoseDetector(weights=args.pose_weights, device=args.device)
    person_detector = PersonDetector(weights=args.person_weights, device=args.device) if polygon else None

    post_machine = PostStateMachine(config) if polygon else None
    tracks: Dict[int, _TrackContext] = {}
    ring_buffer = FrameRingBuffer(keep_sec=config.clip_duration_sec)
    rows_history: Dict[int, list] = {}

    cap = _open_capture(args.source)
    fps = cap.get(cv2.CAP_PROP_FPS) or _DEFAULT_FPS
    live = _is_live_source(str(args.source))
    frame_idx = 0
    start_wall = time.time()
    last_infer_ts: Optional[float] = None

    annotator = None
    if getattr(args, "annotate", False):
        annotate_path = args.annotate_path or str(out_dir / "annotated" / f"{args.gate_id}.mp4")
        annotator = AnnotatedVideoWriter(annotate_path, fps)
        logger.info("annotated video will be written to %s", annotate_path)

    while not stop_requested["flag"]:
        ok, frame = cap.read()
        if not ok:
            if not live:
                break
            logger.warning("source read failed, reconnecting in %.1fs", config.reconnect_interval_sec)
            cap.release()
            time.sleep(config.reconnect_interval_sec)
            cap = _open_capture(args.source)
            fps = cap.get(cv2.CAP_PROP_FPS) or _DEFAULT_FPS
            continue

        ts = time.time() - start_wall if live else frame_idx / fps
        frame_idx += 1
        now_wall = datetime.now()
        quiet = is_quiet_window(now_wall, config)

        ring_buffer.push(ts, frame)
        annotated_frame = frame.copy() if annotator is not None else None

        # Throughput fix: skip inference on frames in between when
        # INFER_FPS caps the rate below the source's native fps. The
        # recorded clip/annotated video still get every frame at full
        # quality above (ring_buffer.push) -- only detection runs less
        # often. Every downstream timer is ts-based (see dt-normalized
        # motion_score and the track-continuity fix), so skipped frames
        # are handled the same way as an ordinary detection gap.
        should_infer = (
            config.infer_fps <= 0
            or last_infer_ts is None
            or (ts - last_infer_ts) >= (1 / config.infer_fps)
        )

        if should_infer:
            last_infer_ts = ts
            pose_result = pose_detector.track(frame)
            people = extract_tracks(pose_result)
            seen_ids = set()
            for person in people:
                seen_ids.add(person.track_id)
                annotation = _process_person(
                    person, frame, ts, now_wall, quiet, tracks, rows_history, args.gate_id,
                    events_csv, dispatcher, ring_buffer, out_dir, config, fps, vlm_gate,
                )
                if annotated_frame is not None and annotation is not None:
                    hud_lines = [
                        f"motion {annotation['motion_score']:.3f}",
                        f"tilt {annotation['head_tilt']:.1f} lean {annotation['spine_lean']:.1f}",
                        f"hand {annotation['hand_at_head']} evidence {annotation['sleep_evidence_seconds']:.0f}s",
                    ]
                    draw_person(annotated_frame, person.track_id, person.bbox, person.keypoints, annotation["state"], config.kp_min_conf, hud_lines)

            for track_id in list(tracks.keys()):
                if track_id not in seen_ids and should_drop_track(ts, tracks[track_id].last_seen_ts, config.track_lost_grace_sec):
                    tracks[track_id].state_machine.on_track_lost()
                    vlm_gate.drop_track(track_id)
                    del tracks[track_id]
                    rows_history.pop(track_id, None)
                # else: missed for less than TRACK_LOST_GRACE_SEC -- left
                # dormant, not deleted (see should_drop_track's docstring)

            if polygon is not None and person_detector is not None:
                person_result = person_detector.track(frame)
                occupants = extract_tracks(person_result)
                occupied = roi_mod.any_track_in_roi([p.bbox for p in occupants], polygon)
                post_result = post_machine.tick(PostTick(ts=ts, occupied=occupied, quiet_window=quiet))
                if post_result.alert_fired:
                    _handle_vacant_alert(ts, now_wall, args.gate_id, events_csv, dispatcher, ring_buffer, out_dir, config, fps, post_machine)
                if annotated_frame is not None:
                    draw_roi(annotated_frame, polygon, occupied)

        if annotator is not None:
            annotator.write(annotated_frame)

        _flush_pending_clips(ts, tracks, rows_history, args.gate_id, events_csv, dispatcher, ring_buffer, out_dir, config, fps)

    cap.release()
    if annotator is not None:
        annotator.release()
    logger.info("shutdown complete, events log at %s", events_csv.path)


def _process_person(person, frame, ts, now_wall, quiet, tracks, rows_history, gate_id, events_csv, dispatcher, ring_buffer, out_dir, config, fps, vlm_gate):
    ctx = tracks.get(person.track_id)
    if ctx is None:
        ctx = _make_track_context(config, ts)
        tracks[person.track_id] = ctx
        rows_history[person.track_id] = []

    gap = ts - ctx.last_seen_ts
    ctx.last_seen_ts = ts
    if gap > config.motion_gap_reset_sec:
        # Track-continuity fix: a detection gap shorter than
        # TRACK_LOST_GRACE_SEC keeps the track's long-timescale state
        # (evidence, calibration, debounce) alive, but the frame-to-frame
        # delta buffers below can't be trusted across an unknown gap --
        # reset them so reconnection doesn't get misread as a motion spike.
        ctx.last_kps = None
        ctx.bbox_tracker.reset()

    if person.keypoints is None:
        return None

    kps = person.keypoints
    person_h = feat.person_height(person.bbox)

    raw_motion = (
        feat.motion_score(ctx.last_kps, kps, person_h, config.kp_min_conf, gap)
        if ctx.last_kps is not None
        else 0
    )
    ctx.last_kps = kps
    motion = ctx.smoother.push(ts, raw_motion)
    motion_short = ctx.short_smoother.push(ts, raw_motion)
    bbox_wake = ctx.bbox_tracker.push(ts, person.bbox, person_h)

    raw_head_tilt, head_tilt_degraded = feat.head_tilt_deg(kps, 0, config.kp_min_conf)
    raw_spine_lean, spine_lean_degraded = feat.spine_lean_deg(kps, 0, config.kp_min_conf)
    head_low_conf = feat.head_conf_low(kps, config.kp_min_conf)
    visible = feat.body_visible(kps, config.kp_min_conf)
    hand_ratio, hand_at_head_degraded = feat.hand_at_head_ratio(kps, person_h, config.kp_min_conf)
    hand_at_head_flag = hand_ratio is not None and hand_ratio < config.hand_head_dist

    ctx.calibrator.observe(ts, not head_low_conf, raw_head_tilt, raw_spine_lean, motion)
    baseline = ctx.calibrator.finalize_if_ready(ts)
    if ctx.calibrator.used_default and not ctx.logged_default_baseline:
        logger.warning(
            "track %s: fewer than calib_min_clean_sec of clean calibration frames, using default_baseline_spine=%s",
            person.track_id, config.default_baseline_spine,
        )
        ctx.logged_default_baseline = True

    head_tilt = raw_head_tilt - baseline
    spine_lean = raw_spine_lean - baseline
    track_age = ts - ctx.first_seen_ts

    # Fix 5: only reach for the VLM once pose math has had nothing to say
    # for LONG_STILL_SEC straight -- it's a second opinion for the case
    # pose features structurally can't see, not a replacement for them.
    raw_signal_present = (
        head_tilt > config.head_tilt_fwd
        or spine_lean > config.spine_lean_fwd
        or hand_at_head_flag
        or (head_low_conf and visible)
    )
    if ctx.state_machine.state is GuardState.STILL and not raw_signal_present:
        if ctx.still_no_signal_since is None:
            ctx.still_no_signal_since = ts
    else:
        ctx.still_no_signal_since = None

    vlm_asleep = False
    if ctx.still_no_signal_since is not None and (ts - ctx.still_no_signal_since) >= config.long_still_sec:
        crop = crop_with_margin(frame, person.bbox, config.vlm_crop_margin_frac)
        verdict, fresh_call = vlm_gate.maybe_assess(ts, person.track_id, crop)
        if fresh_call:
            events_csv.append_vlm_call(gate_id, person.track_id, now_wall.isoformat(), verdict)
        vlm_asleep = verdict in _VLM_ASLEEP_VERDICTS

    result = ctx.state_machine.tick(
        GuardTick(
            ts=ts,
            motion_score=motion,
            motion_short=motion_short,
            bbox_wake=bbox_wake,
            head_tilt=head_tilt,
            spine_lean=spine_lean,
            hand_at_head=hand_at_head_flag,
            head_conf_low=head_low_conf,
            body_visible=visible,
            vlm_asleep=vlm_asleep,
            track_age=track_age,
            quiet_window=quiet,
        )
    )

    ctx.calibrator.maybe_rebaseline(
        ts,
        result.state is GuardState.ACTIVE,
        raw_head_tilt,
        raw_spine_lean,
        config.rebaseline_min_active_sec,
        config.rebaseline_max_head_tilt,
        config.rebaseline_max_shift_deg,
        config.rebaseline_blend_old,
    )

    rows_history[person.track_id].append({
        "timestamp": ts,
        "track_id": person.track_id,
        "state": result.state.value,
        "motion_score": motion,
        "motion_short": motion_short,
        "bbox_wake": bbox_wake,
        "head_tilt": head_tilt,
        "spine_lean": spine_lean,
        "hand_at_head": hand_at_head_flag,
        "head_conf_low": head_low_conf,
        "body_visible": visible,
        "head_tilt_degraded": head_tilt_degraded,
        "spine_lean_degraded": spine_lean_degraded,
        "hand_at_head_degraded": hand_at_head_degraded,
        "vlm_asleep": vlm_asleep,
        "sleep_evidence_seconds": ctx.state_machine.sleep_evidence_seconds,
        "track_age": track_age,
    })
    cutoff = ts - config.clip_duration_sec
    rows_history[person.track_id] = [r for r in rows_history[person.track_id] if r["timestamp"] >= cutoff]

    if result.alert_fired:
        half_window = config.clip_duration_sec / 2
        ctx.pending_alert = _PendingClip(
            alert_ts=ts,
            ready_ts=ts + half_window,
            track_id=person.track_id,
            alert_type="SLEEPING",
            started_at=(now_wall - timedelta(seconds=config.dwell_to_alert)).isoformat(),
            detected_at=now_wall.isoformat(),
        )
        # No acknowledge_alert() here (Fix 2): POST_ALERT keeps managing
        # itself -- a real wake or the next re-alert cooldown tick is
        # what moves it, not the runner flushing this clip.

    return {
        "state": result.state,
        "motion_score": motion,
        "motion_short": motion_short,
        "bbox_wake": bbox_wake,
        "head_tilt": head_tilt,
        "spine_lean": spine_lean,
        "hand_at_head": hand_at_head_flag,
        "sleep_evidence_seconds": ctx.state_machine.sleep_evidence_seconds,
        "track_age": track_age,
        "first_seen_ts": ctx.first_seen_ts,
    }


def _flush_pending_clips(ts, tracks, rows_history, gate_id, events_csv, dispatcher, ring_buffer, out_dir, config, fps):
    for track_id, ctx in list(tracks.items()):
        pending = ctx.pending_alert
        if pending is None or ts < pending.ready_ts:
            continue
        _export_alert(pending, gate_id, events_csv, dispatcher, ring_buffer, rows_history.get(track_id, []), out_dir, config, fps)
        ctx.pending_alert = None


def _alert_dir_name(pending: "_PendingClip") -> str:
    # detected_at already carries microsecond precision, which is what
    # keeps two tracks alerting in the same wall-clock second from
    # colliding on one directory; the track id suffix disambiguates
    # further for the (astronomically unlikely but cheap-to-avoid) case
    # of two tracks sharing the same microsecond timestamp too.
    stamp = pending.detected_at.replace(":", "").replace(".", "").replace("-", "")
    suffix = f"_track{pending.track_id}" if pending.track_id is not None else ""
    return f"{stamp}{suffix}"


def _export_alert(pending, gate_id, events_csv, dispatcher, ring_buffer, rows, out_dir, config, fps):
    half_window = config.clip_duration_sec / 2
    alert_dir = out_dir / "alerts" / gate_id / _alert_dir_name(pending)
    alert_dir.mkdir(parents=True, exist_ok=True)

    clip_frames = ring_buffer.slice(pending.alert_ts, half_window)
    clip_path = alert_dir / "clip.mp4"
    save_clip(clip_frames, str(clip_path), fps)

    rows_path = alert_dir / "rows.csv"
    write_rows_csv(str(rows_path), rows)

    meta = {
        "alert_type": pending.alert_type,
        "gate_id": gate_id,
        "track_id": pending.track_id,
        "started_at": pending.started_at,
        "detected_at": pending.detected_at,
        "duration_sec": config.dwell_to_alert,
        "evidence": {"frames_in_clip": len(clip_frames), "rows_in_window": len(rows)},
    }
    meta_path = alert_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    event = AlertEvent(
        alert_type=pending.alert_type,
        gate_id=gate_id,
        track_id=pending.track_id,
        started_at=pending.started_at,
        detected_at=pending.detected_at,
        duration_sec=config.dwell_to_alert,
        evidence=meta["evidence"],
        clip_path=str(clip_path),
        rows_csv_path=str(rows_path),
        meta_json_path=str(meta_path),
    )
    events_csv.append({
        "timestamp": pending.detected_at,
        "gate_id": gate_id,
        "alert_type": pending.alert_type,
        "track_id": pending.track_id,
        "started_at": pending.started_at,
        "detected_at": pending.detected_at,
        "duration_sec": config.dwell_to_alert,
        "clip_path": str(clip_path),
        "rows_csv_path": str(rows_path),
        "meta_json_path": str(meta_path),
    })
    dispatcher.dispatch(event)


def _handle_vacant_alert(ts, now_wall, gate_id, events_csv, dispatcher, ring_buffer, out_dir, config, fps, post_machine):
    pending = _PendingClip(
        alert_ts=ts,
        ready_ts=ts,  # vacancy has no live "person" ring context; export immediately
        track_id=None,
        alert_type="VACANT_POST",
        started_at=(now_wall - timedelta(seconds=config.vacant_dwell_sec)).isoformat(),
        detected_at=now_wall.isoformat(),
    )
    _export_alert(pending, gate_id, events_csv, dispatcher, ring_buffer, [], out_dir, config, fps)
    post_machine.acknowledge_alert()
