"""Fix-pack acceptance test.

Replays 2_1.mp4 and 2_3.mp4 through the *exact same* per-track pipeline
runner.py uses in production (imported directly, not reimplemented --
this exercises the real shipped code, not a copy that could drift from
it), records the guard track's state at every tick, and checks it
against the human-verified ground-truth windows from the fix pack.

Usage:
    python examples/verify_videos.py \\
        --video-2-1 /path/to/2_1.mp4 --video-2-3 /path/to/2_3.mp4 \\
        --pose-weights /path/to/yolo11n-pose.pt --device mps \\
        --out-dir ./verify_out

Ground-truth windows are approximate human observations, not exact
reproductions of DWELL_TO_SUSPECT/STILL_SUSTAIN_SEC/etc, so a window's
actual state is checked against a small equivalence set (see
EQUIVALENCE below) rather than requiring an exact string match --
e.g. "SLEEPING_SUSPECT" also accepts ALERT/POST_ALERT, since those are
strictly *more* evidence of sleep, not less.

Prints one PASS/FAIL line per window (state at the window's midpoint)
plus final tallies: windows correct / false-negative / false-positive.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import cv2

from guard_monitor.alert.console import ConsoleDispatcher
from guard_monitor.alert.clip_saver import FrameRingBuffer
from guard_monitor.alert.vlm import VLMGate, vlm_from_env
from guard_monitor.config import GuardMonitorConfig, add_config_args, config_from_args
from guard_monitor.cv.detector import PoseDetector
from guard_monitor.cv.tracker import extract_tracks, should_drop_track
from guard_monitor.log.csv_writer import EventsCsvWriter
from guard_monitor.runner import _flush_pending_clips, _process_person


@dataclass
class Window:
    start: float
    end: float
    expected: str
    label: str


# actual-state -> set of expected-labels it satisfies. SLEEPING_SUSPECT
# and ALERT both accept their own "further along" states (ALERT/
# POST_ALERT are strictly more sleep-evidence than plain
# SLEEPING_SUSPECT); ACTIVE and WOKE_UP accept each other since a
# sampled midpoint can land just after the WOKE_UP hold elapses.
EQUIVALENCE = {
    "ACTIVE": {"ACTIVE", "WOKE_UP"},
    "STILL": {"STILL"},
    "SLEEPING_SUSPECT": {"SLEEPING_SUSPECT", "ALERT", "POST_ALERT"},
    "ALERT": {"ALERT", "POST_ALERT"},
    "WOKE_UP": {"WOKE_UP", "ACTIVE"},
}

WINDOWS = {
    "2_1.mp4": [
        Window(0, 60, "ACTIVE", "0:00-1:00 ACTIVE"),
        Window(63, 93, "STILL", "1:03-1:33 STILL"),
        Window(93, 183, "SLEEPING_SUSPECT", "1:33-3:03 SLEEPING_SUSPECT then ALERT by ~3:03"),
        Window(188, 277, "SLEEPING_SUSPECT", "3:08-4:37 SLEEPING_SUSPECT retained / re-alert within cooldown"),
        Window(278, 285, "ALERT", "4:38-4:45 still ALERT/asleep"),
        Window(285, 288, "WOKE_UP", "4:45-4:48 WOKE_UP then ACTIVE"),
    ],
    "2_3.mp4": [
        Window(0, 20, "STILL", "0:00-0:20 STILL -> SLEEPING_SUSPECT (calibration must not absorb the sleeping pose)"),
        Window(20, 30, "ACTIVE", "0:20-0:30 ACTIVE (brief genuine wake)"),
        Window(30, 137, "STILL", "0:30-2:17 STILL, escalating via VLM/posture"),
        Window(137, 188, "ACTIVE", "2:17-3:08 ACTIVE (stands up, moves a chair at 2:32-2:38)"),
        Window(188, 300, "SLEEPING_SUSPECT", "3:08-5:00 STILL -> SLEEPING_SUSPECT via one-sided hand_at_head, ALERT within dwell"),
    ],
}


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video-2-1", default=None, help="omit to skip this video")
    p.add_argument("--video-2-3", default=None, help="omit to skip this video")
    p.add_argument("--pose-weights", default="yolo11n-pose.pt")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out-dir", default="./verify_out")
    add_config_args(p)
    return p


_DIAG_HEADER = [
    "ts", "track_id", "state", "motion_score", "motion_short", "bbox_wake",
    "head_tilt", "spine_lean", "hand_at_head", "sleep_evidence_seconds",
    "track_age", "first_seen_ts",
]


def replay(video_path: str, config: GuardMonitorConfig, pose_weights: str, device: str, out_dir: Path, gate_id: str) -> Tuple[List[Tuple[float, int, str]], int]:
    """Returns (timeline, wake_transition_count). wake_transition_count is
    how many times any track entered WOKE_UP -- a proxy for how often the
    Fix-1 wake condition (bbox backstop / short-burst / sustained long
    threshold) fires, including possible false positives from ordinary
    detection-bbox jitter on real footage."""
    detector = PoseDetector(weights=pose_weights, device=device)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 15

    events_csv = EventsCsvWriter(str(out_dir))
    dispatcher = ConsoleDispatcher()
    vlm_gate = VLMGate(vlm_from_env(config.webhook_timeout_sec), config.vlm_recheck_sec, config.vlm_max_calls_per_hour)
    ring_buffer = FrameRingBuffer(keep_sec=config.clip_duration_sec)

    tracks: dict = {}
    rows_history: dict = {}
    timeline: List[Tuple[float, int, str]] = []
    diag_rows: List[list] = []
    last_state: dict = {}
    wake_transitions = 0

    frame_idx = 0
    now_wall = datetime.now()  # fixed reference; irrelevant for a file replay well outside any quiet window
    quiet = False
    last_infer_ts = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        ts = frame_idx / fps
        frame_idx += 1
        ring_buffer.push(ts, frame)

        # Mirrors runner.run()'s should_infer gate exactly -- this script
        # must exercise the same throughput behavior as production, not
        # a hand-copied approximation of it (a prior version of this
        # file had its own un-gated detector.track(frame) call here,
        # which meant --infer-fps silently did nothing in this script).
        should_infer = (
            config.infer_fps <= 0
            or last_infer_ts is None
            or (ts - last_infer_ts) >= (1 / config.infer_fps)
        )
        if not should_infer:
            _flush_pending_clips(ts, tracks, rows_history, gate_id, events_csv, dispatcher, ring_buffer, out_dir, config, fps)
            continue
        last_infer_ts = ts

        pose_result = detector.track(frame)
        people = extract_tracks(pose_result)
        seen = set()
        for person in people:
            seen.add(person.track_id)
            annotation = _process_person(
                person, frame, ts, now_wall, quiet, tracks, rows_history, gate_id,
                events_csv, dispatcher, ring_buffer, out_dir, config, fps, vlm_gate,
            )
            if annotation is not None:
                state_value = annotation["state"].value
                timeline.append((ts, person.track_id, state_value))
                diag_rows.append([
                    ts, person.track_id, state_value,
                    annotation["motion_score"], annotation["motion_short"], annotation["bbox_wake"],
                    annotation["head_tilt"], annotation["spine_lean"], annotation["hand_at_head"],
                    annotation["sleep_evidence_seconds"],
                    annotation["track_age"], annotation["first_seen_ts"],
                ])
                if state_value == "WOKE_UP" and last_state.get(person.track_id) != "WOKE_UP":
                    wake_transitions += 1
                last_state[person.track_id] = state_value

        for track_id in list(tracks.keys()):
            if track_id not in seen and should_drop_track(ts, tracks[track_id].last_seen_ts, config.track_lost_grace_sec):
                tracks[track_id].state_machine.on_track_lost()
                vlm_gate.drop_track(track_id)
                del tracks[track_id]
                rows_history.pop(track_id, None)

        _flush_pending_clips(ts, tracks, rows_history, gate_id, events_csv, dispatcher, ring_buffer, out_dir, config, fps)

    cap.release()

    with (out_dir / "timeline.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(_DIAG_HEADER)
        writer.writerows(diag_rows)

    return timeline, wake_transitions


def dominant_track_id(timeline: List[Tuple[float, int, str]]) -> Optional[int]:
    counts: dict = {}
    for _, tid, _ in timeline:
        counts[tid] = counts.get(tid, 0) + 1
    return max(counts, key=counts.get) if counts else None


def state_at(timeline: List[Tuple[float, int, str]], track_id: int, ts: float) -> Optional[str]:
    """Last known state for `track_id` at or before `ts`."""
    best = None
    for row_ts, tid, state in timeline:
        if tid != track_id:
            continue
        if row_ts <= ts:
            best = state
        else:
            break
    return best


def main() -> None:
    args = build_arg_parser().parse_args()
    config = config_from_args(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    videos = {"2_1.mp4": args.video_2_1, "2_3.mp4": args.video_2_3}
    total = correct = false_negative = false_positive = 0

    for name, path in videos.items():
        if not path:
            continue
        print(f"\n=== {name} ===")
        timeline, wake_transitions = replay(path, config, args.pose_weights, args.device, out_dir / name.replace(".", "_"), gate_id="verify")
        track_id = dominant_track_id(timeline)
        print(f"dominant track id: {track_id} ({len(timeline)} ticks), WOKE_UP entered {wake_transitions} time(s)")

        for w in WINDOWS[name]:
            mid = (w.start + w.end) / 2
            actual = state_at(timeline, track_id, mid) if track_id is not None else None
            allowed = EQUIVALENCE[w.expected]
            passed = actual in allowed
            total += 1
            if passed:
                correct += 1
                verdict = "PASS"
            else:
                verdict = "FAIL"
                if actual in ("ACTIVE", None) and w.expected in ("STILL", "SLEEPING_SUSPECT", "ALERT"):
                    false_negative += 1
                else:
                    false_positive += 1
            print(f"[{verdict}] t={mid:6.1f}s expected~{w.expected:<17} actual={str(actual):<17} -- {w.label}")

    print("\n=== summary ===")
    print(f"windows correct: {correct}/{total}")
    print(f"false-negative-ish: {false_negative}")
    print(f"false-positive-ish: {false_positive}")


if __name__ == "__main__":
    main()
