"""events.csv (append-only, top-level, continuous) and per-alert
rows.csv (per-frame features for a single alert window).

CSV is the source of truth for this deployment -- no database. The
append-only writer opens in "a" mode and writes the header only if the
file doesn't exist yet, so the log survives process restarts and RTSP
reconnects without ever losing a prior row.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

EVENTS_CSV_HEADER = [
    "timestamp",
    "gate_id",
    "alert_type",
    "track_id",
    "started_at",
    "detected_at",
    "duration_sec",
    "clip_path",
    "rows_csv_path",
    "meta_json_path",
    "note",
]

ROWS_CSV_HEADER = [
    "timestamp",
    "track_id",
    "state",
    "motion_score",
    "motion_short",
    "bbox_wake",
    "head_tilt",
    "spine_lean",
    "hand_at_head",
    "head_conf_low",
    "body_visible",
    "head_tilt_degraded",
    "spine_lean_degraded",
    "hand_at_head_degraded",
    "vlm_asleep",
    "sleep_evidence_seconds",
    "track_age",
]


class EventsCsvWriter:
    def __init__(self, out_dir: str):
        self.path = Path(out_dir) / "events.csv"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with self.path.open("w", newline="") as f:
                csv.writer(f).writerow(EVENTS_CSV_HEADER)

    def append(self, row: dict) -> None:
        with self.path.open("a", newline="") as f:
            csv.writer(f).writerow([row.get(col, "") for col in EVENTS_CSV_HEADER])

    def append_vlm_call(self, gate_id: str, track_id: int, timestamp_iso: str, verdict: str) -> None:
        """Fix 5.5: every VLM call logged into events.csv as its own row
        type (alert_type="VLM_CALL"), distinguishable from real alerts."""
        self.append({
            "timestamp": timestamp_iso,
            "gate_id": gate_id,
            "alert_type": "VLM_CALL",
            "track_id": track_id,
            "note": f"verdict={verdict}",
        })


def write_rows_csv(out_path: str, rows: Iterable[dict]) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(ROWS_CSV_HEADER)
        for row in rows:
            writer.writerow([row.get(col, "") for col in ROWS_CSV_HEADER])
