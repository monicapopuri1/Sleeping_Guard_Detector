"""Dispatcher interface: one method, `dispatch(alert_event)`.

runner.py always talks to alerting through this interface -- it never
knows or cares whether the concrete implementation prints to a console
or POSTs to a webhook.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class AlertEvent:
    alert_type: str  # "SLEEPING" or "VACANT_POST"
    gate_id: str
    track_id: Optional[int]
    started_at: str  # ISO8601
    detected_at: str  # ISO8601
    duration_sec: float
    evidence: dict = field(default_factory=dict)
    clip_path: Optional[str] = None
    rows_csv_path: Optional[str] = None
    meta_json_path: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


class AlertDispatcher:
    def dispatch(self, alert_event: AlertEvent) -> None:
        raise NotImplementedError
