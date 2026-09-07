"""Vacant-post ROI polygons: load, point-in-polygon hit-test.

Config schema (--roi-config):
    {"gates": {"<gate-id>": [[x1, y1], [x2, y2], ...], ...}}

If no --roi-config is given, or the given gate-id has no polygon, the
caller is expected to log a warning and skip vacant-post logic entirely
(sleep detection still runs) -- that decision lives in runner.py, not
here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

Point = Tuple[float, float]


def load_roi_config(path: str) -> dict:
    data = json.loads(Path(path).read_text())
    return data.get("gates", {})


def get_gate_polygon(gates: dict, gate_id: str) -> Optional[List[Point]]:
    polygon = gates.get(gate_id)
    if not polygon:
        return None
    return [(float(x), float(y)) for x, y in polygon]


def point_in_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    """Standard ray-casting point-in-polygon test."""
    x, y = point
    inside = False
    n = len(polygon)
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        crosses = (yi > y) != (yj > y)
        if crosses:
            x_intersect = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < x_intersect:
                inside = not inside
        j = i
    return inside


def bbox_center(bbox) -> Point:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def any_track_in_roi(bboxes: Sequence, polygon: Sequence[Point]) -> bool:
    return any(point_in_polygon(bbox_center(b), polygon) for b in bboxes)
