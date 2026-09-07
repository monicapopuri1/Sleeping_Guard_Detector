"""Fix 5: second-opinion VLM check for long, signal-less stillness.

Pose math has no signal for a guard who sits motionless, upright-ish,
just quietly asleep with no head drop, no hand-on-head, no occlusion --
verified on real footage where this happens for 38+ seconds at a time.
Eyes aren't resolvable at typical gate-house camera distance, so this
does not attempt eyelid/EAR detection; it asks a vision-language model
one direct question about a cropped frame instead.

Two implementations behind the same interface:
  - NullVLM: always "unknown". This is the default -- the system must
    run fully without a VLM configured, just without this signal.
  - OpenAICompatibleVLM: POSTs to an OpenAI-compatible /chat/completions
    endpoint. Configured entirely from env vars (VLM_BASE_URL,
    VLM_MODEL, VLM_API_KEY) per the fix spec -- these are deployment
    secrets/endpoints, not per-site detection tunables, so they don't
    belong in config.py/CLI flags.

VLMGate wraps either implementation with the required rate limits: at
most one call per track per VLM_RECHECK_SEC, and a global cap of
VLM_MAX_CALLS_PER_HOUR.
"""

from __future__ import annotations

import base64
import logging
import os
from collections import deque
from typing import Optional

import cv2
import numpy as np
import requests

logger = logging.getLogger(__name__)

_PROMPT = (
    "One sentence: is this person asleep, dozing, or awake and alert? "
    "Answer with one word: asleep, dozing, or awake."
)
_VALID_VERDICTS = ("asleep", "dozing", "awake")


class VLMClient:
    def assess(self, crop_bgr: np.ndarray) -> str:
        raise NotImplementedError


class NullVLM(VLMClient):
    """Used when no VLM is configured. Always returns 'unknown', which
    the state machine treats as "no additional signal"."""

    def assess(self, crop_bgr: np.ndarray) -> str:
        return "unknown"


class OpenAICompatibleVLM(VLMClient):
    def __init__(self, base_url: str, model: str, api_key: str, timeout_sec: float):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout_sec = timeout_sec

    def assess(self, crop_bgr: np.ndarray) -> str:
        ok, buf = cv2.imencode(".jpg", crop_bgr)
        if not ok:
            return "unknown"
        b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _PROMPT},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    ],
                }
            ],
        }
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=self.timeout_sec,
            )
            response.raise_for_status()
            text = response.json()["choices"][0]["message"]["content"].strip().lower()
        except Exception:
            logger.exception("VLM call failed")
            return "unknown"

        for verdict in _VALID_VERDICTS:
            if verdict in text:
                return verdict
        return "unknown"


def vlm_from_env(timeout_sec: float) -> VLMClient:
    api_key = os.environ.get("VLM_API_KEY")
    if not api_key:
        return NullVLM()
    base_url = os.environ.get("VLM_BASE_URL", "https://api.openai.com/v1")
    model = os.environ.get("VLM_MODEL", "gpt-4o-mini")
    return OpenAICompatibleVLM(base_url, model, api_key, timeout_sec)


class VLMGate:
    """Rate-limits and caches VLM verdicts per track."""

    def __init__(self, client: VLMClient, recheck_sec: float, max_calls_per_hour: float):
        self.client = client
        self.recheck_sec = recheck_sec
        self.max_calls_per_hour = max_calls_per_hour
        self._last_call_ts: dict = {}
        self._cached_verdict: dict = {}
        self._global_call_times: deque = deque()

    def _global_cap_ok(self, ts: float) -> bool:
        cutoff = ts - 3600
        while self._global_call_times and self._global_call_times[0] < cutoff:
            self._global_call_times.popleft()
        return len(self._global_call_times) < self.max_calls_per_hour

    def maybe_assess(self, ts: float, track_id: int, crop_bgr: Optional[np.ndarray]) -> "tuple[str, bool]":
        """Returns (verdict, made_fresh_call). Never calls the VLM more
        than once per track per recheck_sec, and never exceeds the
        global hourly cap -- callers past the cap just keep getting the
        last cached verdict (or "unknown")."""
        last_ts = self._last_call_ts.get(track_id)
        due = last_ts is None or (ts - last_ts) >= self.recheck_sec
        if not due or crop_bgr is None or crop_bgr.size == 0:
            return self._cached_verdict.get(track_id, "unknown"), False
        if not self._global_cap_ok(ts):
            logger.warning("VLM global hourly cap reached, skipping call for track %s", track_id)
            return self._cached_verdict.get(track_id, "unknown"), False

        self._last_call_ts[track_id] = ts
        self._global_call_times.append(ts)
        verdict = self.client.assess(crop_bgr)
        self._cached_verdict[track_id] = verdict
        return verdict, True

    def drop_track(self, track_id: int) -> None:
        self._last_call_ts.pop(track_id, None)
        self._cached_verdict.pop(track_id, None)


def crop_with_margin(frame: np.ndarray, bbox, margin_frac: float) -> np.ndarray:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    mx, my = bw * margin_frac, bh * margin_frac
    x1 = max(0, int(x1 - mx))
    y1 = max(0, int(y1 - my))
    x2 = min(w, int(x2 + mx))
    y2 = min(h, int(y2 + my))
    if x2 <= x1 or y2 <= y1:
        return np.empty((0, 0, 3), dtype=frame.dtype)
    return frame[y1:y2, x1:x2]
