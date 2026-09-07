from __future__ import annotations

import json
import logging
import urllib.request

from guard_monitor.alert.dispatcher import AlertDispatcher, AlertEvent

logger = logging.getLogger(__name__)


class WebhookDispatcher(AlertDispatcher):
    """POSTs the alert as JSON to --webhook-url. stdlib urllib only --
    no extra HTTP client dependency."""

    def __init__(self, url: str, config):
        self.url = url
        self.timeout_sec = config.webhook_timeout_sec

    def dispatch(self, alert_event: AlertEvent) -> None:
        payload = json.dumps(alert_event.to_dict()).encode("utf-8")
        request = urllib.request.Request(
            self.url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_sec) as response:
                logger.debug("webhook dispatch status=%s", response.status)
        except Exception:
            logger.exception("webhook dispatch failed url=%s", self.url)
