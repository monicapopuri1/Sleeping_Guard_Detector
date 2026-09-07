from __future__ import annotations

import json
import logging

from guard_monitor.alert.dispatcher import AlertDispatcher, AlertEvent

logger = logging.getLogger(__name__)


class ConsoleDispatcher(AlertDispatcher):
    """Prints the alert as a single JSON line via stdlib logging (never
    a bare `print` -- see requirement: never print from the model loop)."""

    def dispatch(self, alert_event: AlertEvent) -> None:
        logger.info(json.dumps(alert_event.to_dict()))
