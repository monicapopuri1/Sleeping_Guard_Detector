# guard-monitor

Real-time vacant-post and sleeping-guard detection from a single CCTV
stream per process. Pre-trained YOLO11n-pose + BoT-SORT only -- no
training, no labels.

## Quick start

```
pip install -r requirements.txt
python -m guard_monitor --source /path/to/camera.mp4 --gate-id gate-a \
  --out-dir ./out --roi-config config/roi.example.json
```

`--source` accepts an `rtsp://` URL, a video file path, or a webcam
index (`0`). Results land in `./out/events.csv` (one row per alert) and
`./out/alerts/gate-a/<timestamp>/` (`clip.mp4`, `rows.csv`,
`meta.json`) for each alert.

## Tuning

Every threshold has a CLI flag and a default in `guard_monitor/config.py`.
Run `python -m guard_monitor --help` for the full list (e.g.
`--still-thresh`, `--head-tilt-fwd`, `--dwell-to-alert`,
`--vacant-dwell-min`). No code changes are needed to retune a site.

## Alerts

`alert_dispatcher.py`'s `AlertDispatcher.dispatch(alert_event)` is the
only integration point. `ConsoleDispatcher` (default) logs JSON;
`WebhookDispatcher` (`--webhook-url`) POSTs the same JSON.

## ROI config

`--roi-config` points to a JSON file:

```json
{"gates": {"<gate-id>": [[x1, y1], [x2, y2], ...]}}
```

Omit it to run sleep detection only (vacant-post logic is skipped with
a logged warning).

## Tests

```
pytest
```

`tests/test_state_machine.py` and `tests/test_calibration.py` are pure
Python (no OpenCV/Ultralytics import); `tests/test_features.py` uses
synthetic keypoint arrays. No YOLO integration tests -- too flaky in CI.
