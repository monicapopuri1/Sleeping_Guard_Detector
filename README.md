# guard-monitor

Real-time vacant-post and sleeping-guard detection from a single CCTV
stream per process. Pre-trained YOLO11n-pose + BoT-SORT only -- no
training, no labels.

## Quick start

```
pip install -r requirements.txt
python -m guard_monitor --source /path/to/video.mp4 --gate-id gate-a --out-dir ./out
```

`--source` accepts an `rtsp://` URL, a video file path, or a webcam
index (`0`). First run auto-downloads `yolo11n-pose.pt`; results land
in `./out/events.csv` and `./out/alerts/gate-a/<timestamp>_track<id>/`.

## Running against a video file

```
python -m guard_monitor \
  --source /path/to/video.mp4 \
  --gate-id gate-a \
  --out-dir ./out \
  --pose-weights /path/to/yolo11n-pose.pt \
  --device mps \
  --annotate
```

- `--pose-weights` points at a local `yolo11n-pose.pt` if you already
  have one (skips the auto-download). `--person-weights` is the
  equivalent for `yolo11n.pt`, used only when `--roi-config` is set.
- `--device` is `cpu` by default; use `mps` on Apple Silicon or
  `cuda:0` on an Nvidia GPU for a large speedup.
- `--annotate` writes `./out/annotated/gate-a.mp4` -- bbox, skeleton,
  and per-track state (ACTIVE/STILL/SLEEPING_SUSPECT/ALERT/...)
  overlaid on every frame, plus a HUD line with motion/tilt/lean/
  evidence -- so you can actually watch the detector work. It costs
  extra encode time; drop it for faster headless runs.
- `--infer-fps 3` (or similar) caps pose inference to that many times
  per second instead of every frame -- sleep-detection dwell times are
  tens of seconds long, so this cuts runtime substantially (~9-10x on
  a 5-minute clip in testing) with no measured accuracy loss. Omit it
  to process every frame (the default).
- Add `--roi-config config/roi.example.json` (with real polygon
  coordinates for your camera) to also enable vacant-post detection;
  omit it to run sleep detection only.

Every alert -- including re-alerts for a guard who stays asleep --
appends a row to `./out/events.csv` and writes
`./out/alerts/<gate-id>/<timestamp>_track<id>/{clip.mp4, rows.csv,
meta.json}`.

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
