"""Entrypoint: python -m guard_monitor --source <rtsp|file|0> --gate-id <name>"""

from __future__ import annotations

import argparse
import logging

from guard_monitor.config import add_config_args, config_from_args
from guard_monitor import runner


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m guard_monitor",
        description="Real-time vacant-post and sleeping-guard detection for a single CCTV stream.",
    )
    parser.add_argument("--source", required=True, help="rtsp:// URL, video file path, or webcam index (e.g. 0)")
    parser.add_argument("--gate-id", required=True, help="identifier for this gate house, used in output paths and alerts")
    parser.add_argument("--out-dir", default="./out", help="output root; events.csv and alerts/ are written here")
    parser.add_argument("--roi-config", default=None, help="path to ROI JSON ({'gates': {gate_id: [[x,y],...]}}); omit to disable vacant-post logic")
    parser.add_argument("--webhook-url", default=None, help="if set, alerts POST here as JSON instead of going to the console")
    parser.add_argument("--pose-weights", default="yolo11n-pose.pt", help="YOLO11n-pose weights for sleep detection")
    parser.add_argument("--person-weights", default="yolo11n.pt", help="YOLO11n weights for vacant-post person detection")
    parser.add_argument("--device", default="cpu", help="inference device, e.g. cpu, cuda:0")
    parser.add_argument("--annotate", action="store_true", help="write a debug/visualization MP4 with bboxes, skeletons, per-track state, and ROI overlay (does not affect detection)")
    parser.add_argument("--annotate-path", default=None, help="override annotated-video output path (default: <out-dir>/annotated/<gate-id>.mp4)")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="-v info, -vv debug, -vvv includes ultralytics/cv2 debug")
    add_config_args(parser)
    return parser


def _configure_logging(verbosity: int) -> None:
    level = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}.get(verbosity, logging.DEBUG)
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    _configure_logging(args.verbose)
    config = config_from_args(args)
    runner.run(args, config)


if __name__ == "__main__":
    main()
