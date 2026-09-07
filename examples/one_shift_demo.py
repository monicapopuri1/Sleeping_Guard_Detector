"""Runs one full shift of guard-monitor against a sample video.

Edit SOURCE / GATE_ID / OUT_DIR below, then:

    python examples/one_shift_demo.py

This is a thin convenience wrapper -- it just builds the same argv the
real deployment uses and hands it to guard_monitor's own CLI parser, so
`python -m guard_monitor --source ... --gate-id ...` (see README) stays
the one true entrypoint.
"""

import sys

SOURCE = "sample_shift.mp4"
GATE_ID = "gate-a"
OUT_DIR = "./out"
ROI_CONFIG = "config/roi.example.json"


def main() -> None:
    sys.argv = [
        "guard_monitor",
        "--source", SOURCE,
        "--gate-id", GATE_ID,
        "--out-dir", OUT_DIR,
        "--roi-config", ROI_CONFIG,
        "-v",
    ]
    from guard_monitor.__main__ import main as guard_monitor_main

    guard_monitor_main()


if __name__ == "__main__":
    main()
