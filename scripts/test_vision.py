"""Live vision-detector calibration — RUN THIS IN YOUR TERMINAL (camera access).

Opens your webcam, runs the exact distress detector the server uses, and prints a
live metric line plus every event it fires. Do each action and watch it register:

    • lean/drop fast toward the floor          -> FALL
    • slump your head down, then hold still     -> COLLAPSE
    • shake / jerk quickly side to side         -> SEIZURE
    • sit motionless ~10 s                       -> UNRESPONSIVE
    • put a hand to your chest/throat ~2 s       -> CHEST_CLUTCH
    • lean hard to one side ~5 s                 -> SLUMP

    .venv/bin/python scripts/test_vision.py

The metric line shows what the detector sees so you can tune (every threshold is an
env var, e.g. VIGIL_DROP_FRAC, VIGIL_STILL_MOTION, VIGIL_FAINT_S — see vision.py).
macOS: your terminal needs Camera permission (run scripts/grant_camera.py once).
Ctrl+C to stop.
"""

from __future__ import annotations

import time

from vigil.perception.vision import VisionDetector


def main() -> None:
    import cv2

    fired: list = []
    det = VisionDetector(emit=fired.append, emit_status=lambda *a: None)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[test] could not open camera. Run scripts/grant_camera.py, then retry.")
        return
    print("[test] camera open — act out the scenarios above. Ctrl+C to stop.\n")

    last_print = 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.03)
            continue
        det.process(frame, ts=time.time())

        for e in fired:
            print(f"\n  >>> FIRED: {e.kind.upper()}  (conf {e.confidence})\n")
        fired.clear()

        now = time.time()
        if now - last_print >= 0.25:
            last_print = now
            m = det.metrics
            if m.get("present"):
                print(
                    f"  {m.get('state', '?'):7} motion {m.get('motion', 0):5.2f}  "
                    f"drop {m.get('drop', 0):5.2f}  vy {m.get('vy', 0):+5.2f}  "
                    f"reversals {m.get('reversals', 0):2d}  still {m.get('still_s', 0):4.1f}s"
                    + " "
                    * 4,
                    end="\r",
                )
            else:
                print("  no person in frame" + " " * 40, end="\r")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[test] stopped.")
