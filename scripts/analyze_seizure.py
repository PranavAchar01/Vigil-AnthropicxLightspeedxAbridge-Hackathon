"""Run the real vision detector over seizure video(s) and dump the seizure signal
per frame, so thresholds can be calibrated against actual footage (recall) while we
keep the false-positive guards (precision).

Measures, per frame: tremor-EMA energy, keypoint reversals, motion EMA, FSM state,
person-detected, and every 'seizure' (or other) event the detector fires.

    uv run python scripts/analyze_seizure.py /path/a.mov /path/b.mov ...
"""

from __future__ import annotations

import statistics as st
import sys

import numpy as np

from vigil.events import PerceptionEvent
from vigil.perception.vision import VisionDetector


def analyze(path: str) -> dict:
    import cv2

    events: list[tuple[float, str, float]] = []
    det = VisionDetector(emit=lambda e: events.append((e.ts, e.kind, e.confidence)))

    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    rows = []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        ts = i / fps
        det.process(frame, ts=ts)
        m = det.metrics
        rows.append({
            "t": round(ts, 2),
            "present": bool(m.get("present")),
            "tremor": float(m.get("tremor", 0.0)),
            "reversals": int(m.get("reversals", 0)),
            "motion": float(m.get("motion", 0.0)),
            "state": det.st.fsm,
        })
        i += 1
    cap.release()

    name = path.split("/")[-1]
    seiz = [e for e in events if e[1] == "seizure"]
    present = [r for r in rows if r["present"]]
    trem = [r["tremor"] for r in present]
    rev = [r["reversals"] for r in present]
    mot = [r["motion"] for r in present]

    def s(v):
        return f"min={min(v):.3f} med={st.median(v):.3f} p90={np.percentile(v,90):.3f} max={max(v):.3f}" if v else "n/a"

    print(f"\n=== {name} ({i} frames, {i/fps:.1f}s) ===")
    print(f"  person detected: {len(present)}/{i} frames")
    print(f"  tremor-EMA energy : {s(trem)}")
    print(f"  reversals/window  : {s(rev)}")
    print(f"  motion EMA        : {s(mot)}")
    fired = "  ".join(f"{t:.1f}s(conf {c:.2f})" for t, k, c in seiz) or "NONE"
    print(f"  >>> seizure fired : {fired}")
    other = sorted({k for _, k, _ in events if k != "seizure"})
    if other:
        print(f"  (other events fired: {other})")
    return {"name": name, "fired": len(seiz) > 0, "trem": trem, "rev": rev, "n": i}


def main() -> None:
    paths = sys.argv[1:]
    if not paths:
        sys.exit("usage: analyze_seizure.py video1 [video2 ...]")
    results = [analyze(p) for p in paths]
    hit = sum(r["fired"] for r in results)
    print("\n" + "=" * 60)
    print(f"RECALL on current thresholds: {hit}/{len(results)} videos fired 'seizure'")
    all_trem = [t for r in results for t in r["trem"]]
    if all_trem:
        print(f"tremor energy across all footage: median={st.median(all_trem):.3f} "
              f"p90={np.percentile(all_trem,90):.3f} max={max(all_trem):.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
