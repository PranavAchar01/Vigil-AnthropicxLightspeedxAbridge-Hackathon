"""Guided scream calibration for YOUR mic + room noise — RUN IN YOUR TERMINAL.

Walks you through four phases (QUIET -> TALK -> BACKGROUND NOISE -> SCREAM),
scoring the AST distress model on each and logging every window so the threshold
can be set in the REAL gap between your screams and your room noise (instead of
guessing). Also measures an EXPANDED distress-class set, so we can tell whether
adding classes would catch your scream better.

Writes /tmp/vigil_scream_calib.jsonl (read back for tuning) and prints a summary
with a recommended VIGIL_SCREAM_THRESHOLD.

    uv run python scripts/calibrate_scream.py

macOS: your terminal needs Microphone permission (System Settings > Privacy &
Security > Microphone), then quit + reopen it.
"""

from __future__ import annotations

import json
import queue
import statistics as st
import time

import numpy as np

from vigil.perception import audio as A

LOG = "/tmp/vigil_scream_calib.jsonl"

# Loudness-backend params — keep in sync with .env (VIGIL_SCREAM_RMS_LO/HI, THRESHOLD).
RMS_LO, RMS_HI = 0.06, 0.16
THR = 0.30

# Expanded human-distress AudioSet classes to ALSO measure (superset of the live set).
EXPANDED = (
    "Screaming", "Shout", "Yell", "Bellow", "Whoop", "Children shouting", "Battle cry",
    "Wail, moan", "Crying, sobbing", "Groan", "Whimper", "Roar",
)

# (phase key, seconds, on-screen instruction)
PHASES = [
    ("ambient", 7.0, "Just your ROOM AS-IS — the background noise you already have, no talking"),
    ("talk", 6.0, "TALK normally, conversation volume"),
    ("scream", 12.0, "SCREAM the way you'd really test it, close to the mic — several bursts"),
]


def bar(x: float, w: int = 26) -> str:
    n = int(np.clip(x, 0, 1) * w)
    return "█" * n + "·" * (w - n)


def main() -> None:
    try:
        b = A.ASTBackend()
        name2idx = {n: int(i) for i, n in b._top.items()}
        narrow_idx = list(b._idx)
        exp_idx = [name2idx[n] for n in EXPANDED if n in name2idx]
        print(f"[calib] AST ready (device={b._device}) — logging AST distress for reference too.")
    except Exception as e:  # noqa: BLE001 — transformers optional; loud mode is what we tune
        b, narrow_idx, exp_idx = None, [], []
        print(f"[calib] AST unavailable ({type(e).__name__}) — loudness-only calibration (fine).")
    print(f"[calib] Logging to {LOG}")
    print("[calib] Follow each prompt; there's a 3-2-1 countdown before each phase.\n")

    import sounddevice as sd

    q: queue.Queue[np.ndarray] = queue.Queue()

    def cb(indata, frames, tinfo, status):
        q.put(indata[:, 0].copy())

    ring = np.zeros(A.WINDOW_SAMPLES, dtype=np.float32)
    since = 0
    rows: list[dict] = []

    with open(LOG, "w") as fh, sd.InputStream(
        samplerate=A.SAMPLE_RATE, channels=1, dtype="float32",
        blocksize=A.HOP_SAMPLES, callback=cb,
    ):
        for phase, dur, instr in PHASES:
            print(f"\n=== {phase.upper()}  ({dur:.0f}s) — {instr} ===")
            for c in (3, 2, 1):
                print(f"   starting in {c}...", end="\r", flush=True)
                time.sleep(1.0)
            print("   GO" + " " * 24)
            t_end = time.time() + dur
            while time.time() < t_end:
                try:
                    block = q.get(timeout=0.5)
                except queue.Empty:
                    continue
                n = len(block)
                ring = np.roll(ring, -n)
                ring[-n:] = block
                since += n
                if since < A.HOP_SAMPLES:
                    continue
                since = 0

                w = ring.copy()
                rms = float(np.sqrt(np.mean(w * w)) + 1e-9)
                # spectral centroid (brightness / rough pitch) — screams are brighter
                mag = np.abs(np.fft.rfft(w * np.hanning(len(w))))
                freqs = np.fft.rfftfreq(len(w), 1.0 / A.SAMPLE_RATE)
                centroid = float((freqs * mag).sum() / (mag.sum() + 1e-9))
                loud = float(np.clip((rms - RMS_LO) / (RMS_HI - RMS_LO), 0.0, 1.0))  # loud-backend score
                if b is not None:
                    p = b._probs(w).detach().cpu().numpy()
                    narrow = float(p[narrow_idx].max())
                    expanded = float(p[exp_idx].max())
                    top5i = np.argsort(p)[-5:][::-1]
                    top5 = [[b._top.get(int(i), str(int(i))), round(float(p[int(i)]), 3)] for i in top5i]
                else:
                    narrow, expanded, top5 = 0.0, 0.0, []

                row = {
                    "phase": phase, "t": round(time.time(), 2), "rms": round(rms, 4),
                    "centroid": round(centroid, 1), "loud": round(loud, 3),
                    "narrow": round(narrow, 4), "expanded": round(expanded, 4), "top5": top5,
                }
                rows.append(row)
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                fired = "  >>> SCREAM (loud) <<<" if loud >= THR else ""
                print(f"  {phase:8s} {bar(loud)} loud={loud:5.3f} rms={rms:5.3f} "
                      f"cent={centroid:6.0f}Hz  (AST={narrow:.3f}){fired}" + " " * 4)

    _summary(rows)


def _stats(vals: list[float]) -> str:
    if not vals:
        return "n/a"
    return f"min={min(vals):.3f} med={st.median(vals):.3f} max={max(vals):.3f}"


def _summary(rows: list[dict]) -> None:
    print("\n" + "=" * 66 + "\n  CALIBRATION SUMMARY (loudness backend)\n" + "=" * 66)
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(r["phase"], []).append(r)
    for ph in ("ambient", "talk", "scream"):
        rs = by.get(ph, [])
        print(f"  {ph:8s} loud[{_stats([r['loud'] for r in rs])}]  "
              f"rms[{_stats([r['rms'] for r in rs])}]  "
              f"cent[{_stats([r['centroid'] for r in rs])}]  AST[{_stats([r['narrow'] for r in rs])}]")

    ns_loud = [r["loud"] for ph in ("ambient", "talk") for r in by.get(ph, [])]
    sc_loud = [r["loud"] for r in by.get("scream", [])]
    ns_cent = [r["centroid"] for ph in ("ambient", "talk") for r in by.get(ph, [])]
    sc_cent = [r["centroid"] for r in by.get("scream", [])]
    if ns_loud and sc_loud:
        print("-" * 66)
        fired = sum(v >= THR for v in sc_loud)
        false_fire = sum(v >= THR for v in ns_loud)
        print(f"  at THRESHOLD={THR}:  screams firing {fired}/{len(sc_loud)}   "
              f"false-fires on talk/ambient {false_fire}/{len(ns_loud)}")
        if sc_cent and ns_cent and st.median(sc_cent) > max(ns_cent):
            gate = round((max(ns_cent) + st.median(sc_cent)) / 2, 0)
            print(f"  screams are brighter (cent {st.median(sc_cent):.0f}Hz vs talk max "
                  f"{max(ns_cent):.0f}Hz) -> can set VIGIL_SCREAM_CENT_LO={gate:.0f} for a pitch gate")
    print(f"\n  full log: {LOG}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[calib] stopped early (partial log saved).")
