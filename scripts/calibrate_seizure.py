"""Calibrate seizure detection to YOUR webcam — run while the server is up.

    uv run uvicorn vigil.server.app:app --port 8000     # terminal 1 (camera)
    .venv/bin/python scripts/calibrate_seizure.py       # terminal 2 (this)

Two guided phases in front of the camera: (1) behave NORMALLY ~30s, (2) perform the
seizure shake ~12s. The script reads the live detector's per-frame spectral metrics
from the server, finds the amplitude/concentration thresholds with the widest margin
between the two, VERIFIES them offline against both recordings, and promotes them to
config/tuned.env. Restart the server afterwards to load the promoted values.

NOTE: phase 2 may genuinely fire a seizure event -> Claude re-triage -> ONE real
nurse call (per-issue dedup caps it). That is the system working.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import requests

BASE = "http://127.0.0.1:8000"
SUSTAIN_S = 1.2  # the detector's osc_sustain_s — thresholds are chosen around it


def fetch_since(ts: float) -> list[dict]:
    r = requests.get(f"{BASE}/calibrate/metrics", params={"since": ts}, timeout=10)
    r.raise_for_status()
    return r.json()["rows"]


def countdown(label: str, seconds: int) -> tuple[float, list[dict]]:
    input(f"\n>>> {label}\n    Press ENTER to start the {seconds}s capture...")
    t0 = time.time()
    for remaining in range(seconds, 0, -1):
        print(f"    capturing... {remaining:2d}s", end="\r", flush=True)
        time.sleep(1)
    print()
    rows = fetch_since(t0 - 0.5)
    rows = [r for r in rows if r["ts"] <= time.time() + 0.5]
    if len(rows) < seconds * 5:
        sys.exit(
            f"only {len(rows)} frames captured (expected >= {seconds * 5}). "
            "Is the camera live on the server (STREAMING on the dashboard)?"
        )
    return t0, rows


def longest_run_s(rows: list[dict], amp: float, conc: float) -> float:
    best = run_start = 0.0
    prev_ok = False
    for r in rows:
        ok = r["osc_amp"] >= amp and r["osc_conc"] >= conc
        if ok and not prev_ok:
            run_start = r["ts"]
        if ok:
            best = max(best, r["ts"] - run_start)
        prev_ok = ok
    return best


def main() -> None:
    try:
        requests.get(f"{BASE}/calibrate/metrics", timeout=5).raise_for_status()
    except Exception as e:  # noqa: BLE001
        sys.exit(f"server not reachable on :8000 ({e}) — start it first (see docstring)")

    print("Vigil seizure calibration — two phases in front of the webcam.")
    _, normal = countdown(
        "PHASE 1 / NORMAL: sit or stand as usual — look around, type, talk, sip water.",
        30,
    )
    _, seizure = countdown(
        "PHASE 2 / SEIZURE: perform the demo seizure shake CONTINUOUSLY the whole time.",
        12,
    )

    # grid-search thresholds: normal must never sustain half the window; seizure must
    # sustain well past it. Prefer the candidate with the widest multiplicative margin.
    best = None
    for amp in (0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15):
        for conc in (0.5, 0.6, 0.7, 0.8):
            n_run = longest_run_s(normal, amp, conc)
            s_run = longest_run_s(seizure, amp, conc)
            if s_run < SUSTAIN_S * 1.5 or n_run > SUSTAIN_S * 0.5:
                continue
            margin = s_run / max(n_run, 0.05)
            if best is None or margin > best[0]:
                best = (margin, amp, conc, n_run, s_run)

    print("\n── results ──")
    if best is None:
        print("NO SEPARATION FOUND. Diagnostics:")
        for name, rows in (("normal", normal), ("seizure", seizure)):
            amps = sorted(r["osc_amp"] for r in rows)
            print(
                f"  {name:8s}: frames={len(rows)} amp p50={amps[len(amps) // 2]:.3f} "
                f"max={amps[-1]:.3f} longest@0.05/0.7={longest_run_s(rows, 0.05, 0.7):.2f}s"
            )
        sys.exit(
            "Shake harder/faster (a rhythmic 3-6 Hz tremor) in phase 2, or move less "
            "in phase 1, then re-run."
        )

    margin, amp, conc, n_run, s_run = best
    print(f"  chosen: amp>={amp}  conc>={conc}")
    print(f"  normal  longest sustained: {n_run:.2f}s  (must stay < {SUSTAIN_S}s)")
    print(f"  seizure longest sustained: {s_run:.2f}s  (fires after {SUSTAIN_S}s)")
    print(f"  separation margin: {margin:.1f}x")

    tuned = Path(__file__).resolve().parent.parent / "config" / "tuned.env"
    lines = [
        ln
        for ln in tuned.read_text().splitlines()
        if not ln.startswith(("VIGIL_SEIZURE_OSC_AMP=", "VIGIL_SEIZURE_OSC_CONC="))
    ]
    lines += [f"VIGIL_SEIZURE_OSC_AMP={amp}", f"VIGIL_SEIZURE_OSC_CONC={conc}"]
    tuned.write_text("\n".join(lines) + "\n")
    print(f"\npromoted to {tuned.name} — RESTART the server to load them, then the")
    print("demo shake will fire 'seizure' within ~2s, and normal behavior never will.")


if __name__ == "__main__":
    main()
