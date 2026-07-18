"""Live scream-detector calibration — RUN THIS IN YOUR TERMINAL (mic access).

Opens your mic and prints the AST AudioSet distress score every ~0.5 s, plus the
single label the model thinks it hears. Talk, clap, play music -> the distress bar
stays near 0. Actually scream -> it spikes and you'll see `>>> SCREAM DETECTED`.

    .venv/bin/python scripts/test_scream.py            # runs until Ctrl+C
    VIGIL_SCREAM_THRESHOLD=0.25 .venv/bin/python scripts/test_scream.py

macOS: your terminal needs Microphone permission
(System Settings > Privacy & Security > Microphone), then quit + reopen it.
"""

from __future__ import annotations

import queue
import time

import numpy as np

from vigil.config import settings
from vigil.perception import audio as A

THR = settings.scream_threshold


def bar(x: float, width: int = 28) -> str:
    n = int(np.clip(x, 0, 1) * width)
    return "█" * n + "·" * (width - n)


def main() -> None:
    backend = A.ASTBackend()  # loads from the HF cache (downloaded once)
    print(f"[test] AST ready (device={backend._device}); threshold={THR}")
    print("[test] listening — talk normally, then scream. Ctrl+C to stop.\n")

    import sounddevice as sd

    q: queue.Queue[np.ndarray] = queue.Queue()
    ring = np.zeros(A.WINDOW_SAMPLES, dtype=np.float32)
    since = 0
    recent: list[bool] = []
    last_fire = 0.0

    def cb(indata, frames, tinfo, status):
        q.put(indata[:, 0].copy())

    with sd.InputStream(
        samplerate=A.SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=A.HOP_SAMPLES,
        callback=cb,
    ):
        while True:
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

            rms = float(np.sqrt(np.mean(ring * ring)) + 1e-9)
            if rms < A.ENERGY_GATE_RMS:
                recent = (recent + [False])[-A.CONFIRM_WINDOW :]
                print(f"  quiet   rms={rms:5.3f}  {bar(0)}  0.000", end="\r")
                continue

            dist, top, topp = backend.score_verbose(ring.copy())
            recent = (recent + [dist >= THR])[-A.CONFIRM_WINDOW :]
            hit = ""
            if sum(recent) >= A.CONFIRM_HITS and time.time() - last_fire >= A.COOLDOWN_SEC:
                last_fire = time.time()
                recent = []
                hit = "   >>> SCREAM DETECTED <<<"
            print(f"  distress {bar(dist)} {dist:5.3f}  [hears: {top} {topp:.2f}]{hit}" + " " * 8)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[test] stopped.")
