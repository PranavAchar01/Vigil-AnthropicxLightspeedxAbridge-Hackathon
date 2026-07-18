"""Real-time scream / human-distress detection from the laptop mic.

Primary backend : Google YAMNet (AudioSet 521 classes) via TF-Hub SavedModel.
Fallback backend: RMS energy spike x spectral centroid (pure NumPy, no ML deps),
                  so the demo still works if TensorFlow isn't installed / fails.
Emits PerceptionEvents(kind="scream") through the sink.

macOS: the terminal/IDE running Python needs Microphone permission
(System Settings > Privacy & Security > Microphone) or the mic reads all zeros.
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Callable

import numpy as np

from vigil.events import Modality, PerceptionEvent

SAMPLE_RATE = 16_000
WINDOW_SAMPLES = 15_600  # ~0.975 s -> exactly one YAMNet frame
HOP_SAMPLES = 7_680  # ~0.48 s -> ~2 inferences / second
COOLDOWN_SEC = 1.5

# AudioSet distress display_names (matched by name, not hardcoded index).
DISTRESS = {"Screaming", "Shout", "Yell", "Bellow", "Whoop", "Children shouting"}

Sink = Callable[[PerceptionEvent], None]


class YamnetBackend:
    THRESHOLD = 0.35

    def __init__(self) -> None:
        import csv

        import tensorflow as tf
        import tensorflow_hub as hub

        self._m = hub.load("https://tfhub.dev/google/yamnet/1")  # Keras-free path
        with tf.io.gfile.GFile(self._m.class_map_path().numpy()) as f:
            names = [r["display_name"] for r in csv.DictReader(f)]
        self._idx = np.array([i for i, n in enumerate(names) if n in DISTRESS], dtype=int)

    def score(self, window: np.ndarray) -> float:
        scores, _emb, _spec = self._m(window)
        s = scores.numpy().mean(axis=0)
        return float(s[self._idx].max()) if self._idx.size else 0.0


class HeuristicBackend:
    """Loud AND high-frequency -> scream-like. No ML deps."""

    THRESHOLD = 0.60

    def __init__(self) -> None:
        self._floor = 1e-3
        self._alpha = 0.05

    def score(self, window: np.ndarray) -> float:
        w = window.astype(np.float32)
        rms = float(np.sqrt(np.mean(w * w)) + 1e-9)
        mag = np.abs(np.fft.rfft(w * np.hanning(len(w))))
        freqs = np.fft.rfftfreq(len(w), 1.0 / SAMPLE_RATE)
        centroid = float((freqs * mag).sum() / (mag.sum() + 1e-9))
        ratio = rms / (self._floor + 1e-9)
        if ratio < 3.0:  # only adapt the noise floor when it's quiet
            self._floor = (1 - self._alpha) * self._floor + self._alpha * rms
        loud = float(np.clip((ratio - 4.0) / 8.0, 0.0, 1.0))
        bright = float(np.clip((centroid - 1200.0) / 2000.0, 0.0, 1.0))
        return loud * bright


def build_backend():
    try:
        b = YamnetBackend()
        print("[audio] YAMNet backend ready")
        return b
    except Exception as e:  # noqa: BLE001 — any import/download/TF failure -> fallback
        print(f"[audio] YAMNet unavailable ({e!r}); using heuristic fallback")
        return HeuristicBackend()


class ScreamDetector:
    def __init__(self, sink: Sink, threshold: float | None = None) -> None:
        self._sink = sink
        self._backend = build_backend()
        self._threshold = threshold if threshold is not None else self._backend.THRESHOLD
        self._q: queue.Queue[np.ndarray] = queue.Queue()
        self._ring = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
        self._since = 0
        self._last_fire = 0.0

    def _cb(self, indata, frames, time_info, status):  # PortAudio thread — keep cheap
        self._q.put(indata[:, 0].copy())

    def _consume(self, stop_event) -> None:
        while not stop_event.is_set():
            try:
                block = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            n = len(block)
            self._ring = np.roll(self._ring, -n)
            self._ring[-n:] = block
            self._since += n
            if self._since < HOP_SAMPLES:
                continue
            self._since = 0
            conf = self._backend.score(self._ring.copy())
            if conf >= self._threshold:
                now = time.time()
                if now - self._last_fire >= COOLDOWN_SEC:
                    self._last_fire = now
                    self._sink(
                        PerceptionEvent(
                            ts=now,
                            modality=Modality.AUDIO,
                            kind="scream",
                            confidence=round(conf, 3),
                        )
                    )

    def run(self, stop_event) -> None:
        import sounddevice as sd

        threading.Thread(target=self._consume, args=(stop_event,), daemon=True).start()
        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=HOP_SAMPLES,
                callback=self._cb,
            ):
                print("[audio] listening for distress vocalizations")
                while not stop_event.is_set():
                    time.sleep(0.1)
        except Exception as e:  # noqa: BLE001 — no mic / permission denied
            print(f"[audio] microphone unavailable ({e!r}); audio disabled")


def run_audio(sink: Sink, stop_event, threshold: float | None = None) -> None:
    """Blocking; run in a daemon thread."""
    ScreamDetector(sink, threshold=threshold).run(stop_event)
