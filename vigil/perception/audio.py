"""Real-time scream / human-distress detection from the laptop mic.

Backend: the Audio Spectrogram Transformer (AST) fine-tuned on AudioSet — a real
learned audio-event classifier (SOTA open-source on AudioSet), NOT a loudness
heuristic. It scores 527 AudioSet classes; we watch the human-distress family
(Screaming, Shout, Yell, Bellow, Whoop, Children shouting, Battle cry). It runs on
the same torch that YOLO already uses — no TensorFlow, no torchaudio.

Two guards keep it honest so it never "treats anything loud as a scream":
  * an energy gate skips near-silence, so the model only runs on real sound; and
  * a 2-of-3-window confirmation rejects single-frame false positives.
Emits PerceptionEvents(kind="scream") through the sink.

macOS: the terminal/IDE running Python needs Microphone permission
(System Settings > Privacy & Security > Microphone) or the mic reads all zeros.
"""

from __future__ import annotations

import os
import queue
import threading
import time
from collections import deque
from typing import Callable

import numpy as np

from vigil.events import Modality, PerceptionEvent

SAMPLE_RATE = 16_000
WINDOW_SAMPLES = 32_000  # 2.0 s of audio fed to AST (padded internally to its frame len)
HOP_SAMPLES = 8_000  # 0.5 s -> ~2 inferences / second
COOLDOWN_SEC = 2.0
ENERGY_GATE_RMS = 0.006  # below this the room is quiet -> don't even run the model
CONFIRM_WINDOW = 3  # look back over the last N scored windows
CONFIRM_HITS = 2  # fire only if >= this many crossed the threshold (kills transients)

# AudioSet human-distress display_names we treat as a "scream" hard signal
# (matched by name against the model's label map, never a hardcoded index).
DISTRESS = ("Screaming", "Shout", "Yell", "Bellow", "Whoop", "Children shouting", "Battle cry")

MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"

Sink = Callable[[PerceptionEvent], None]


def _envf(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


class ASTBackend:
    """AST (AudioSet) distress classifier. score() -> P(distress) in [0, 1]."""

    THRESHOLD = 0.30  # sigmoid prob of the loudest distress class; non-screams score ~0.001

    def __init__(self) -> None:
        import torch
        from transformers import ASTFeatureExtractor, ASTForAudioClassification

        self._torch = torch
        want = os.environ.get("VIGIL_AUDIO_DEVICE", "cpu").lower()
        self._device = "mps" if (want == "mps" and torch.backends.mps.is_available()) else "cpu"
        self._fe = ASTFeatureExtractor.from_pretrained(MODEL_ID)
        self._model = ASTForAudioClassification.from_pretrained(MODEL_ID).to(self._device).eval()

        id2label = self._model.config.id2label
        self._idx = [int(i) for i, n in id2label.items() if n in DISTRESS]
        if not self._idx:
            raise RuntimeError("AST label map has no distress classes")
        self._top = {int(i): n for i, n in id2label.items()}

    def _probs(self, window: np.ndarray):
        torch = self._torch
        with torch.no_grad():
            x = self._fe(window, sampling_rate=SAMPLE_RATE, return_tensors="pt").to(self._device)
            return torch.sigmoid(self._model(**x).logits)[0]

    def score(self, window: np.ndarray) -> float:
        return float(self._probs(window)[self._idx].max())

    def score_verbose(self, window: np.ndarray) -> tuple[float, str, float]:
        """(distress_score, top_label_name, top_label_prob) — for live tuning/debug."""
        probs = self._probs(window)
        distress = float(probs[self._idx].max())
        top = int(probs.argmax())
        return distress, self._top.get(top, str(top)), float(probs[top])


class HeuristicBackend:
    """Last-resort loud-AND-high-frequency detector. OFF by default: the AST model
    is the real detector. Enable only via VIGIL_ALLOW_HEURISTIC_AUDIO=1 when the
    model can't be loaded — it is deliberately conservative to avoid false pages."""

    THRESHOLD = 0.80

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
        loud = float(np.clip((ratio - 6.0) / 8.0, 0.0, 1.0))
        bright = float(np.clip((centroid - 1500.0) / 2000.0, 0.0, 1.0))
        return loud * bright


class LoudPitchBackend:
    """Absolute-loudness (+ optional pitch) scream detector, tuned to ONE mic.

    Use this when the AST model doesn't recognize a given user's screams (it labels
    them 'Speech', scoring the distress classes ~0). Verified on-device: close-mic
    screaming is ~5x louder than talk/ambient (RMS ~0.17 vs ~0.03), so an absolute
    RMS ramp separates them cleanly. An optional spectral-centroid gate rejects loud
    non-scream sounds. score() -> confidence in [0, 1]. Enable with VIGIL_SCREAM_MODE=loud.
    """

    THRESHOLD = 0.30

    def __init__(self) -> None:
        self._lo = _envf("VIGIL_SCREAM_RMS_LO", 0.06)  # RMS at/below -> 0 (below loud talk)
        self._hi = _envf("VIGIL_SCREAM_RMS_HI", 0.16)  # RMS at/above -> 1 (solid scream)
        self._cent_lo = _envf("VIGIL_SCREAM_CENT_LO", 0.0)  # Hz pitch gate; 0 = disabled

    def _feats(self, window: np.ndarray) -> tuple[float, float]:
        w = window.astype(np.float32)
        rms = float(np.sqrt(np.mean(w * w)) + 1e-9)
        mag = np.abs(np.fft.rfft(w * np.hanning(len(w))))
        freqs = np.fft.rfftfreq(len(w), 1.0 / SAMPLE_RATE)
        centroid = float((freqs * mag).sum() / (mag.sum() + 1e-9))
        return rms, centroid

    def score(self, window: np.ndarray) -> float:
        rms, centroid = self._feats(window)
        loud = float(np.clip((rms - self._lo) / (self._hi - self._lo + 1e-9), 0.0, 1.0))
        if self._cent_lo > 0.0 and centroid < self._cent_lo:
            loud *= 0.3  # loud but not bright enough to be a scream -> suppress
        return loud

    def score_verbose(self, window: np.ndarray) -> tuple[float, str, float]:
        rms, centroid = self._feats(window)
        s = self.score(window)
        return s, f"loud(rms={rms:.3f} cent={centroid:.0f}Hz)", s


def build_backend():
    """AST is the default learned backend. VIGIL_SCREAM_MODE=loud forces the absolute
    loudness backend (for mics where AST mislabels screams as speech). The old
    adaptive heuristic remains as a last resort when AST fails to load."""
    mode = os.environ.get("VIGIL_SCREAM_MODE", "ast").lower()
    if mode == "loud":
        print("[audio] loudness scream backend active (VIGIL_SCREAM_MODE=loud)")
        return LoudPitchBackend()
    try:
        b = ASTBackend()
        print(f"[audio] AST AudioSet scream classifier ready (device={b._device})")
        return b
    except Exception as e:  # noqa: BLE001 — import/download/load failure
        if os.environ.get("VIGIL_ALLOW_HEURISTIC_AUDIO") == "1":
            print(f"[audio] AST unavailable ({e!r}); using conservative heuristic fallback")
            return HeuristicBackend()
        print(
            f"[audio] AST scream model unavailable ({e!r}); audio detection DISABLED.\n"
            f"        Enable it with:  uv pip install transformers   (torch is already present)"
        )
        return None


class ScreamDetector:
    def __init__(self, sink: Sink, threshold: float | None = None) -> None:
        self._sink = sink
        self._backend = build_backend()
        self._threshold = (
            threshold if threshold is not None else getattr(self._backend, "THRESHOLD", 0.30)
        )
        self._q: queue.Queue[np.ndarray] = queue.Queue()
        self._ring = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
        self._since = 0
        self._last_fire = 0.0
        self._recent: deque[bool] = deque(maxlen=CONFIRM_WINDOW)

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

            window = self._ring.copy()
            # Energy gate: a scream is loud. Skip the model on a quiet room — saves
            # CPU and guarantees silence can never score as distress.
            rms = float(np.sqrt(np.mean(window * window)) + 1e-9)
            if rms < ENERGY_GATE_RMS:
                self._recent.append(False)
                continue

            conf = self._backend.score(window)
            self._recent.append(conf >= self._threshold)
            # Confirm across windows: a real scream sustains; a click does not.
            if sum(self._recent) >= CONFIRM_HITS:
                now = time.time()
                if now - self._last_fire >= COOLDOWN_SEC:
                    self._last_fire = now
                    self._recent.clear()
                    self._sink(
                        PerceptionEvent(
                            ts=now,
                            modality=Modality.AUDIO,
                            kind="scream",
                            confidence=round(conf, 3),
                        )
                    )

    def run(self, stop_event) -> None:
        if self._backend is None:
            return  # audio disabled; build_backend already explained why
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
