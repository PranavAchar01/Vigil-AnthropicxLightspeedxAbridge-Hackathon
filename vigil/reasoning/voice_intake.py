"""Voice -> text for intake triage.

Turns a spoken patient intake into a transcript we can grade. Speech-to-text goes
through ElevenLabs Scribe over the REST API (reusing the ELEVENLABS_API_KEY already
configured for the outbound calls — no new provider, no new key). Optionally records
a few seconds from the local microphone via sounddevice (already a dependency).
Real audio only — there is no simulated transcript path.
"""

from __future__ import annotations

import logging
import wave
from pathlib import Path

import requests

from vigil.config import settings

log = logging.getLogger("vigil.voice_intake")

BASE = "https://api.elevenlabs.io/v1"
STT_MODEL = "scribe_v1"
SAMPLE_RATE = 16000


def transcribe(
    audio: bytes | str | Path,
    *,
    filename: str = "intake.wav",
    content_type: str = "audio/wav",
) -> str:
    """Transcribe audio bytes (or a file path) to text via ElevenLabs Scribe.

    Accepts any format Scribe supports (wav/mp3/m4a/webm/ogg/flac); the browser
    recorder sends webm/opus, the CLI sends wav.
    """
    settings.require("elevenlabs_api_key")
    if isinstance(audio, (str, Path)):
        filename = Path(audio).name
        audio = Path(audio).read_bytes()

    resp = requests.post(
        f"{BASE}/speech-to-text",
        headers={"xi-api-key": settings.elevenlabs_api_key},
        data={"model_id": STT_MODEL},
        files={"file": (filename, audio, content_type)},
        timeout=120,
    )
    resp.raise_for_status()
    text = (resp.json().get("text") or "").strip()
    log.info("transcribed %d bytes -> %d chars", len(audio), len(text))
    return text


def record_wav(path: str | Path, seconds: float = 20.0, samplerate: int = SAMPLE_RATE) -> Path:
    """Record `seconds` of mono audio from the default mic to a 16-bit WAV file."""
    import sounddevice as sd  # local import: optional at import time

    log.info("recording %.0fs of intake audio…", seconds)
    frames = sd.rec(int(seconds * samplerate), samplerate=samplerate, channels=1, dtype="int16")
    sd.wait()
    path = Path(path)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)  # int16
        w.setframerate(samplerate)
        w.writeframes(frames.tobytes())
    return path
