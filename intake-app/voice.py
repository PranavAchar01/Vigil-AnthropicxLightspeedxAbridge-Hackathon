"""Speech-to-text via ElevenLabs Scribe — self-contained. Needs ELEVENLABS_API_KEY."""

from __future__ import annotations

import os

import requests

BASE = "https://api.elevenlabs.io/v1"
STT_MODEL = "scribe_v1"


def transcribe(
    audio: bytes, *, filename: str = "intake.webm", content_type: str = "audio/webm"
) -> str:
    key = os.environ.get("ELEVENLABS_API_KEY", "")
    if not key:
        raise RuntimeError("ELEVENLABS_API_KEY is not set.")
    resp = requests.post(
        f"{BASE}/speech-to-text",
        headers={"xi-api-key": key},
        data={"model_id": STT_MODEL},
        files={"file": (filename, audio, content_type)},
        timeout=120,
    )
    resp.raise_for_status()
    return (resp.json().get("text") or "").strip()
