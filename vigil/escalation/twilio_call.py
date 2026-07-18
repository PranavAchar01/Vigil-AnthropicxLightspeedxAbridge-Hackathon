"""Direct outbound call to the nurse via the Twilio REST API.

Used when the ElevenLabs<->Twilio integration isn't available (e.g. a Twilio
trial account, where importing a number into ElevenLabs fails with a policy
error but a direct call still works). The escalation summary is spoken with
Twilio TTS via a twimlets `echo` URL — trial accounts allow a `Url` parameter
but not inline `Twiml`.
"""

from __future__ import annotations

import logging
from urllib.parse import quote
from xml.sax.saxutils import escape

import requests

from vigil.config import settings

log = logging.getLogger("vigil.escalation.twilio")


def twilio_call_configured() -> bool:
    # Auth is satisfied by either an Account Auth Token or a scoped API Key + Secret.
    has_auth = bool(settings.twilio_auth_token) or bool(
        settings.twilio_api_key_sid and settings.twilio_api_key_secret
    )
    return bool(
        settings.twilio_account_sid
        and has_auth
        and settings.twilio_from_number
        and settings.nurse_phone_number
    )


def place_call(to_number: str, spoken_text: str) -> tuple[str, str]:
    """Place a real outbound call that speaks `spoken_text`. Returns (call_sid, status).

    Uses a natural Amazon Polly neural voice (VIGIL_TWILIO_TTS_VOICE) instead of
    Twilio's robotic default — this is the fallback path when the OpenAI Realtime
    voice bridge isn't up (no public tunnel)."""
    voice = settings.twilio_tts_voice
    voice_attr = f' voice="{escape(voice, {chr(34): "&quot;"})}"' if voice else ""
    twiml = f"<Response><Say{voice_attr}>{escape(spoken_text)}</Say></Response>"
    url = "https://twimlets.com/echo?Twiml=" + quote(twiml, safe="")
    # Prefer a scoped API Key (SK...) + Secret for auth; fall back to the Account
    # Auth Token. Either way the REST path is scoped by the Account SID (AC...).
    if settings.twilio_api_key_sid and settings.twilio_api_key_secret:
        auth = (settings.twilio_api_key_sid, settings.twilio_api_key_secret)
    else:
        auth = (settings.twilio_account_sid, settings.twilio_auth_token)
    r = requests.post(
        f"https://api.twilio.com/2010-04-01/Accounts/{settings.twilio_account_sid}/Calls.json",
        auth=auth,
        data={"To": to_number, "From": settings.twilio_from_number, "Url": url},
        timeout=20,
    )
    r.raise_for_status()
    d = r.json()
    return str(d.get("sid", "")), str(d.get("status", ""))
