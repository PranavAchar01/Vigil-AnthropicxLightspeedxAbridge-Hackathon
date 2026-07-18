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
    return bool(
        settings.twilio_account_sid
        and settings.twilio_auth_token
        and settings.twilio_from_number
        and settings.nurse_phone_number
    )


def place_call(to_number: str, spoken_text: str) -> tuple[str, str]:
    """Place a real outbound call that speaks `spoken_text`. Returns (call_sid, status)."""
    twiml = f"<Response><Say>{escape(spoken_text)}</Say></Response>"
    url = "https://twimlets.com/echo?Twiml=" + quote(twiml, safe="")
    r = requests.post(
        f"https://api.twilio.com/2010-04-01/Accounts/{settings.twilio_account_sid}/Calls.json",
        auth=(settings.twilio_account_sid, settings.twilio_auth_token),
        data={"To": to_number, "From": settings.twilio_from_number, "Url": url},
        timeout=20,
    )
    r.raise_for_status()
    d = r.json()
    return str(d.get("sid", "")), str(d.get("status", ""))
