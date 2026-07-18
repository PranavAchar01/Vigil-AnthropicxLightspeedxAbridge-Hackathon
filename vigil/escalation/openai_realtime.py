"""A REAL conversational voice agent on the phone call, via OpenAI Realtime.

The escalation call (to the charge nurse, or a patient check-in) is a genuine
speech-to-speech conversation — the agent listens, understands, and talks back in
real time, grounded in the patient's chart and the re-triage decision. It can hold
a back-and-forth: answer the nurse's questions, pull live patient status, and take
their acknowledgement.

How it works (the canonical Twilio <-> OpenAI Realtime bridge):

    Twilio call  ──(mu-law 8kHz Media Stream, bidirectional)──►  our /media-stream WS
    our WS       ──(mu-law audio frames)──►  OpenAI Realtime API  ──(mu-law back)──┐
       ▲                                                                            │
       └────────────────────── agent speech streamed back to the caller ───────────┘

OpenAI Realtime speaks G.711 mu-law ("g711_ulaw") natively — the exact codec Twilio
uses — so audio passes straight through with NO resampling or transcoding. The whole
thing is env-gated: set OPENAI_API_KEY (+ a Twilio number + a public URL for Twilio
to reach the stream) and it lights up; otherwise the caller falls back to the
ElevenLabs agent or a one-shot Twilio TTS call.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from urllib.parse import urlparse

import requests

from vigil.config import settings

log = logging.getLogger("vigil.escalation.realtime")

OPENAI_WS = "wss://api.openai.com/v1/realtime"
# G.711 mu-law, 8 kHz — Twilio's telephony codec and an OpenAI Realtime audio format,
# so no transcoding is needed on either leg of the bridge. "audio/pcmu" is the name
# in the GA Realtime schema (the older beta used "g711_ulaw").
AUDIO_FORMAT = "audio/pcmu"

# distress words that make a patient's spoken reply NON-reassuring (shared contract
# with the ElevenLabs check-in path).
DISTRESS_WORDS = ("help", "can't", "cannot breathe", "chest", "pain", "dizzy", "worse")


# --------------------------------------------------------------------------- #
# Configuration gate
# --------------------------------------------------------------------------- #
def realtime_call_configured() -> bool:
    """True when a bridged OpenAI-Realtime phone call can actually be placed:
    an OpenAI key, a Twilio number to dial FROM (+ auth), and a public URL Twilio
    can reach our media stream at."""
    has_twilio_auth = bool(settings.twilio_auth_token) or bool(
        settings.twilio_api_key_sid and settings.twilio_api_key_secret
    )
    return bool(
        settings.openai_api_key
        and settings.twilio_account_sid
        and has_twilio_auth
        and settings.twilio_from_number
        and settings.public_url
    )


# --------------------------------------------------------------------------- #
# Per-call context registry — stashed when a call is placed, read back when
# Twilio connects the media stream (keyed by an opaque id in the stream URL).
# --------------------------------------------------------------------------- #
class CallContext:
    """Everything the bridge needs to run one conversation, plus the channel the
    (threaded) escalation ladder waits on for a patient check-in's outcome."""

    def __init__(self, role: str, fields: dict[str, str]) -> None:
        self.role = role  # "nurse" | "patient"
        self.fields = fields  # patient_name, patient_id, chart_summary, incident, esi
        self.caller_turns: list[str] = []  # what the human said (for check-in evaluation)
        self.agent_turns: list[str] = []
        self.done = threading.Event()  # set when the call/bridge ends


_CTX: dict[str, CallContext] = {}
_CTX_LOCK = threading.Lock()
_CTX_SEQ = 0


def register_context(role: str, fields: dict[str, str]) -> str:
    """Stash a call's context and return its opaque id (goes in the stream URL)."""
    global _CTX_SEQ
    with _CTX_LOCK:
        _CTX_SEQ += 1
        cid = f"c{_CTX_SEQ}-{int(time.monotonic() * 1000) % 100000}"
        _CTX[cid] = CallContext(role, fields)
        # keep the map from growing unbounded across a long-running server
        if len(_CTX) > 64:
            for old in list(_CTX)[:-64]:
                _CTX.pop(old, None)
        return cid


def get_context(cid: str) -> CallContext | None:
    with _CTX_LOCK:
        return _CTX.get(cid)


# --------------------------------------------------------------------------- #
# Persona instructions — what makes it hold a real, grounded conversation
# --------------------------------------------------------------------------- #
def _nurse_instructions(f: dict[str, str]) -> str:
    name = f.get("patient_name", "the patient")
    incident = f.get("incident", "a change in condition")
    esi = f.get("esi", "an updated ESI")
    return (
        "You are Vigil, an automated clinical safety monitor phoning the CHARGE NURSE "
        "about a waiting-room patient whose condition just changed. ALWAYS SPEAK IN "
        "ENGLISH. Speak the way a triage nurse hands off: calm, concise, specific.\n\n"
        "CRITICAL — NEVER FABRICATE. State ONLY the facts given below. Do NOT invent or "
        "infer anything not listed: no room numbers, no vital signs, no mechanism of "
        "injury, no symptoms, no history, no 'stable/unstable' — nothing you were not "
        "given. If the nurse asks for something not in the DATA below, use the "
        "get_patient_status tool if it is a live-status question; otherwise say exactly: "
        "\"I don't have that in the chart.\" You report observations and the model's "
        "re-triage only; you do NOT diagnose or give medical orders.\n\n"
        "Open immediately (do not wait to be prompted) with this line, verbatim except "
        "for the bracketed values:\n"
        f'  "This is Vigil, automated monitoring. Calling about {name}. {incident}. '
        f'Re-triaged to {esi}."\n\n'
        "Then answer questions in 1-2 short sentences using ONLY the DATA below and the "
        "get_patient_status tool. Before ending, confirm the nurse is heading over.\n\n"
        "DATA (the only facts you may state):\n"
        f"- Patient name: {name}\n"
        f"- Chart: {f.get('chart_summary', 'no chart on file')}\n"
        f"- What Vigil observed: {incident}\n"
        f"- Re-triage: {esi}\n"
    )


def _patient_instructions(f: dict[str, str]) -> str:
    name = f.get("patient_name", "there")
    return (
        "You are Vigil, an automated care assistant making a brief, warm check-in call "
        "to a patient in the waiting room. ALWAYS SPEAK IN ENGLISH. You are calling "
        f"{name} because our monitoring noticed "
        "something. Open by greeting them by name and asking how they're feeling right "
        "now. Find out, gently: can they breathe okay, are they in pain, do they need "
        "help immediately. Be calm, kind, and brief — under a minute. If they sound "
        "distressed, confused, or can't answer clearly, reassure them that help is on "
        "the way. Do not diagnose. Keep turns to 1-2 short sentences."
    )


def instructions_for(ctx: CallContext) -> str:
    return _patient_instructions(ctx.fields) if ctx.role == "patient" else _nurse_instructions(
        ctx.fields
    )


# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested without any network)
# --------------------------------------------------------------------------- #
def wss_base() -> str:
    """Derive the public wss:// origin from VIGIL_PUBLIC_URL (an https tunnel)."""
    host = urlparse(settings.public_url).netloc or settings.public_url.replace("https://", "").replace(
        "http://", ""
    )
    return f"wss://{host}"


def build_twiml(context_id: str) -> str:
    """The TwiML Twilio fetches when the call connects: open a bidirectional media
    stream to our bridge. <Connect><Stream> (not <Start>) so audio flows BOTH ways."""
    stream_url = f"{wss_base()}/media-stream/{context_id}"
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        "<Connect>"
        f'<Stream url="{stream_url}" />'
        "</Connect>"
        "</Response>"
    )


def openai_to_twilio_frame(stream_sid: str, audio_b64: str) -> dict:
    """OpenAI Realtime audio delta (already base64 mu-law) -> a Twilio media frame."""
    return {"event": "media", "streamSid": stream_sid, "media": {"payload": audio_b64}}


def twilio_media_to_openai(payload_b64: str) -> dict:
    """A Twilio inbound media payload (base64 mu-law) -> an OpenAI append event."""
    return {"type": "input_audio_buffer.append", "audio": payload_b64}


def session_update(ctx: CallContext) -> dict:
    """The session.update that configures the Realtime agent for this call (GA schema:
    nested audio config, mu-law both ways, server VAD, live transcription)."""
    tools = []
    if ctx.role == "nurse":
        tools = [
            {
                "type": "function",
                "name": "get_patient_status",
                "description": (
                    "Get the patient's CURRENT live status from Vigil's monitoring "
                    "(posture, movement, latest ESI and re-triage rationale). Call this "
                    "when the nurse asks how the patient is doing right now."
                ),
                "parameters": {"type": "object", "properties": {}},
            }
        ]
    return {
        "type": "session.update",
        "session": {
            "type": "realtime",
            "instructions": instructions_for(ctx),
            "output_modalities": ["audio"],
            "audio": {
                "input": {
                    "format": {"type": AUDIO_FORMAT},  # mu-law in <- Twilio, no transcode
                    "turn_detection": {"type": "server_vad"},
                    "transcription": {"model": "whisper-1"},
                },
                "output": {
                    "format": {"type": AUDIO_FORMAT},  # mu-law out -> Twilio, no transcode
                    "voice": settings.openai_realtime_voice,
                },
            },
            "tools": tools,
            "tool_choice": "auto",
        },
    }


# --------------------------------------------------------------------------- #
# Placing the bridged call (Twilio REST — reuses the trial-friendly auth)
# --------------------------------------------------------------------------- #
def _twilio_auth() -> tuple[str, str]:
    if settings.twilio_api_key_sid and settings.twilio_api_key_secret:
        return settings.twilio_api_key_sid, settings.twilio_api_key_secret
    return settings.twilio_account_sid, settings.twilio_auth_token


def place_bridged_call(role: str, to_number: str, fields: dict[str, str]) -> tuple[str, str, str]:
    """Register the call context and place a Twilio call whose TwiML connects the
    audio to our OpenAI-Realtime bridge. Returns (context_id, call_sid, status). The
    context_id lets a caller look up the CallContext to await the conversation outcome
    (e.g. a patient check-in). Raises on a Twilio error."""
    cid = register_context(role, fields)
    twiml_url = f"{settings.public_url.rstrip('/')}/twiml/{cid}"
    r = requests.post(
        f"https://api.twilio.com/2010-04-01/Accounts/{settings.twilio_account_sid}/Calls.json",
        auth=_twilio_auth(),
        data={"To": to_number, "From": settings.twilio_from_number, "Url": twiml_url},
        timeout=20,
    )
    r.raise_for_status()
    d = r.json()
    return cid, str(d.get("sid", "")), str(d.get("status", ""))


# --------------------------------------------------------------------------- #
# The live bridge (runs on the server event loop, one per connected call)
# --------------------------------------------------------------------------- #
async def run_bridge(twilio_ws, context_id: str, on_turn=None, live_status=None) -> None:
    """Bridge a connected Twilio Media Stream to an OpenAI Realtime session.

    `twilio_ws` is the accepted FastAPI WebSocket. `on_turn(role, text)` (optional) is
    called with each completed transcript turn (for the dashboard/Supabase feed).
    `live_status()` (optional) returns the patient-status dict for the get_patient_status
    tool. Cleans up and sets ctx.done on exit.
    """
    from websockets.asyncio.client import connect as ws_connect

    ctx = get_context(context_id)
    if ctx is None:
        log.warning("media stream for unknown context %s — closing", context_id)
        await twilio_ws.close()
        return

    # GA Realtime: just the bearer token (no OpenAI-Beta header — that selects the
    # retired beta shape, which is disabled on current accounts).
    headers = {"Authorization": f"Bearer {settings.openai_api_key}"}
    url = f"{OPENAI_WS}?model={settings.openai_realtime_model}"
    state = {"stream_sid": ""}

    def _emit_turn(role: str, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        (ctx.caller_turns if role != "agent" else ctx.agent_turns).append(text)
        if on_turn is not None:
            try:
                on_turn(role, text)
            except Exception:  # noqa: BLE001 — a feed hiccup must never kill the call
                log.debug("on_turn sink failed", exc_info=True)

    try:
        async with ws_connect(url, additional_headers=headers, max_size=None) as openai_ws:
            await openai_ws.send(json.dumps(session_update(ctx)))
            # Make the agent OPEN the conversation (don't wait for the human to speak).
            # IMPORTANT: a bare response.create — no per-response `instructions` — so the
            # opener INHERITS the full session persona (English + the grounded opening
            # line + the DATA). Passing response.instructions here would REPLACE the
            # persona for this turn, which drops the language anchor and the facts.
            await openai_ws.send(json.dumps({"type": "response.create"}))

            async def twilio_to_openai() -> None:
                while True:
                    raw = await twilio_ws.receive_text()  # raises on caller hang-up
                    data = json.loads(raw)
                    ev = data.get("event")
                    if ev == "start":
                        state["stream_sid"] = data["start"]["streamSid"]
                        log.info("media stream %s started (%s)", context_id, ctx.role)
                    elif ev == "media":
                        await openai_ws.send(
                            json.dumps(twilio_media_to_openai(data["media"]["payload"]))
                        )
                    elif ev == "stop":
                        break

            async def openai_to_twilio() -> None:
                async for raw in openai_ws:
                    ev = json.loads(raw)
                    t = ev.get("type")
                    # agent audio -> Twilio (GA: response.output_audio.delta; also accept
                    # the beta name for portability). delta is base64 mu-law -> passthrough.
                    if (
                        t in ("response.output_audio.delta", "response.audio.delta")
                        and ev.get("delta")
                        and state["stream_sid"]
                    ):
                        await twilio_ws.send_text(
                            json.dumps(openai_to_twilio_frame(state["stream_sid"], ev["delta"]))
                        )
                    elif t == "input_audio_buffer.speech_started":
                        # barge-in: the caller started talking — stop our playback so the
                        # agent doesn't talk over them.
                        if state["stream_sid"]:
                            await twilio_ws.send_text(
                                json.dumps({"event": "clear", "streamSid": state["stream_sid"]})
                            )
                        await openai_ws.send(json.dumps({"type": "response.cancel"}))
                    elif t == "conversation.item.input_audio_transcription.completed":
                        _emit_turn(
                            "nurse" if ctx.role == "nurse" else "patient", ev.get("transcript", "")
                        )
                    elif t in (
                        "response.output_audio_transcript.done",
                        "response.audio_transcript.done",
                    ):
                        _emit_turn("agent", ev.get("transcript", ""))
                    elif t == "response.function_call_arguments.done":
                        await _handle_tool_call(openai_ws, ev, live_status)
                    elif t == "error":
                        log.warning("openai realtime error: %s", ev.get("error"))

            # Whichever leg ends first (caller hangs up, or the OpenAI socket closes)
            # tears down the other, so the bridge never hangs and ctx.done always fires.
            t1 = asyncio.create_task(twilio_to_openai())
            t2 = asyncio.create_task(openai_to_twilio())
            done, pending = await asyncio.wait({t1, t2}, return_when=asyncio.FIRST_COMPLETED)
            for p in pending:
                p.cancel()
                try:
                    await p
                except (asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass
            for d in done:
                if d.exception():
                    log.info("bridge leg for %s ended: %r", context_id, d.exception())
    except Exception as e:  # noqa: BLE001 — always release the waiter + report honestly
        log.info("bridge for %s ended: %r", context_id, e)
    finally:
        ctx.done.set()
        try:
            await twilio_ws.close()
        except Exception:  # noqa: BLE001
            pass


async def _handle_tool_call(openai_ws, ev: dict, live_status) -> None:
    """Answer a get_patient_status function call with real live status, then let the
    agent continue speaking."""
    name = ev.get("name")
    call_id = ev.get("call_id")
    result: dict = {}
    if name == "get_patient_status" and live_status is not None:
        try:
            result = live_status() or {}
        except Exception:  # noqa: BLE001
            result = {"error": "status unavailable"}
    await openai_ws.send(
        json.dumps(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": json.dumps(result),
                },
            }
        )
    )
    await openai_ws.send(json.dumps({"type": "response.create"}))


def evaluate_checkin(ctx: CallContext) -> tuple[bool, bool, str]:
    """From a finished patient check-in, derive (answered, reassuring, transcript) —
    the same contract the ladder uses for the ElevenLabs check-in path."""
    said = " ".join(ctx.caller_turns).lower()
    answered = bool(said.strip())
    reassuring = answered and not any(w in said for w in DISTRESS_WORDS)
    return answered, reassuring, " | ".join(ctx.caller_turns) or "(no response)"
