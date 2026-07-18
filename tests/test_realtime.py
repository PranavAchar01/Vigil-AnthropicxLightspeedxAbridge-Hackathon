"""Tests for the OpenAI-Realtime voice bridge — the parts that run without a network:
config gating, TwiML generation, the audio-frame passthrough both ways, the session
config, persona grounding, the context registry, and check-in evaluation."""

from __future__ import annotations

import json
from types import SimpleNamespace

from vigil.escalation import openai_realtime as rt


def _settings(**over):
    base = dict(
        openai_api_key="sk-test",
        openai_realtime_model="gpt-4o-realtime-preview",
        openai_realtime_voice="alloy",
        twilio_account_sid="ACxxx",
        twilio_auth_token="tok",
        twilio_api_key_sid="",
        twilio_api_key_secret="",
        twilio_from_number="+15550001111",
        public_url="https://abc123.trycloudflare.com",
    )
    base.update(over)
    return SimpleNamespace(**base)


# --------------------------- configuration gate ---------------------------


def test_configured_when_all_present(monkeypatch):
    monkeypatch.setattr(rt, "settings", _settings())
    assert rt.realtime_call_configured() is True


def test_not_configured_without_openai_key(monkeypatch):
    monkeypatch.setattr(rt, "settings", _settings(openai_api_key=""))
    assert rt.realtime_call_configured() is False


def test_not_configured_without_public_url(monkeypatch):
    # No public URL => Twilio can't reach the media stream => cannot bridge.
    monkeypatch.setattr(rt, "settings", _settings(public_url=""))
    assert rt.realtime_call_configured() is False


def test_configured_with_api_key_auth_instead_of_token(monkeypatch):
    monkeypatch.setattr(
        rt, "settings", _settings(twilio_auth_token="", twilio_api_key_sid="SKxx", twilio_api_key_secret="s")
    )
    assert rt.realtime_call_configured() is True


# --------------------------- TwiML / stream URL ---------------------------


def test_build_twiml_opens_bidirectional_stream(monkeypatch):
    monkeypatch.setattr(rt, "settings", _settings())
    xml = rt.build_twiml("c7-42")
    assert "<Connect>" in xml and "<Stream" in xml  # bidirectional (not <Start>)
    assert "wss://abc123.trycloudflare.com/media-stream/c7-42" in xml


# --------------------------- audio passthrough (both legs) ---------------------------


def test_twilio_media_to_openai_append():
    ev = rt.twilio_media_to_openai("BASE64ULAW")
    assert ev == {"type": "input_audio_buffer.append", "audio": "BASE64ULAW"}


def test_openai_delta_to_twilio_media_frame():
    frame = rt.openai_to_twilio_frame("STREAMSID", "BASE64ULAW")
    assert frame["event"] == "media"
    assert frame["streamSid"] == "STREAMSID"
    assert frame["media"]["payload"] == "BASE64ULAW"


# --------------------------- session config + persona grounding ---------------------------


def test_nurse_session_is_mulaw_and_has_status_tool(monkeypatch):
    monkeypatch.setattr(rt, "settings", _settings())
    ctx = rt.CallContext("nurse", {"patient_name": "Pranav Achar", "incident": "fainted", "esi": "ESI 2"})
    s = rt.session_update(ctx)["session"]
    assert s["type"] == "realtime"  # GA schema
    assert s["audio"]["input"]["format"]["type"] == "audio/pcmu"
    assert s["audio"]["output"]["format"]["type"] == "audio/pcmu"  # mu-law both ways
    assert s["audio"]["output"]["voice"] == "alloy"
    names = [t["name"] for t in s["tools"]]
    assert "get_patient_status" in names  # nurse can pull live status
    # persona is grounded in the actual incident
    assert "Pranav Achar" in s["instructions"] and "fainted" in s["instructions"]


def test_patient_session_has_no_tools_and_greets_by_name(monkeypatch):
    monkeypatch.setattr(rt, "settings", _settings())
    ctx = rt.CallContext("patient", {"patient_name": "Sahiel"})
    s = rt.session_update(ctx)["session"]
    assert s["tools"] == []  # a check-in doesn't expose the status tool
    assert "Sahiel" in s["instructions"]


# --------------------------- context registry ---------------------------


def test_context_registry_roundtrip():
    cid = rt.register_context("nurse", {"patient_name": "X"})
    ctx = rt.get_context(cid)
    assert ctx is not None and ctx.role == "nurse" and ctx.fields["patient_name"] == "X"
    assert rt.get_context("nope") is None


# --------------------------- check-in evaluation ---------------------------


def test_evaluate_checkin_reassuring():
    ctx = rt.CallContext("patient", {})
    ctx.caller_turns = ["I'm okay, just resting"]
    answered, reassuring, transcript = rt.evaluate_checkin(ctx)
    assert answered is True and reassuring is True and "okay" in transcript


def test_evaluate_checkin_distress_is_not_reassuring():
    ctx = rt.CallContext("patient", {})
    ctx.caller_turns = ["I can't breathe, my chest hurts"]
    answered, reassuring, _ = rt.evaluate_checkin(ctx)
    assert answered is True and reassuring is False  # distress words => escalate


def test_evaluate_checkin_no_answer_escalates():
    ctx = rt.CallContext("patient", {})
    answered, reassuring, transcript = rt.evaluate_checkin(ctx)
    assert answered is False and reassuring is False and transcript == "(no response)"


# --------------------------- session payload is JSON-serializable ---------------------------


def test_session_update_serializes(monkeypatch):
    monkeypatch.setattr(rt, "settings", _settings())
    ctx = rt.CallContext("nurse", {"patient_name": "A", "incident": "seizure", "esi": "ESI 1"})
    json.dumps(rt.session_update(ctx))  # must not raise
