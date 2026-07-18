"""Real outbound calls to the charge nurse (and optional patient check-in) via
ElevenLabs Conversational AI. No simulation, no fabricated results — every action
here either places a real call through the REST API or reports a real, honest
failure (e.g. "not configured"). Provide the keys and it works.

The connected ElevenLabs Agents MCP can CREATE the agents (scripts/setup_elevenlabs.py);
the dial itself goes through the REST API with an xi-api-key + a provisioned
Twilio number. Escalation script is filled at call time via dynamic_variables.
"""

from __future__ import annotations

import logging
import threading
import time

import requests

from vigil.chart import PatientChart
from vigil.config import settings
from vigil.escalation.ladder import CheckinResult
from vigil.escalation.openai_realtime import (
    evaluate_checkin,
    get_context,
    place_bridged_call,
    realtime_call_configured,
)
from vigil.escalation.twilio_call import twilio_call_configured
from vigil.events import BusEvent, EscalationAction, TriageDecision

log = logging.getLogger("vigil.escalation")
BASE = "https://api.elevenlabs.io/v1"

# --------------------------------------------------------------------------- #
# Outbound-call rate limit — so a Twilio TRIAL is NEVER spam-dialed.
# One successful nurse call per server run by default (VIGIL_MAX_NURSE_CALLS);
# a placed call is reserved atomically, and released only if it fails to dial.
# --------------------------------------------------------------------------- #
MAX_CALLS = settings.max_nurse_calls
COOLDOWN_S = settings.nurse_call_cooldown_s
_CALL_LOCK = threading.Lock()
_CALL_LOG: dict[str, float] = {}  # number -> last successful-dial monotonic time
_CALL_COUNT = 0


def reset_call_gate() -> None:
    """Clear the outbound-call counter (new demo run / tests)."""
    global _CALL_COUNT
    with _CALL_LOCK:
        _CALL_COUNT = 0
        _CALL_LOG.clear()


def _reserve_call(to_number: str) -> tuple[bool, str]:
    """Atomically claim a call slot. Returns (allowed, reason_if_blocked)."""
    global _CALL_COUNT
    with _CALL_LOCK:
        now = time.monotonic()
        if 0 <= MAX_CALLS <= _CALL_COUNT:
            return False, f"already placed {_CALL_COUNT} call(s); max {MAX_CALLS} per run"
        last = _CALL_LOG.get(to_number)
        if last is not None and COOLDOWN_S > 0 and now - last < COOLDOWN_S:
            return False, f"cooldown {COOLDOWN_S:g}s not elapsed"
        _CALL_COUNT += 1
        _CALL_LOG[to_number] = now
        return True, ""


def _release_call(to_number: str) -> None:
    """Return the slot if the call never actually dialed (so a later real incident can)."""
    global _CALL_COUNT
    with _CALL_LOCK:
        _CALL_COUNT = max(0, _CALL_COUNT - 1)
        _CALL_LOG.pop(to_number, None)


# distress words that make a patient's spoken reply NON-reassuring
DISTRESS_WORDS = ("help", "can't", "cannot breathe", "chest", "pain", "dizzy", "worse")


def nurse_call_configured() -> bool:
    return bool(
        settings.elevenlabs_api_key
        and settings.elevenlabs_agent_id
        and settings.elevenlabs_phone_number_id
        and settings.nurse_phone_number
    )


def checkin_configured() -> bool:
    return bool(
        settings.elevenlabs_api_key
        and settings.elevenlabs_checkin_agent_id
        and settings.elevenlabs_phone_number_id
        and settings.patient_kiosk_number
    )


def _headers() -> dict:
    return {"xi-api-key": settings.elevenlabs_api_key, "Content-Type": "application/json"}


def _place_outbound(agent_id: str, to_number: str, dynamic_vars: dict[str, str]) -> str:
    """Place a real outbound call; return the conversation_id (raises on error)."""
    r = requests.post(
        f"{BASE}/convai/twilio/outbound-call",
        headers=_headers(),
        json={
            "agent_id": agent_id,
            "agent_phone_number_id": settings.elevenlabs_phone_number_id,
            "to_number": to_number,
            "conversation_initiation_client_data": {"dynamic_variables": dynamic_vars},
        },
        timeout=15,
    )
    r.raise_for_status()
    return str(r.json().get("conversation_id", ""))


def _fetch_transcript(conversation_id: str, wait_s: float = 25.0) -> list[dict]:
    """Poll the conversation until it ends (or timeout); return transcript turns."""
    deadline = time.time() + wait_s
    turns: list[dict] = []
    while time.time() < deadline:
        try:
            r = requests.get(
                f"{BASE}/convai/conversations/{conversation_id}", headers=_headers(), timeout=10
            )
            r.raise_for_status()
            data = r.json()
            turns = data.get("transcript", []) or []
            if data.get("status") in ("done", "processed", "failed"):
                break
        except Exception as e:  # noqa: BLE001
            log.warning("transcript poll failed: %r", e)
        time.sleep(2.0)
    return turns


def _capture_conversation(
    conversation_id: str, patient_name: str | None, wait_s: float = 120.0
) -> None:
    """Poll an ElevenLabs conversation and log each turn (nurse Q + agent A + tool
    calls) to the Supabase feed — the automatable alternative to a post-call webhook."""
    from vigil.server import supabase_sink as supa

    seen: set[int] = set()
    deadline = time.time() + wait_s
    while time.time() < deadline:
        try:
            r = requests.get(
                f"{BASE}/convai/conversations/{conversation_id}",
                headers={"xi-api-key": settings.elevenlabs_api_key},
                timeout=10,
            )
            r.raise_for_status()
            data = r.json()
            for i, turn in enumerate(data.get("transcript", []) or []):
                if i in seen:
                    continue
                text = (turn.get("message") or turn.get("text") or "").strip()
                if not text:
                    continue
                seen.add(i)
                who = "nurse" if turn.get("role") in ("user", "human") else "agent"
                supa.log_event(
                    "conversation_turn",
                    source="elevenlabs",
                    patient=patient_name,
                    summary=f"{who}: {text[:120]}",
                    payload={
                        "role": who,
                        "text": text,
                        "conversation_id": conversation_id,
                        "turn": i,
                    },
                )
            if data.get("status") in ("done", "failed", "processed"):
                break
        except Exception as e:  # noqa: BLE001
            log.debug("conversation capture poll failed: %r", e)
        time.sleep(2.0)


def capture_conversation_async(conversation_id: str, patient_name: str | None) -> None:
    threading.Thread(
        target=_capture_conversation, args=(conversation_id, patient_name), daemon=True
    ).start()


class NurseCallHandler:
    """Concrete EscalationHandlers, bound to the incident's chart."""

    def __init__(self, chart: PatientChart, bus=None) -> None:
        self.chart = chart
        self.bus = bus

    def _emit(self, status: str, **extra) -> None:
        if self.bus is not None:
            self.bus.publish_from_thread(
                BusEvent(type="call_status", payload={"status": status, **extra})
            )

    def _dynamic_vars(self, decision: TriageDecision) -> dict[str, str]:
        vitals = ", ".join(
            f"{v.label} {v.value:g}{v.unit}" for v in self.chart.latest_vitals.values()
        )
        return {
            # patient_id is injected here (never LLM-filled) so the get_patient_status
            # tool always fetches the RIGHT patient — no hallucinated IDs over the phone.
            "patient_id": self.chart.patient_id,
            "patient_name": self.chart.name,
            "chart_summary": (
                f"{self.chart.age} year old, {self.chart.visit_title}. "
                f"Active: {'; '.join(self.chart.active_conditions[:4]) or 'none'}. "
                f"Latest vitals: {vitals or 'none on file'}."
            ),
            "incident": decision.spoken_summary,
            "esi": f"ESI {decision.new_esi}",
        }

    def _twilio_call(self, decision: TriageDecision) -> EscalationAction:
        from vigil.escalation.twilio_call import place_call

        self._emit("dialing", to=settings.nurse_phone_number, provider="twilio")
        try:
            sid, status = place_call(settings.nurse_phone_number, decision.spoken_summary)
            self._emit("ringing", conversation_id=sid, provider="twilio")
            return EscalationAction(
                kind="nurse_call",
                target=settings.nurse_phone_number,
                message=decision.spoken_summary,
                status="dialing",
                provider_ref=sid,
            )
        except Exception as e:  # noqa: BLE001
            log.error("twilio nurse call failed: %r", e)
            self._emit("failed", error=str(e))
            return EscalationAction(
                kind="nurse_call",
                target=settings.nurse_phone_number,
                message=decision.spoken_summary,
                status="failed",
                provider_ref=str(e),
            )

    def call_nurse(self, decision: TriageDecision) -> EscalationAction:
        """Gate every outbound nurse call through the anti-spam limiter, then dial."""
        to = settings.nurse_phone_number or "nurse"
        allowed, why = _reserve_call(to)
        if not allowed:
            log.info("nurse call suppressed: %s", why)
            self._emit("suppressed", to=settings.nurse_phone_number, message=why)
            return EscalationAction(
                kind="nurse_call",
                target=settings.nurse_phone_number,
                message=decision.spoken_summary,
                status="skipped",
                provider_ref=why,
            )
        action = self._do_call_nurse(decision)
        if action.status == "failed":  # never dialed → free the slot for a later real incident
            _release_call(to)
        return action

    def _realtime_call(self, decision: TriageDecision) -> EscalationAction:
        """A REAL back-and-forth phone conversation via OpenAI Realtime (bridged over
        Twilio). Returns a 'failed' action (never raises) so the ladder can fall back."""
        self._emit("dialing", to=settings.nurse_phone_number, provider="openai-realtime")
        try:
            _cid, sid, _status = place_bridged_call(
                "nurse", settings.nurse_phone_number, self._dynamic_vars(decision)
            )
            self._emit("ringing", conversation_id=sid, provider="openai-realtime")
            return EscalationAction(
                kind="nurse_call",
                target=settings.nurse_phone_number,
                message=decision.spoken_summary,
                status="dialing",
                provider_ref=sid,
            )
        except Exception as e:  # noqa: BLE001
            log.error("openai-realtime nurse call failed: %r", e)
            self._emit("failed", error=str(e))
            return EscalationAction(
                kind="nurse_call",
                target=settings.nurse_phone_number,
                message=decision.spoken_summary,
                status="failed",
                provider_ref=str(e),
            )

    def _elevenlabs_call(self, decision: TriageDecision) -> EscalationAction:
        self._emit("dialing", to=settings.nurse_phone_number, provider="elevenlabs")
        try:
            conversation_id = _place_outbound(
                settings.elevenlabs_agent_id,
                settings.nurse_phone_number,
                self._dynamic_vars(decision),
            )
            self._emit("ringing", conversation_id=conversation_id, provider="elevenlabs")
            # capture the live nurse<->agent dialogue into the Supabase feed
            capture_conversation_async(conversation_id, self.chart.name)
            return EscalationAction(
                kind="nurse_call",
                target=settings.nurse_phone_number,
                message=decision.spoken_summary,
                status="dialing",
                provider_ref=conversation_id,
            )
        except Exception as e:  # noqa: BLE001
            log.error("elevenlabs nurse call failed: %r", e)
            self._emit("failed", error=str(e))
            return EscalationAction(
                kind="nurse_call",
                target=settings.nurse_phone_number,
                message=decision.spoken_summary,
                status="failed",
                provider_ref=str(e),
            )

    def _do_call_nurse(self, decision: TriageDecision) -> EscalationAction:
        # Transport ladder, best voice first: a REAL OpenAI-Realtime conversation ->
        # the ElevenLabs agent -> a one-shot Twilio TTS call (works on a trial account).
        # Each transport falls through to the next only if it fails to dial.
        if realtime_call_configured():
            action = self._realtime_call(decision)
            if action.status != "failed":
                return action
        if nurse_call_configured():
            action = self._elevenlabs_call(decision)
            if action.status != "failed":
                return action
        if twilio_call_configured():
            return self._twilio_call(decision)

        log.error("no telephony configured — set OPENAI_API_KEY+TWILIO_*, ELEVENLABS_*, or TWILIO_*")
        self._emit("failed", error="No telephony configured")
        return EscalationAction(
            kind="nurse_call",
            target="",
            message=decision.spoken_summary,
            status="failed",
            provider_ref="not-configured",
        )

    def _realtime_checkin(self) -> CheckinResult:
        """Real OpenAI-Realtime check-in call to the patient; wait for the conversation
        to end, then derive the outcome from what they actually said."""
        to = settings.patient_kiosk_number
        if not to:
            log.info("no PATIENT_KIOSK_NUMBER for realtime check-in — escalating")
            return CheckinResult(False, False, "no patient number configured — escalating")
        self._emit("patient_checkin", to=to, provider="openai-realtime")
        try:
            cid, _sid, _status = place_bridged_call("patient", to, {"patient_name": self.chart.name})
        except Exception as e:  # noqa: BLE001
            log.error("realtime check-in dial failed: %r", e)
            return CheckinResult(False, False, f"check-in failed ({e}) — escalating")
        ctx = get_context(cid)
        if ctx is None:
            return CheckinResult(False, False, "check-in context lost — escalating")
        ctx.done.wait(timeout=90.0)  # bounded so the ladder never blocks forever
        answered, reassuring, transcript = evaluate_checkin(ctx)
        return CheckinResult(answered, reassuring, transcript)

    def check_in_patient(self, decision: TriageDecision) -> CheckinResult:
        """Place a real voice check-in with the patient. If not configured, escalate
        (fail-safe) rather than fabricate a response."""
        # Preferred: a real OpenAI-Realtime conversation with the patient. Place the
        # bridged call, wait for it to end, then derive answered/reassuring from what
        # the patient actually said (same contract as the ElevenLabs path).
        if realtime_call_configured():
            return self._realtime_checkin()

        if not checkin_configured():
            log.info("patient check-in not configured — escalating to nurse")
            return CheckinResult(
                answered=False,
                reassuring=False,
                transcript="patient check-in not configured — escalating",
            )
        self._emit("patient_checkin", to=settings.patient_kiosk_number)
        try:
            conv = _place_outbound(
                settings.elevenlabs_checkin_agent_id,
                settings.patient_kiosk_number,
                {"patient_name": self.chart.name},
            )
            turns = _fetch_transcript(conv)
            patient_turns = [t.get("message", "") for t in turns if t.get("role") == "user"]
            answered = any(txt.strip() for txt in patient_turns)
            said = " ".join(patient_turns).lower()
            reassuring = answered and not any(w in said for w in DISTRESS_WORDS)
            return CheckinResult(
                answered=answered,
                reassuring=reassuring,
                transcript=" | ".join(patient_turns) or "(no response)",
            )
        except Exception as e:  # noqa: BLE001
            log.error("patient check-in failed: %r", e)
            return CheckinResult(
                answered=False, reassuring=False, transcript=f"check-in failed ({e}) — escalating"
            )
