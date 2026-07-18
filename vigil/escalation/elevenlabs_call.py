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
import time

import requests

from vigil.chart import PatientChart
from vigil.config import settings
from vigil.escalation.ladder import CheckinResult
from vigil.escalation.twilio_call import twilio_call_configured
from vigil.events import BusEvent, EscalationAction, TriageDecision

log = logging.getLogger("vigil.escalation")
BASE = "https://api.elevenlabs.io/v1"

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
        # Prefer the ElevenLabs AI voice when its telephony is configured; otherwise
        # (e.g. a Twilio trial account) place a direct Twilio call, which still works.
        if nurse_call_configured():
            self._emit("dialing", to=settings.nurse_phone_number)
            try:
                conversation_id = _place_outbound(
                    settings.elevenlabs_agent_id,
                    settings.nurse_phone_number,
                    self._dynamic_vars(decision),
                )
                self._emit("ringing", conversation_id=conversation_id)
                return EscalationAction(
                    kind="nurse_call",
                    target=settings.nurse_phone_number,
                    message=decision.spoken_summary,
                    status="dialing",
                    provider_ref=conversation_id,
                )
            except Exception as e:  # noqa: BLE001
                log.error("elevenlabs nurse call failed: %r", e)
                if twilio_call_configured():
                    log.info("falling back to direct Twilio call")
                    return self._twilio_call(decision)
                self._emit("failed", error=str(e))
                return EscalationAction(
                    kind="nurse_call",
                    target=settings.nurse_phone_number,
                    message=decision.spoken_summary,
                    status="failed",
                    provider_ref=str(e),
                )

        if twilio_call_configured():
            return self._twilio_call(decision)

        log.error("no telephony configured — set ELEVENLABS_* or TWILIO_* + NURSE_PHONE_NUMBER")
        self._emit("failed", error="No telephony configured")
        return EscalationAction(
            kind="nurse_call",
            target="",
            message=decision.spoken_summary,
            status="failed",
            provider_ref="not-configured",
        )

    def check_in_patient(self, decision: TriageDecision) -> CheckinResult:
        """Place a real voice check-in with the patient. If not configured, escalate
        (fail-safe) rather than fabricate a response."""
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
