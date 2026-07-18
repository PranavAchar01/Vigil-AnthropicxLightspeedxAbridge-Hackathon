"""Initial ESI grade from a spoken intake (Gap B).

Given a patient's spoken reason-for-visit (transcribed) and any known chart data,
assign the FIRST ESI level by running the full ESI v4 decision tree with Claude,
forced through a single strict tool call. Unlike re-triage (triage.py) this is
NOT monotonic — there is no prior to clamp against. Two ESI-algorithm invariants
ARE enforced in code (never trusted to the model): Decision A implies ESI 1, and
danger-zone vitals up-triage a would-be ESI 3 to ESI 2. On a genuine API failure
it fails SAFE to ESI 2 (emergent) and flags for clinician review — it never
silently assigns a low acuity to an ungraded patient.
"""

from __future__ import annotations

import logging

from vigil.chart import PatientChart
from vigil.config import settings
from vigil.events import InitialTriageDecision
from vigil.reasoning.prompts import (
    INITIAL_TRIAGE_SYSTEM_PROMPT,
    INITIAL_TRIAGE_TOOL,
    build_intake_message,
)

log = logging.getLogger("vigil.initial_triage")

MODELS = [settings.model, "claude-sonnet-5", settings.fast_model]

_client = None


class ReasoningNotConfigured(RuntimeError):
    """Raised when ANTHROPIC_API_KEY is not set — intake triage cannot run."""


def _get_client():
    global _client
    if _client is None:
        import anthropic

        _client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    return _client


def _call(model: str, user_message: str):
    return _get_client().messages.create(
        model=model,
        max_tokens=640,
        system=[
            {
                "type": "text",
                "text": INITIAL_TRIAGE_SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        tools=[INITIAL_TRIAGE_TOOL],
        tool_choice={"type": "tool", "name": "initial_triage_decision"},
        messages=[{"role": "user", "content": user_message}],
    )


def _extract(resp) -> dict:
    for block in resp.content:
        if block.type == "tool_use" and block.name == "initial_triage_decision":
            return dict(block.input)
    raise ValueError(f"no initial_triage_decision block (stop_reason={resp.stop_reason})")


def _finalize(raw: dict, transcript: str, chart: PatientChart | None) -> InitialTriageDecision:
    esi = int(raw["esi"])
    point = str(raw.get("esi_decision_point", ""))
    danger = bool(raw.get("danger_zone_vitals", False))

    # ESI-algorithm invariants enforced in CODE (not trusted to the model):
    # 1) Decision A is the ONLY path to ESI 1, and it always yields 1.
    if point == "A":
        esi = 1
    # 2) Danger-zone vitals up-triage a would-be ESI 3 to ESI 2 (Decision D, exactly
    #    as written in the handbook — D is evaluated at the ESI-3 node only).
    if danger and esi == 3:
        esi = 2
        point = "D"
    # 3) Keep decision-point <-> level consistent for the audit trail.
    if esi == 1:
        point = "A"

    return InitialTriageDecision(
        patient_id=chart.patient_id if chart else "",
        chief_complaint=str(raw.get("chief_complaint", "")),
        esi=esi,
        esi_decision_point=point,
        esi_criteria=str(raw.get("esi_criteria", "")),
        predicted_resources=[str(r) for r in raw.get("predicted_resources", [])],
        danger_zone_vitals=danger,
        red_flags=[str(r) for r in raw.get("red_flags", [])],
        rationale=str(raw.get("rationale", "")),
        spoken_summary=str(raw.get("spoken_summary", "")),
        confidence=float(raw.get("confidence", 0.0)),
        needs_confirmation=True,
        transcript=transcript,
    )


def _fail_safe(transcript: str, chart: PatientChart | None) -> InitialTriageDecision:
    """On genuine API failure: assign a conservative ESI 2 and flag for a clinician.
    Over-triage is safe; silently under-grading an ungraded patient is not."""
    return _finalize(
        {
            "esi": 2,
            "esi_decision_point": "B",
            "esi_criteria": "Automated intake unavailable — clinician must triage manually.",
            "predicted_resources": [],
            "danger_zone_vitals": False,
            "red_flags": ["intake grader unavailable"],
            "chief_complaint": (transcript[:60] or "unspecified"),
            "confidence": 0.0,
            "rationale": "Intake triage model unavailable; defaulting to emergent pending review.",
            "spoken_summary": "Intake grader offline. Please triage this patient manually now.",
        },
        transcript,
        chart,
    )


def grade(transcript: str, chart: PatientChart | None = None) -> InitialTriageDecision:
    """Assign the initial ESI from a spoken intake (+ optional chart)."""
    if not settings.anthropic_api_key:
        raise ReasoningNotConfigured("ANTHROPIC_API_KEY is not set — intake triage is unavailable.")
    if not transcript.strip():
        raise ValueError("empty intake transcript")

    user_message = build_intake_message(transcript, chart)
    import anthropic

    retryable = (
        anthropic.APITimeoutError,
        anthropic.RateLimitError,
        anthropic.InternalServerError,
        anthropic.APIConnectionError,
    )
    last_err: Exception | None = None
    for model in MODELS:
        try:
            return _finalize(_extract(_call(model, user_message)), transcript, chart)
        except retryable as e:
            last_err = e
            log.warning("model %s failed (%s); falling back", model, type(e).__name__)
        except Exception as e:  # noqa: BLE001 — schema/other: try next, then fail safe
            last_err = e
            log.warning("model %s error (%r); falling back", model, e)
    log.error("all intake-triage models failed: %s", last_err)
    return _fail_safe(transcript, chart)
