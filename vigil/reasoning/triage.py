"""Chart-grounded MONOTONIC re-triage with Claude.

Input: a PatientChart + a FusedEvent + the prior ESI. Output: a TriageDecision,
forced via a single strict tool call. The acuity-only-increases invariant is
enforced in CODE (never trusted to the model). Requires a real ANTHROPIC_API_KEY —
there is no offline stand-in. Falls through a model chain, and on a genuine API
failure fails SAFE by paging a clinician (never silent, never fabricated).
"""

from __future__ import annotations

import logging

from vigil.chart import PatientChart
from vigil.config import settings
from vigil.events import Action, FusedEvent, TriageDecision
from vigil.reasoning.prompts import RETRIAGE_TOOL, SYSTEM_PROMPT, build_user_message

log = logging.getLogger("vigil.triage")

MODELS = [settings.model, "claude-sonnet-5", settings.fast_model]

_client = None


class ReasoningNotConfigured(RuntimeError):
    """Raised when ANTHROPIC_API_KEY is not set — re-triage cannot run."""


def _get_client():
    global _client
    if _client is None:
        import anthropic

        _client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    return _client


def _call(model: str, user_message: str):
    return _get_client().messages.create(
        model=model,
        max_tokens=512,
        system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
        tools=[RETRIAGE_TOOL],
        tool_choice={"type": "tool", "name": "retriage_decision"},
        messages=[{"role": "user", "content": user_message}],
        # No `thinking`: forced tool_choice is incompatible with it, and off = lowest latency.
    )


def _extract(resp) -> dict:
    for block in resp.content:
        if block.type == "tool_use" and block.name == "retriage_decision":
            return dict(block.input)
    raise ValueError(f"no retriage_decision block (stop_reason={resp.stop_reason})")


def _finalize(raw: dict, chart: PatientChart) -> TriageDecision:
    prior = chart.baseline_esi
    # 1) MONOTONIC clamp — the model can never de-escalate acuity.
    new_esi = min(int(raw["new_esi"]), prior)
    action = str(raw["action"])
    # 2) Action must not under-respond to high acuity.
    if new_esi <= 2 and action == "hold":
        action = "page_immediately"
    elif new_esi == 3 and action == "hold":
        action = "voice_checkin"
    # 3) escalate is DERIVED, not trusted.
    escalate = new_esi < prior or action == "page_immediately"
    return TriageDecision(
        patient_id=chart.patient_id,
        prior_esi=prior,
        new_esi=new_esi,
        escalate=escalate,
        action=Action(action),
        esi_decision_point=str(raw.get("esi_decision_point", "")),
        esi_criteria=str(raw.get("esi_criteria", "")),
        rationale=str(raw["rationale"]),
        spoken_summary=str(raw["spoken_summary"]),
    )


def _fail_safe(chart: PatientChart) -> TriageDecision:
    """On genuine API failure: page a clinician. Real safety behavior, not data."""
    return _finalize(
        {
            "new_esi": min(2, chart.baseline_esi),
            "action": "page_immediately",
            "esi_decision_point": "A",
            "esi_criteria": "Fail-safe: re-triage model unavailable — paging a clinician.",
            "rationale": "Re-triage model unavailable; failing safe by paging a clinician.",
            "spoken_summary": "Monitoring system error. Please check on this patient now.",
        },
        chart,
    )


def decide(chart: PatientChart, fused: FusedEvent) -> TriageDecision:
    if not settings.anthropic_api_key:
        raise ReasoningNotConfigured(
            "ANTHROPIC_API_KEY is not set — re-triage reasoning is unavailable."
        )

    user_message = build_user_message(chart, fused)
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
            return _finalize(_extract(_call(model, user_message)), chart)
        except retryable as e:
            last_err = e
            log.warning("model %s failed (%s); falling back", model, type(e).__name__)
        except Exception as e:  # noqa: BLE001 — schema/other: try next, then fail safe
            last_err = e
            log.warning("model %s error (%r); falling back", model, e)
    log.error("all re-triage models failed: %s", last_err)
    return _fail_safe(chart)
