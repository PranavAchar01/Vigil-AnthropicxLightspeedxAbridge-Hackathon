"""Tier 0 deterministic re-triage used before any optional model call."""

from __future__ import annotations

import hashlib
import json

from vigil.chart import PatientChart
from vigil.events import Action, FusedEvent, Severity, TriageDecision
from vigil.monitoring import PatientMonitor


def decide_tier_zero(
    chart: PatientChart, monitor: PatientMonitor, fused: FusedEvent
) -> TriageDecision:
    prior = monitor.current_esi
    evidence = [_evidence_label(kind) for kind in fused.kinds]
    risk_evidence = _relevant_risks(monitor.risk_factors, fused.kinds)
    evidence.extend(risk_evidence)

    if fused.severity == Severity.HARD:
        new_esi = min(prior, 2)
        action = Action.PAGE_IMMEDIATELY
        confidence = min(0.98, max(0.78, fused.confidence + 0.08 * len(risk_evidence)))
        rationale = (
            f"Tier 0 detected {fused.summary.lower()}. "
            f"The current chart adds {', '.join(risk_evidence) if risk_evidence else 'no additional high-risk modifier'}. "
            "The monotonic safety rule permits urgency to rise only."
        )
    else:
        concerning_nonresponse = "non_response" in fused.kinds
        if concerning_nonresponse and monitor.risk_factors:
            new_esi = min(prior, 2)
            action = Action.PAGE_IMMEDIATELY
            confidence = max(0.78, fused.confidence)
            rationale = (
                "The patient did not respond to a directed check-in and has charted risk factors. "
                "A clinician assessment is required now."
            )
        else:
            new_esi = prior
            action = Action.VOICE_CHECKIN
            confidence = max(0.58, fused.confidence)
            rationale = (
                f"Tier 0 detected {fused.summary.lower()}, but the signal is not independently diagnostic. "
                "Vigil will check in using voice and on-screen text before paging."
            )

    snapshot = {
        "patient_id": monitor.patient_id,
        "prior_esi": prior,
        "risk_factors": monitor.risk_factors,
        "event": fused.model_dump(mode="json"),
        "baseline_deviation": monitor.baseline_deviation,
    }
    snapshot_hash = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(snapshot, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )
    spoken = _spoken_summary(chart, fused, prior, new_esi, action)
    return TriageDecision(
        patient_id=monitor.patient_id,
        prior_esi=prior,
        new_esi=min(prior, new_esi),
        escalate=action != Action.HOLD,
        action=action,
        rationale=rationale,
        spoken_summary=spoken,
        confidence=round(confidence, 2),
        evidence=evidence,
        input_snapshot_hash=snapshot_hash,
        reasoning_tier="tier_0",
    )


def _relevant_risks(risks: list[str], kinds: list[str]) -> list[str]:
    relevant: list[str] = []
    if "anticoagulated" in risks and any(
        k in kinds for k in ("fall", "collapse", "gait_instability")
    ):
        relevant.append("anticoagulant increases bleeding risk")
    if "cardiac_history" in risks and any(
        k in kinds for k in ("collapse", "chest_clutch", "slump", "distress_phrase")
    ):
        relevant.append("cardiac history")
    if "respiratory_risk" in risks and any(
        k in kinds for k in ("labored_breathing", "distress_phrase", "non_response")
    ):
        relevant.append("respiratory history")
    if "low_oxygen_saturation" in risks:
        relevant.append("low charted oxygen saturation")
    return relevant


def _evidence_label(kind: str) -> str:
    return {
        "fall": "fall detected",
        "collapse": "collapse detected",
        "scream": "distress vocalization",
        "motionless": "movement below baseline",
        "slump": "posture declined from baseline",
        "agitation": "movement above baseline",
        "chest_clutch": "repeated chest guarding",
        "gait_instability": "gait instability",
        "labored_breathing": "possible labored breathing",
        "distress_phrase": "patient reported worsening symptoms",
        "non_response": "no response to directed check-in",
        "companion_alarm": "companion requested urgent help",
    }.get(kind, kind.replace("_", " "))


def _spoken_summary(
    chart: PatientChart,
    fused: FusedEvent,
    prior: int,
    new_esi: int,
    action: Action,
) -> str:
    if action == Action.PAGE_IMMEDIATELY:
        return (
            f"{chart.name}, currently ESI {prior}. {fused.summary}. "
            f"Recommending ESI {new_esi} and immediate nurse assessment."
        )
    return (
        f"{chart.name}, currently ESI {prior}. {fused.summary}. "
        "Starting an accessible patient check-in before paging."
    )
