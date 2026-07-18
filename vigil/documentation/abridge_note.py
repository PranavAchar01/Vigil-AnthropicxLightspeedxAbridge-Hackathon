"""Abridge-style ambient documentation: incident -> SOAP note -> FHIR R4 Bundle.

Closes the loop on Abridge's core thesis (ambient in, structured note out). The
escalation documents itself: we generate a SOAP incident note (Claude, with a
deterministic fallback), pair it with a short incident transcript, and assemble a
FHIR transaction Bundle (Encounter + ESI Observation + event Observation +
DocumentReference x2 + Flag + Communication) referencing the existing Patient by
urn:uuid. No FHIR server at the venue -> we write the Bundle to disk and surface
it on the dashboard.
"""

from __future__ import annotations

import base64
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from vigil.chart import PatientChart
from vigil.config import settings
from vigil.events import EscalationAction, FusedEvent, TriageDecision

log = logging.getLogger("vigil.docs")

# verified code systems / codes
LOINC = "http://loinc.org"
SCT = "http://snomed.info/sct"
V3_ACT = "http://terminology.hl7.org/CodeSystem/v3-ActCode"
OBS_CAT = "http://terminology.hl7.org/CodeSystem/observation-category"
COMM_CAT = "http://terminology.hl7.org/CodeSystem/communication-category"
FLAG_CAT = "http://terminology.hl7.org/CodeSystem/flag-category"
US_DOC_CAT = "http://hl7.org/fhir/us/core/CodeSystem/us-core-documentreference-category"
ESI_LOINC = "75636-1"  # Emergency severity index [ESI]
ED_NOTE = "34111-5"  # Emergency department Note
FALL_SCT = "217082002"  # Accidental fall
AT_RISK_SCT = "129839007"  # At risk for falls


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _urn() -> str:
    return f"urn:uuid:{uuid.uuid4()}"


def _b64(text: str) -> str:
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


@dataclass
class Incident:
    patient_uuid: str
    detected_at: str
    event_display: str
    prior_esi: int
    new_esi: int
    soap_note_text: str = ""
    transcript_text: str = ""
    nurse_ref: str = "Practitioner/charge-nurse"
    device_ref: str = "Device/vigil-edge"
    confidence: float | None = None
    encounter_urn: str = field(default_factory=_urn)


def _pt(inc: Incident) -> dict:
    return {"reference": f"urn:uuid:{inc.patient_uuid}"}


def build_encounter(inc: Incident) -> dict:
    return {
        "resourceType": "Encounter",
        "status": "in-progress",
        "class": {"system": V3_ACT, "code": "EMER", "display": "emergency"},
        "priority": {
            "coding": [{"system": LOINC, "code": ESI_LOINC, "display": "Emergency severity index"}],
            "text": f"ESI-{inc.new_esi}",
        },
        "subject": _pt(inc),
        "period": {"start": inc.detected_at},
        "reasonCode": [
            {
                "coding": [{"system": SCT, "code": FALL_SCT, "display": "Accidental fall"}],
                "text": inc.event_display,
            }
        ],
    }


def build_acuity_observation(inc: Incident, enc: str) -> dict:
    return {
        "resourceType": "Observation",
        "status": "final",
        "category": [{"coding": [{"system": OBS_CAT, "code": "survey", "display": "Survey"}]}],
        "code": {
            "coding": [
                {"system": LOINC, "code": ESI_LOINC, "display": "Emergency severity index [ESI]"}
            ],
            "text": "Re-triage acuity",
        },
        "subject": _pt(inc),
        "encounter": {"reference": enc},
        "effectiveDateTime": inc.detected_at,
        "valueInteger": inc.new_esi,
        "note": [
            {
                "text": f"Acuity escalated ESI-{inc.prior_esi} -> ESI-{inc.new_esi} on "
                f"Vigil detection of {inc.event_display.lower()}."
            }
        ],
    }


def build_event_observation(inc: Incident, enc: str) -> dict:
    obs = {
        "resourceType": "Observation",
        "status": "final",
        "category": [{"coding": [{"system": OBS_CAT, "code": "activity", "display": "Activity"}]}],
        "code": {
            "coding": [{"system": SCT, "code": FALL_SCT, "display": "Accidental fall"}],
            "text": inc.event_display,
        },
        "subject": _pt(inc),
        "encounter": {"reference": enc},
        "effectiveDateTime": inc.detected_at,
        "valueCodeableConcept": {"text": "Detected"},
        "device": {"reference": inc.device_ref},
    }
    if inc.confidence is not None:
        obs["component"] = [
            {
                "code": {"text": "detection confidence"},
                "valueQuantity": {
                    "value": inc.confidence,
                    "unit": "probability",
                    "system": "http://unitsofmeasure.org",
                    "code": "1",
                },
            }
        ]
    return obs


def build_soap_documentreference(inc: Incident, enc: str) -> dict:
    return {
        "resourceType": "DocumentReference",
        "status": "current",
        "docStatus": "preliminary",
        "type": {
            "coding": [{"system": LOINC, "code": ED_NOTE, "display": "Emergency department Note"}],
            "text": "Ambient SOAP note",
        },
        "category": [
            {
                "coding": [
                    {"system": US_DOC_CAT, "code": "clinical-note", "display": "Clinical Note"}
                ]
            }
        ],
        "subject": _pt(inc),
        "date": _now(),
        "author": [{"reference": inc.device_ref}],
        "content": [
            {
                "attachment": {
                    "contentType": "text/markdown",
                    "language": "en-US",
                    "data": _b64(inc.soap_note_text),
                    "title": "Vigil ambient SOAP note",
                    "creation": inc.detected_at,
                }
            }
        ],
        "context": {"encounter": [{"reference": enc}], "period": {"start": inc.detected_at}},
    }


def build_transcript_documentreference(inc: Incident, enc: str) -> dict:
    doc = build_soap_documentreference(inc, enc)
    doc["docStatus"] = "final"
    doc["type"] = {
        "coding": [{"system": LOINC, "code": "11488-4", "display": "Consultation note"}],
        "text": "Ambient transcript",
    }
    doc["content"] = [
        {
            "attachment": {
                "contentType": "text/plain",
                "language": "en-US",
                "data": _b64(inc.transcript_text),
                "title": "Vigil ambient transcript",
                "creation": inc.detected_at,
            }
        }
    ]
    return doc


def build_risk_flag(inc: Incident, enc: str) -> dict:
    return {
        "resourceType": "Flag",
        "status": "active",
        "category": [{"coding": [{"system": FLAG_CAT, "code": "safety", "display": "Safety"}]}],
        "code": {
            "coding": [{"system": SCT, "code": AT_RISK_SCT, "display": "At risk for falls"}],
            "text": f"Elevated risk — {inc.event_display} (ESI-{inc.new_esi})",
        },
        "subject": _pt(inc),
        "period": {"start": inc.detected_at},
        "encounter": {"reference": enc},
        "author": {"reference": inc.device_ref},
    }


def build_nurse_communication(inc: Incident, enc: str, about: list[str]) -> dict:
    return {
        "resourceType": "Communication",
        "status": "completed",
        "category": [{"coding": [{"system": COMM_CAT, "code": "alert", "display": "Alert"}]}],
        "priority": "stat",
        "subject": _pt(inc),
        "encounter": {"reference": enc},
        "sender": {"reference": inc.device_ref},
        "recipient": [{"reference": inc.nurse_ref}],
        "sent": _now(),
        "about": [{"reference": u} for u in about],
        "payload": [
            {
                "contentString": f"ESCALATION: {inc.event_display} at {inc.detected_at}. "
                f"Re-triaged ESI-{inc.prior_esi} -> ESI-{inc.new_esi}. Nurse response requested."
            }
        ],
    }


def build_incident_bundle(inc: Incident) -> dict:
    enc = inc.encounter_urn
    acuity, event, flag = _urn(), _urn(), _urn()
    soap, transcript, comm = _urn(), _urn(), _urn()
    entries = [
        (enc, build_encounter(inc), "Encounter"),
        (acuity, build_acuity_observation(inc, enc), "Observation"),
        (event, build_event_observation(inc, enc), "Observation"),
        (soap, build_soap_documentreference(inc, enc), "DocumentReference"),
        (transcript, build_transcript_documentreference(inc, enc), "DocumentReference"),
        (flag, build_risk_flag(inc, enc), "Flag"),
        (
            comm,
            build_nurse_communication(inc, enc, about=[flag, acuity, event, soap]),
            "Communication",
        ),
    ]
    return {
        "resourceType": "Bundle",
        "type": "transaction",
        "entry": [
            {"fullUrl": full, "resource": res, "request": {"method": "POST", "url": rt}}
            for full, res, rt in entries
        ],
    }


# --------------------------------------------------------------------------- #
# Ambient note + transcript generation
# --------------------------------------------------------------------------- #


def _fallback_soap(chart: PatientChart, fused: FusedEvent, decision: TriageDecision) -> str:
    vitals = ", ".join(f"{v.label} {v.value:g}{v.unit}" for v in chart.latest_vitals.values())
    conds = "; ".join(chart.active_conditions[:6]) or "none listed"
    return (
        f"# Vigil Incident Note — {chart.name}\n\n"
        f"**S:** Patient in waiting room for '{chart.visit_title}'. Vigil monitoring "
        f"detected {fused.summary.lower()} (signals: {', '.join(fused.kinds)}; "
        f"confidence {fused.confidence:g}).\n\n"
        f"**O:** Charted active conditions: {conds}. Latest vitals on file: "
        f"{vitals or 'none'}. Prior triage ESI-{decision.prior_esi}.\n\n"
        f"**A:** Acuity re-scored ESI-{decision.prior_esi} → ESI-{decision.new_esi}. "
        f"{decision.rationale}\n\n"
        f"**P:** Action: {decision.action.value.replace('_', ' ')}. "
        f"Charge nurse notified. Continue continuous monitoring; do not de-escalate.\n"
    )


def generate_soap_note(chart: PatientChart, fused: FusedEvent, decision: TriageDecision) -> str:
    if not settings.anthropic_api_key:
        return _fallback_soap(chart, fused, decision)
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        prompt = (
            "Write a concise ambient SOAP incident note (markdown, S/O/A/P headers) "
            "documenting a waiting-room deterioration that Vigil detected and escalated. "
            "Ground it strictly in the data below; do not invent findings.\n\n"
            f"PATIENT CHART:\n{chart.to_context()}\n\n"
            f"DETECTED EVENT: {fused.summary} (signals {fused.kinds}, "
            f"confidence {fused.confidence}).\n"
            f"RE-TRIAGE: ESI-{decision.prior_esi} -> ESI-{decision.new_esi}; "
            f"action {decision.action.value}; rationale: {decision.rationale}"
        )
        resp = client.messages.create(
            model=settings.fast_model,
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in resp.content if b.type == "text").strip() or _fallback_soap(
            chart, fused, decision
        )
    except Exception as e:  # noqa: BLE001
        log.warning("SOAP generation failed (%r); using fallback", e)
        return _fallback_soap(chart, fused, decision)


def _transcript(
    chart: PatientChart,
    fused: FusedEvent,
    decision: TriageDecision,
    actions: list[EscalationAction],
) -> str:
    lines = [
        f"[00:00] (Vigil) {fused.summary} — signals {', '.join(fused.kinds)}.",
        f"[00:01] (Vigil→Claude) Re-triage {chart.name}: ESI-{decision.prior_esi} → "
        f"ESI-{decision.new_esi}. {decision.rationale}",
    ]
    for a in actions:
        if a.kind == "patient_checkin":
            lines.append(f"[00:03] (Vigil→Patient) Voice check-in. Patient: {a.message}")
        elif a.kind == "nurse_call":
            lines.append(f'[00:05] (Vigil→Nurse) "{a.message}" [{a.status}]')
    return "\n".join(lines)


def write_incident(
    chart: PatientChart,
    fused: FusedEvent,
    decision: TriageDecision,
    actions: list[EscalationAction],
) -> tuple[str, str]:
    """Generate the note + bundle and write to disk. Returns (note_text, bundle_path)."""
    note = generate_soap_note(chart, fused, decision)
    transcript = _transcript(chart, fused, decision, actions)
    inc = Incident(
        patient_uuid=chart.patient_id,
        detected_at=datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        event_display=fused.summary,
        prior_esi=decision.prior_esi,
        new_esi=decision.new_esi,
        soap_note_text=note,
        transcript_text=transcript,
        confidence=fused.confidence,
    )
    bundle = build_incident_bundle(inc)
    out_dir = settings.cohort_path.parent / "incidents"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = inc.detected_at.replace(":", "").replace("-", "")
    path = out_dir / f"incident-{chart.patient_id[:8]}-{stamp}.json"
    path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    log.info("wrote FHIR incident bundle -> %s", path)
    return note, str(path)
