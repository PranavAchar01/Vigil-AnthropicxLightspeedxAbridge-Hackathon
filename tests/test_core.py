"""Tests for Vigil's safety-critical logic: fusion, the escalation ladder, the
monotonic re-triage clamp, the offline heuristic, the fall-angle sign convention,
and the FHIR bundle shape. All run offline (no camera, mic, or API key)."""

from __future__ import annotations

import base64

import numpy as np
import pytest

from vigil.chart import PatientChart, Vital
from vigil.escalation.ladder import CheckinResult, run_ladder
from vigil.events import Action, FusedEvent, Modality, PerceptionEvent, Severity, TriageDecision
from vigil.perception import vision
from vigil.perception.fusion import EventFuser
from vigil.reasoning import triage


def make_chart(esi: int = 3, conditions=None, meds=None) -> PatientChart:
    return PatientChart(
        patient_id="pt-123",
        encounter_id="enc-1",
        name="Test Patient",
        gender="female",
        age=71,
        visit_title="COVID-19 isolation",
        active_conditions=conditions or ["Pneumonia (disorder)", "Coronary artery disease"],
        active_medications=meds or ["atenolol 50 MG Oral Tablet"],
        latest_vitals={
            "spo2": Vital("spo2", 85.8, "%", "2021"),
            "heart_rate": Vital("heart_rate", 165.0, "/min", "2021"),
        },
        baseline_esi=esi,
    )


# --------------------------- fusion ---------------------------


def test_scream_plus_fall_is_hard():
    f = EventFuser(window_s=4.0)
    f.add(PerceptionEvent(ts=100.0, modality=Modality.AUDIO, kind="scream", confidence=0.5))
    fused = f.add(PerceptionEvent(ts=100.5, modality=Modality.VISION, kind="fall", confidence=0.9))
    assert fused is not None
    assert fused.severity == Severity.HARD
    assert set(fused.kinds) == {"scream", "fall"}


def test_quiet_scream_alone_is_soft():
    f = EventFuser(window_s=4.0)
    fused = f.add(PerceptionEvent(ts=10.0, modality=Modality.AUDIO, kind="scream", confidence=0.4))
    assert fused is not None and fused.severity == Severity.SOFT


def test_fall_alone_is_hard():
    f = EventFuser(window_s=4.0)
    fused = f.add(PerceptionEvent(ts=1.0, modality=Modality.VISION, kind="fall", confidence=0.8))
    assert fused is not None and fused.severity == Severity.HARD


# --------------------------- escalation ladder ---------------------------


class FakeHandlers:
    def __init__(self, answered=False, reassuring=False):
        self.calls = []
        self._answered, self._reassuring = answered, reassuring

    def call_nurse(self, decision):
        self.calls.append("nurse")
        from vigil.events import EscalationAction

        return EscalationAction(
            kind="nurse_call", target="+1", message=decision.spoken_summary, status="completed"
        )

    def check_in_patient(self, decision):
        self.calls.append("checkin")
        return CheckinResult(self._answered, self._reassuring, "…")


def _decision(action: Action, escalate=True, new=2, prior=3) -> TriageDecision:
    return TriageDecision(
        patient_id="pt-123",
        prior_esi=prior,
        new_esi=new,
        escalate=escalate,
        action=action,
        rationale="r",
        spoken_summary="s",
    )


def test_ladder_page_immediately_calls_nurse():
    h = FakeHandlers()
    actions = run_ladder(_decision(Action.PAGE_IMMEDIATELY), h)
    assert h.calls == ["nurse"]
    assert actions[0].kind == "nurse_call"


def test_ladder_voice_checkin_escalates_on_no_answer():
    h = FakeHandlers(answered=False)
    actions = run_ladder(_decision(Action.VOICE_CHECKIN), h)
    assert h.calls == ["checkin", "nurse"]  # bad/absent answer -> escalate
    assert [a.kind for a in actions] == ["patient_checkin", "nurse_call"]


def test_ladder_voice_checkin_holds_on_good_answer():
    h = FakeHandlers(answered=True, reassuring=True)
    run_ladder(_decision(Action.VOICE_CHECKIN), h)
    assert h.calls == ["checkin"]  # reassuring answer -> no nurse call


def test_ladder_hold_does_nothing():
    h = FakeHandlers()
    actions = run_ladder(_decision(Action.HOLD, escalate=False), h)
    assert h.calls == []
    assert actions[0].kind == "none"


# --------------------------- monotonic clamp ---------------------------


def test_clamp_blocks_deescalation():
    chart = make_chart(esi=3)
    # model tries to make the patient LESS acute (raise ESI number) -> must clamp to prior
    d = triage._finalize(
        {"new_esi": 5, "action": "hold", "rationale": "r", "spoken_summary": "s"}, chart
    )
    assert d.new_esi == 3
    assert d.is_monotonic()


def test_high_acuity_forces_page():
    chart = make_chart(esi=3)
    d = triage._finalize(
        {"new_esi": 2, "action": "hold", "rationale": "r", "spoken_summary": "s"}, chart
    )
    assert d.action == Action.PAGE_IMMEDIATELY  # ESI<=2 may not 'hold'
    assert d.escalate is True


def test_escalation_is_derived_not_trusted():
    chart = make_chart(esi=3)
    d = triage._finalize(
        {"new_esi": 1, "action": "page_immediately", "rationale": "r", "spoken_summary": "s"}, chart
    )
    assert d.new_esi == 1 and d.escalate is True


# --------------------------- reasoning requires a real key ---------------------------


def test_decide_requires_api_key(monkeypatch):
    # No mock/heuristic stand-in: without a key, re-triage must fail loudly.
    import types

    monkeypatch.setattr(triage, "settings", types.SimpleNamespace(anthropic_api_key=""))
    chart = make_chart(esi=3, conditions=["Coronary artery disease"], meds=["apixaban"])
    fused = FusedEvent(
        kinds=["scream", "fall"],
        severity=Severity.HARD,
        confidence=0.9,
        summary="Scream + collapse detected",
    )
    with pytest.raises(triage.ReasoningNotConfigured):
        triage.decide(chart, fused)


# --------------------------- fall geometry sign convention ---------------------------


def test_torso_angle_upright_vs_lying():
    kp = np.zeros((17, 2), dtype=float)
    # upright: shoulders above hips (image y grows downward)
    kp[vision.L_SH] = (100, 100)
    kp[vision.R_SH] = (120, 100)
    kp[vision.L_HIP] = (100, 200)
    kp[vision.R_HIP] = (120, 200)
    assert vision.torso_angle_deg(kp) < 15  # ~0 = upright

    # lying: shoulders beside hips (horizontal torso)
    kp[vision.L_SH] = (100, 150)
    kp[vision.R_SH] = (100, 160)
    kp[vision.L_HIP] = (250, 150)
    kp[vision.R_HIP] = (250, 160)
    assert vision.torso_angle_deg(kp) > 75  # ~90 = horizontal


# --------------------------- live status store (agent tool) ---------------------------


def test_live_status_snapshot_roundtrip():
    from vigil.server import status as ps

    pid = "pt-live-1"
    ps.update_vision(pid, "on the floor", "still", moved=False)
    ps.mark_event(pid, "fall")
    ps.update_retriage(
        pid,
        new_esi=1,
        prev_esi=3,
        rationale="collapse in cardiac pt",
        spoken_summary="page now",
        chart_summary="prior MI",
    )
    snap = ps.snapshot(pid)
    assert snap["patient_id"] == pid
    assert snap["posture"] == "on the floor"
    assert snap["fall_detected"] is True
    assert snap["triage"]["esi_level"] == 1
    assert snap["triage"]["esi_changed"] is True
    assert snap["triage"]["direction"] == "worsening"  # 3 -> 1 is more acute
    assert snap["in_view"] is True  # just updated
    assert snap["chart_summary"] == "prior MI"


def test_live_status_unknown_patient_is_safe_default():
    from vigil.server import status as ps

    snap = ps.snapshot("never-seen")
    assert snap["posture"] == "unknown"
    assert snap["triage"]["esi_level"] is None


# --------------------------- face gallery (patient identity) ---------------------------


def test_face_gallery_matches_nearest_above_threshold():
    from vigil.perception.faces import FaceGallery

    g = FaceGallery(threshold=0.5)
    g.add("pt-a", "Ariane", [1.0, 0.0, 0.0])
    g.add("pt-b", "Dick", [0.0, 1.0, 0.0])
    # a vector close to A resolves to A
    m = g.identify([0.9, 0.1, 0.0])
    assert m is not None and m[0] == "pt-a"
    # an orthogonal / dissimilar vector stays unidentified (below threshold)
    assert g.identify([0.0, 0.0, 1.0]) is None


def test_face_gallery_json_roundtrip(tmp_path):
    from vigil.perception.faces import FaceGallery

    g = FaceGallery()
    g.add("pt-a", "Ariane", [0.1, 0.2, 0.3])
    p = tmp_path / "gallery.json"
    p.write_text(g.to_json())
    g2 = FaceGallery.load(p)
    assert len(g2) == 1
    assert g2.identify([0.1, 0.2, 0.3])[0] == "pt-a"


# --------------------------- FHIR bundle shape ---------------------------


def test_incident_bundle_is_valid_transaction():
    from vigil.documentation.abridge_note import Incident, build_incident_bundle

    inc = Incident(
        patient_uuid="c8f1e0a2-1234-4abc-9def-0123456789ab",
        detected_at="2026-07-17T14:32:05Z",
        event_display="Scream + collapse detected",
        prior_esi=3,
        new_esi=2,
        soap_note_text="S: ...\nO: ...\nA: ...\nP: ...",
        transcript_text="[00:00] fall",
        confidence=0.94,
    )
    b = build_incident_bundle(inc)
    assert b["type"] == "transaction"
    assert len(b["entry"]) == 8
    types = [e["resource"]["resourceType"] for e in b["entry"]]
    assert types == [
        "Encounter",
        "Observation",
        "Observation",
        "DocumentReference",
        "DocumentReference",
        "Flag",
        "Communication",
        "Provenance",
    ]
    enc = b["entry"][0]["resource"]
    assert enc["class"]["code"] == "EMER"  # R4: single Coding, not array
    acuity = b["entry"][1]["resource"]
    assert acuity["valueInteger"] == 2
    doc = b["entry"][3]["resource"]
    decoded = base64.b64decode(doc["content"][0]["attachment"]["data"]).decode()
    assert decoded.startswith("S:")
    assert "context" in doc and isinstance(doc["context"]["encounter"], list)  # R4 array
    for e in b["entry"]:
        assert e["fullUrl"].startswith("urn:uuid:")
        assert e["request"]["method"] == "POST"
    provenance = b["entry"][7]["resource"]
    assert provenance["resourceType"] == "Provenance"
    assert len(provenance["target"]) == 4
    assert len(provenance["entity"]) == 2
