"""Tests for the integration-free multi-patient product layer."""

from __future__ import annotations

from vigil.chart import PatientChart, Vital
from vigil.demo import DemoController, seed_demo
from vigil.events import Action, BusEvent, Modality, PerceptionEvent, Severity
from vigil.intake import IntakeRequest, assess_intake
from vigil.monitoring import (
    BindingState,
    ConsentRecord,
    EscalationState,
    MonitorRegistry,
)
from vigil.reasoning.rules import decide_tier_zero
from vigil.security import (
    AuditChain,
    BreakGlassManager,
    Role,
    queue_item_for_role,
)


def chart(pid: str, name: str, esi: int = 3, meds: list[str] | None = None) -> PatientChart:
    return PatientChart(
        patient_id=pid,
        encounter_id=f"enc-{pid}",
        name=name,
        gender="female",
        age=67,
        visit_title="Chest pain",
        active_conditions=["Coronary artery disease"],
        active_medications=meds or [],
        latest_vitals={"spo2": Vital("spo2", 91, "%", "2026")},
        baseline_esi=esi,
    )


def registry_with_two() -> MonitorRegistry:
    registry = MonitorRegistry()
    registry.add_chart(chart("p1", "Maria", meds=["apixaban"]), consent=ConsentRecord())
    registry.add_chart(chart("p2", "Jordan", esi=4), consent=ConsentRecord())
    registry.bind_track(1, "p1", now=100)
    registry.bind_track(2, "p2", now=100)
    return registry


def test_consent_gate_prevents_chart_binding():
    registry = MonitorRegistry()
    registry.add_chart(chart("p1", "Maria"), consent=None)
    binding = registry.bind_track(7, "p1", now=100)
    assert binding.state == BindingState.UNBOUND
    assert binding.patient_id is None


def test_track_binding_decays_to_stale_then_unbound():
    registry = registry_with_two()
    assert registry.binding(1, now=145).state == BindingState.BOUND
    assert registry.binding(1, now=146).state == BindingState.STALE
    expired = registry.binding(1, now=221)
    assert expired.state == BindingState.UNBOUND
    assert expired.patient_id is None


def test_fusion_windows_are_isolated_per_patient():
    registry = registry_with_two()
    first = registry.ingest(
        PerceptionEvent(
            ts=101,
            modality=Modality.VISION,
            kind="slump",
            confidence=0.7,
            track_id=1,
        )
    )
    second = registry.ingest(
        PerceptionEvent(
            ts=101.2,
            modality=Modality.AUDIO,
            kind="labored_breathing",
            confidence=0.7,
            track_id=2,
        )
    )
    assert first.patient_id == "p1" and first.fused.severity == Severity.SOFT
    assert second.patient_id == "p2" and second.fused.severity == Severity.SOFT
    assert first.fused.kinds == ["slump"]
    assert second.fused.kinds == ["labored_breathing"]


def test_unbound_tracks_only_emit_gross_safety_events():
    registry = MonitorRegistry()
    soft = registry.ingest(
        PerceptionEvent(modality=Modality.VISION, kind="slump", confidence=0.8, track_id=99)
    )
    hard = registry.ingest(
        PerceptionEvent(modality=Modality.VISION, kind="fall", confidence=0.8, track_id=99)
    )
    assert soft.safety_only is True and soft.fused is None
    assert hard.safety_only is True and hard.fused is not None
    assert hard.patient_id is None


def test_time_and_chart_risk_raise_queue_priority():
    registry = MonitorRegistry()
    now = 10_000.0
    risky = registry.add_chart(
        chart("p1", "Maria", meds=["apixaban"]), consent=ConsentRecord(), wait_minutes=120
    )
    stable = registry.add_chart(chart("p2", "Jordan"), consent=ConsentRecord(), wait_minutes=5)
    # Rebase the synthetic wait timestamps so the assertion does not depend on wall time.
    registry._monitors[risky.patient_id].wait_started_at = now - 120 * 60
    registry._monitors[risky.patient_id].last_assessed_at = now - 120 * 60
    registry._monitors[stable.patient_id].wait_started_at = now - 5 * 60
    registry._monitors[stable.patient_id].last_assessed_at = now - 5 * 60
    ranked = registry.queue(now=now)
    assert ranked[0][0].patient_id == "p1"
    assert ranked[0][1] > ranked[1][1]


def test_monotonic_override_rejects_lower_urgency():
    registry = registry_with_two()
    registry.override_esi("p1", 2)
    try:
        registry.override_esi("p1", 3)
    except ValueError as exc:
        assert "cannot lower urgency" in str(exc)
    else:
        raise AssertionError("de-escalation should be rejected")


def test_tier_zero_uses_chart_risk_and_snapshot_hash():
    registry = registry_with_two()
    result = registry.ingest(
        PerceptionEvent(
            ts=101,
            modality=Modality.VISION,
            kind="fall",
            confidence=0.9,
            track_id=1,
        )
    )
    decision = decide_tier_zero(registry.chart("p1"), registry.monitor("p1"), result.fused)
    assert decision.new_esi == 2
    assert decision.action == Action.PAGE_IMMEDIATELY
    assert decision.is_monotonic()
    assert decision.input_snapshot_hash.startswith("sha256:")
    assert "anticoagulant increases bleeding risk" in decision.evidence


def test_server_side_redaction_removes_clinical_fields():
    registry = registry_with_two()
    monitor = registry.monitor("p1")
    patient_chart = registry.chart("p1")
    charge = queue_item_for_role(monitor, patient_chart, None, 50, Role.CHARGE_NURSE)
    front = queue_item_for_role(monitor, patient_chart, None, 50, Role.FRONT_DESK)
    security = queue_item_for_role(monitor, patient_chart, None, 50, Role.SECURITY)
    compliance = queue_item_for_role(monitor, patient_chart, None, 50, Role.COMPLIANCE)
    assert "chart" in charge and "current_esi" in charge
    assert "chart" not in front and "current_esi" not in front
    assert "name" not in security and "visit" not in security
    assert "name" not in compliance and compliance["patient_ref"].startswith("pt_")


def test_audit_chain_detects_tampering_and_compliance_redacts_phi():
    chain = AuditChain()
    chain.append(
        actor="rn-1",
        role=Role.CHARGE_NURSE,
        action="chart_read",
        resource="patient:p1",
        outcome="success",
        metadata={"patient_id": "p1", "name": "Maria"},
    )
    assert chain.verify()["valid"] is True
    compliance = chain.blocks(Role.COMPLIANCE)[0]
    assert compliance["resource"].startswith("patient:pt_")
    assert "name" not in compliance["metadata"]
    chain._blocks[0].outcome = "denied"
    assert chain.verify()["valid"] is False


def test_break_glass_requires_reason_and_expires():
    manager = BreakGlassManager()
    try:
        manager.grant("rn-1", "p1", "urgent")
    except ValueError:
        pass
    else:
        raise AssertionError("short reason should be rejected")
    grant = manager.grant("rn-1", "p1", "Patient collapsed outside assigned unit", ttl_seconds=10)
    assert manager.active(grant.grant_id, now=grant.granted_at + 5) is not None
    assert manager.active(grant.grant_id, now=grant.expires_at + 1) is None


def test_intake_rule_floor_overrides_unsafe_proposal():
    result = assess_intake(
        IntakeRequest(
            name="Maria",
            age=67,
            chief_complaint="Chest pain",
            medications=["apixaban"],
            proposed_esi=4,
        )
    )
    assert result.initial_esi == 2
    assert result.floor_applied is True
    assert "cardiac_history" in result.risk_factors


def test_demo_replay_creates_independent_ranked_alerts():
    registry = MonitorRegistry()
    audit = AuditChain()
    events: list[BusEvent] = []
    controller = DemoController(registry, audit, events.append)
    controller.reset()
    controller.advance()
    first = registry.active_alert("demo-vega")
    assert first is not None and first.state == EscalationState.CHECKIN
    controller.advance()
    upgraded = registry.active_alert("demo-vega")
    assert upgraded.alert_id == first.alert_id
    assert upgraded.state == EscalationState.PAGE_PENDING
    assert registry.monitor("demo-idris").current_esi == 4
    assert any(event.type == "decision" for event in events)


def test_seed_demo_has_three_bound_patients():
    registry = MonitorRegistry()
    charts = seed_demo(registry)
    assert set(charts) == {"demo-vega", "demo-idris", "demo-park"}
    assert len(registry.monitors()) == 3
    assert all(registry.binding(track).state == BindingState.BOUND for track in (101, 102, 103))
