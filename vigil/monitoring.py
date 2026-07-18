"""Multi-patient monitoring state for the waiting-room command center.

This module is deliberately independent from cameras, model providers, and the
web framework. Live sensors and deterministic demo replays both enter through
the same per-patient fusion path.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass
from enum import Enum

from pydantic import BaseModel, Field

from vigil.chart import PatientChart
from vigil.events import FusedEvent, PerceptionEvent, Severity, TriageDecision
from vigil.perception.fusion import EventFuser


class BindingState(str, Enum):
    BOUND = "bound"
    STALE = "stale"
    UNBOUND = "unbound"


class EscalationState(str, Enum):
    MONITORING = "monitoring"
    CHECKIN = "checkin"
    PAGE_PENDING = "page_pending"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"


class ConsentRecord(BaseModel):
    granted_at: float = Field(default_factory=time.time)
    scope: str = "continuous_retriage"
    expires_at: float | None = None
    method: str = "demo_kiosk"

    def active(self, now: float | None = None) -> bool:
        current = now if now is not None else time.time()
        return self.expires_at is None or current < self.expires_at


class BaselineProfile(BaseModel):
    mean_torso_angle: float = 0.0
    motion_energy_p50: float = 0.0
    motion_energy_p10: float = 0.0
    vocalization_rate: float = 0.0
    established: bool = False
    sample_count: int = 0


class TrackBinding(BaseModel):
    track_id: int
    patient_id: str | None = None
    bound_at: float = Field(default_factory=time.time)
    last_face_confirm: float = Field(default_factory=time.time)
    confidence: float = 0.0
    state: BindingState = BindingState.UNBOUND

    def refreshed(self, now: float | None = None) -> "TrackBinding":
        current = now if now is not None else time.time()
        age = current - self.last_face_confirm
        if self.patient_id is None or age > 120:
            next_state = BindingState.UNBOUND
        elif age > 45:
            next_state = BindingState.STALE
        else:
            next_state = BindingState.BOUND
        return self.model_copy(update={"state": next_state})


class AlertRecord(BaseModel):
    alert_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    patient_id: str
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    severity: Severity
    state: EscalationState
    title: str
    evidence: list[str] = Field(default_factory=list)
    prior_esi: int
    current_esi: int
    acknowledgement_due_at: float | None = None
    acknowledged_by: str | None = None
    acknowledged_at: float | None = None
    feedback: str | None = None


class PatientMonitor(BaseModel):
    patient_id: str
    name: str
    age: int | None = None
    gender: str = "unknown"
    visit: str = ""
    seat: str = ""
    consent: ConsentRecord | None = None
    initial_esi: int = Field(ge=1, le=5)
    current_esi: int = Field(ge=1, le=5)
    risk_factors: list[str] = Field(default_factory=list)
    baseline: BaselineProfile = Field(default_factory=BaselineProfile)
    baseline_deviation: float = Field(default=0.0, ge=0.0, le=1.0)
    wait_started_at: float = Field(default_factory=time.time)
    last_assessed_at: float = Field(default_factory=time.time)
    track_ids: set[int] = Field(default_factory=set)
    escalation_state: EscalationState = EscalationState.MONITORING
    active_alert_id: str | None = None
    latest_signal: str = "Stable baseline"
    last_event_at: float | None = None
    reassessment_target_minutes: int = 30
    accessibility_mode: str = "voice_and_text"
    preferred_language: str = "English"


@dataclass(frozen=True)
class IngestResult:
    patient_id: str | None
    fused: FusedEvent | None
    safety_only: bool = False


REASSESSMENT_TARGETS = {1: 5, 2: 10, 3: 30, 4: 60, 5: 120}


def derive_risk_factors(chart: PatientChart) -> list[str]:
    text = " ".join(chart.active_conditions + chart.active_medications).lower()
    risks: list[str] = []
    rules = {
        "anticoagulated": ("apixaban", "warfarin", "rivaroxaban", "anticoagul"),
        "cardiac_history": ("coronary", "cardiac", "heart failure", "arrhythm"),
        "respiratory_risk": ("pneumonia", "copd", "asthma", "respiratory"),
        "diabetes": ("diabetes", "insulin"),
        "high_comorbidity": ("chronic kidney", "cancer", "sepsis"),
    }
    for label, needles in rules.items():
        if any(needle in text for needle in needles):
            risks.append(label)
    spo2 = chart.latest_vitals.get("spo2")
    if spo2 and spo2.value < 92:
        risks.append("low_oxygen_saturation")
    return risks


class MonitorRegistry:
    """Thread-safe source of truth for patients, bindings, alerts, and ranking."""

    def __init__(self, *, fusion_window_s: float = 4.0, alert_budget: int = 8) -> None:
        self._lock = threading.RLock()
        self._charts: dict[str, PatientChart] = {}
        self._monitors: dict[str, PatientMonitor] = {}
        self._bindings: dict[int, TrackBinding] = {}
        self._fusers: dict[str, EventFuser] = {}
        self._alerts: dict[str, AlertRecord] = {}
        self._fusion_window_s = fusion_window_s
        self._alert_budget_limit = alert_budget
        self._alerts_created = 0

    def clear(self) -> None:
        with self._lock:
            self._charts.clear()
            self._monitors.clear()
            self._bindings.clear()
            self._fusers.clear()
            self._alerts.clear()
            self._alerts_created = 0

    def add_chart(
        self,
        chart: PatientChart,
        *,
        consent: ConsentRecord | None,
        wait_minutes: int = 0,
        seat: str = "",
        language: str = "English",
        accessibility_mode: str = "voice_and_text",
    ) -> PatientMonitor:
        now = time.time()
        monitor = PatientMonitor(
            patient_id=chart.patient_id,
            name=chart.name,
            age=chart.age,
            gender=chart.gender,
            visit=chart.visit_title,
            seat=seat,
            consent=consent,
            initial_esi=chart.baseline_esi,
            current_esi=chart.baseline_esi,
            risk_factors=derive_risk_factors(chart),
            wait_started_at=now - wait_minutes * 60,
            last_assessed_at=now - wait_minutes * 60,
            reassessment_target_minutes=REASSESSMENT_TARGETS[chart.baseline_esi],
            preferred_language=language,
            accessibility_mode=accessibility_mode,
        )
        with self._lock:
            self._charts[chart.patient_id] = chart
            self._monitors[chart.patient_id] = monitor
            self._fusers[chart.patient_id] = EventFuser(window_s=self._fusion_window_s)
        return monitor.model_copy(deep=True)

    def monitor(self, patient_id: str) -> PatientMonitor | None:
        with self._lock:
            value = self._monitors.get(patient_id)
            return value.model_copy(deep=True) if value else None

    def chart(self, patient_id: str) -> PatientChart | None:
        with self._lock:
            return self._charts.get(patient_id)

    def monitors(self) -> list[PatientMonitor]:
        with self._lock:
            return [item.model_copy(deep=True) for item in self._monitors.values()]

    def bind_track(
        self,
        track_id: int,
        patient_id: str | None,
        *,
        confidence: float = 1.0,
        now: float | None = None,
    ) -> TrackBinding:
        current = now if now is not None else time.time()
        with self._lock:
            monitor = self._monitors.get(patient_id or "")
            consented = bool(monitor and monitor.consent and monitor.consent.active(current))
            bound_patient = patient_id if consented else None
            binding = TrackBinding(
                track_id=track_id,
                patient_id=bound_patient,
                bound_at=current,
                last_face_confirm=current,
                confidence=confidence if bound_patient else 0.0,
                state=BindingState.BOUND if bound_patient else BindingState.UNBOUND,
            )
            prior = self._bindings.get(track_id)
            if prior and prior.patient_id and prior.patient_id in self._monitors:
                self._monitors[prior.patient_id].track_ids.discard(track_id)
            self._bindings[track_id] = binding
            if bound_patient:
                self._monitors[bound_patient].track_ids.add(track_id)
            return binding.model_copy(deep=True)

    def binding(self, track_id: int, *, now: float | None = None) -> TrackBinding | None:
        with self._lock:
            binding = self._bindings.get(track_id)
            if binding is None:
                return None
            refreshed = binding.refreshed(now)
            self._bindings[track_id] = refreshed
            if refreshed.state == BindingState.UNBOUND and refreshed.patient_id:
                monitor = self._monitors.get(refreshed.patient_id)
                if monitor:
                    monitor.track_ids.discard(track_id)
                refreshed = refreshed.model_copy(update={"patient_id": None})
                self._bindings[track_id] = refreshed
            return refreshed.model_copy(deep=True)

    def ingest(self, event: PerceptionEvent) -> IngestResult:
        binding = self.binding(event.track_id, now=event.ts)
        if binding is None or binding.patient_id is None:
            safety_fuser = EventFuser(window_s=self._fusion_window_s, cooldown_s=0)
            fused = (
                safety_fuser.add(event)
                if event.kind in {"fall", "collapse", "companion_alarm"}
                else None
            )
            return IngestResult(patient_id=None, fused=fused, safety_only=True)

        patient_id = binding.patient_id
        with self._lock:
            monitor = self._monitors[patient_id]
            deviation = min(1.0, monitor.baseline_deviation + _deviation_increment(event.kind))
            monitor.baseline_deviation = deviation
            monitor.latest_signal = _signal_label(event.kind)
            monitor.last_event_at = event.ts
            fuser = self._fusers[patient_id]
            fused = fuser.add(event)
        return IngestResult(patient_id=patient_id, fused=fused)

    def apply_decision(self, decision: TriageDecision) -> PatientMonitor:
        with self._lock:
            monitor = self._monitors[decision.patient_id]
            monitor.current_esi = min(monitor.current_esi, decision.new_esi)
            monitor.last_assessed_at = time.time()
            monitor.reassessment_target_minutes = REASSESSMENT_TARGETS[monitor.current_esi]
            return monitor.model_copy(deep=True)

    def override_esi(self, patient_id: str, new_esi: int) -> PatientMonitor:
        if new_esi < 1 or new_esi > 5:
            raise ValueError("ESI must be between 1 and 5")
        with self._lock:
            monitor = self._monitors[patient_id]
            if new_esi > monitor.current_esi:
                raise ValueError("Vigil cannot lower urgency")
            monitor.current_esi = new_esi
            monitor.last_assessed_at = time.time()
            monitor.reassessment_target_minutes = REASSESSMENT_TARGETS[new_esi]
            monitor.latest_signal = "Clinician urgency override"
            return monitor.model_copy(deep=True)

    def create_or_update_alert(self, decision: TriageDecision, fused: FusedEvent) -> AlertRecord:
        with self._lock:
            monitor = self._monitors[decision.patient_id]
            state = (
                EscalationState.PAGE_PENDING
                if decision.action.value == "page_immediately"
                else EscalationState.CHECKIN
            )
            existing = self._alerts.get(monitor.active_alert_id or "")
            if existing and existing.state not in {
                EscalationState.ACKNOWLEDGED,
                EscalationState.RESOLVED,
            }:
                existing.severity = fused.severity
                existing.state = state
                existing.title = fused.summary
                existing.evidence = list(dict.fromkeys(existing.evidence + decision.evidence))
                existing.current_esi = monitor.current_esi
                existing.updated_at = time.time()
                if state == EscalationState.PAGE_PENDING:
                    existing.acknowledgement_due_at = time.time() + 60
                alert = existing
            else:
                alert = AlertRecord(
                    patient_id=decision.patient_id,
                    severity=fused.severity,
                    state=state,
                    title=fused.summary,
                    evidence=decision.evidence or list(fused.kinds),
                    prior_esi=decision.prior_esi,
                    current_esi=monitor.current_esi,
                    acknowledgement_due_at=(
                        time.time() + 60 if state == EscalationState.PAGE_PENDING else None
                    ),
                )
                self._alerts[alert.alert_id] = alert
                self._alerts_created += 1
            monitor.active_alert_id = alert.alert_id
            monitor.escalation_state = alert.state
            return alert.model_copy(deep=True)

    def acknowledge(self, alert_id: str, actor: str) -> AlertRecord:
        with self._lock:
            alert = self._alerts[alert_id]
            now = time.time()
            alert.state = EscalationState.ACKNOWLEDGED
            alert.acknowledged_by = actor
            alert.acknowledged_at = now
            alert.updated_at = now
            self._monitors[alert.patient_id].escalation_state = EscalationState.ACKNOWLEDGED
            return alert.model_copy(deep=True)

    def feedback(self, alert_id: str, outcome: str) -> AlertRecord:
        if outcome not in {"confirmed", "false_alarm"}:
            raise ValueError("outcome must be confirmed or false_alarm")
        with self._lock:
            alert = self._alerts[alert_id]
            alert.feedback = outcome
            alert.state = EscalationState.RESOLVED
            alert.updated_at = time.time()
            monitor = self._monitors[alert.patient_id]
            monitor.escalation_state = EscalationState.RESOLVED
            monitor.active_alert_id = None
            if outcome == "false_alarm":
                monitor.baseline_deviation = max(0.0, monitor.baseline_deviation - 0.3)
            return alert.model_copy(deep=True)

    def request_medical_assist(self, seat: str, actor: str) -> AlertRecord:
        """Route a nonclinical staff request into the same clinical alert queue."""
        normalized = seat.strip().upper()
        with self._lock:
            monitor = next(
                (item for item in self._monitors.values() if item.seat.strip().upper() == normalized),
                None,
            )
            if monitor is None:
                raise KeyError(seat)

            now = time.time()
            existing = self._alerts.get(monitor.active_alert_id or "")
            if existing and existing.state not in {
                EscalationState.ACKNOWLEDGED,
                EscalationState.RESOLVED,
            }:
                existing.state = EscalationState.PAGE_PENDING
                existing.title = "Staff requested medical assist"
                existing.evidence = list(
                    dict.fromkeys(existing.evidence + [f"request routed by {actor}"])
                )
                existing.acknowledgement_due_at = now + 60
                existing.updated_at = now
                alert = existing
            else:
                alert = AlertRecord(
                    patient_id=monitor.patient_id,
                    severity=Severity.HARD,
                    state=EscalationState.PAGE_PENDING,
                    title="Staff requested medical assist",
                    evidence=[f"request routed by {actor}"],
                    prior_esi=monitor.current_esi,
                    current_esi=monitor.current_esi,
                    acknowledgement_due_at=now + 60,
                )
                self._alerts[alert.alert_id] = alert
                self._alerts_created += 1

            monitor.active_alert_id = alert.alert_id
            monitor.escalation_state = EscalationState.PAGE_PENDING
            monitor.latest_signal = "Staff requested medical assist"
            monitor.last_event_at = now
            return alert.model_copy(deep=True)

    def alerts(self) -> list[AlertRecord]:
        with self._lock:
            return [a.model_copy(deep=True) for a in self._alerts.values()]

    def active_alert(self, patient_id: str) -> AlertRecord | None:
        with self._lock:
            monitor = self._monitors.get(patient_id)
            if not monitor or not monitor.active_alert_id:
                return None
            alert = self._alerts.get(monitor.active_alert_id)
            return alert.model_copy(deep=True) if alert else None

    def queue(self, *, now: float | None = None) -> list[tuple[PatientMonitor, float]]:
        current = now if now is not None else time.time()
        with self._lock:
            ranked = [
                (monitor.model_copy(deep=True), _priority(monitor, current))
                for monitor in self._monitors.values()
            ]
        return sorted(ranked, key=lambda item: item[1], reverse=True)

    def alert_budget(self) -> dict[str, int]:
        with self._lock:
            return {"used": self._alerts_created, "limit": self._alert_budget_limit}


def _priority(monitor: PatientMonitor, now: float) -> float:
    wait_minutes = max(0.0, (now - monitor.wait_started_at) / 60)
    since_assessment = max(0.0, (now - monitor.last_assessed_at) / 60)
    target = max(5, monitor.reassessment_target_minutes)
    chart_risk = min(2.0, len(monitor.risk_factors) * 0.35)
    alert_boost = {
        EscalationState.PAGE_PENDING: 35,
        EscalationState.CHECKIN: 18,
        EscalationState.ACKNOWLEDGED: 8,
    }.get(monitor.escalation_state, 0)
    value = (
        (6 - monitor.current_esi) * 12
        + monitor.baseline_deviation * 24
        + min(4.0, since_assessment / target) * 9
        + chart_risk * 8
        + min(8.0, wait_minutes / 30)
        + alert_boost
    )
    return round(value, 2)


def _deviation_increment(kind: str) -> float:
    if kind in {"fall", "collapse", "companion_alarm"}:
        return 0.75
    if kind in {"labored_breathing", "non_response", "distress_phrase", "chest_clutch"}:
        return 0.4
    return 0.25


def _signal_label(kind: str) -> str:
    return {
        "fall": "Fall detected",
        "collapse": "Collapse detected",
        "scream": "Distress vocalization",
        "motionless": "Motion below baseline",
        "slump": "Posture declining",
        "agitation": "Movement above baseline",
        "chest_clutch": "Repeated chest guarding",
        "gait_instability": "Gait instability",
        "labored_breathing": "Possible labored breathing",
        "distress_phrase": "Patient reports worsening",
        "non_response": "No response to check-in",
        "companion_alarm": "Companion requested help",
    }.get(kind, kind.replace("_", " ").title())
