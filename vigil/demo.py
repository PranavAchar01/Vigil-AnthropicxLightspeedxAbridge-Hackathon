"""Deterministic three-patient replay for stage-safe product demonstrations."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

from vigil.chart import PatientChart, load_cohort
from vigil.events import BusEvent, PerceptionEvent
from vigil.monitoring import ConsentRecord, MonitorRegistry
from vigil.reasoning.rules import decide_tier_zero
from vigil.security import AuditChain, Role


DEMO_COHORT = Path(__file__).resolve().parent.parent / "data" / "demo_cohort.json"
DEMO_LAYOUT = [
    ("demo-vega", 101, 128, "A3", "Spanish", "voice_and_text"),
    ("demo-idris", 102, 82, "B1", "English", "voice_and_text"),
    ("demo-park", 103, 37, "C2", "English", "text_preferred"),
]


def demo_charts() -> list[PatientChart]:
    return load_cohort(DEMO_COHORT)


def seed_demo(registry: MonitorRegistry) -> dict[str, PatientChart]:
    registry.clear()
    charts = {chart.patient_id: chart for chart in demo_charts()}
    for patient_id, track_id, wait, seat, language, mode in DEMO_LAYOUT:
        registry.add_chart(
            charts[patient_id],
            consent=ConsentRecord(),
            wait_minutes=wait,
            seat=seat,
            language=language,
            accessibility_mode=mode,
        )
        registry.bind_track(track_id, patient_id, confidence=0.96)
    return charts


class DemoController:
    def __init__(
        self,
        registry: MonitorRegistry,
        audit: AuditChain,
        emit: Callable[[BusEvent], None],
    ) -> None:
        self.registry = registry
        self.audit = audit
        self.emit = emit
        self.step = 0
        self.charts: dict[str, PatientChart] = {}

    def reset(self) -> dict:
        self.charts = seed_demo(self.registry)
        self.audit.clear()
        self.step = 0
        self.audit.append(
            actor="demo-runner",
            role=Role.CHARGE_NURSE,
            action="demo_reset",
            resource="waiting_room",
            outcome="success",
            metadata={"patients": len(self.charts)},
        )
        self.emit(
            BusEvent(
                type="demo_status",
                payload={
                    "step": 0,
                    "title": "Waiting room baseline established",
                    "detail": "Three consented patients are independently monitored.",
                },
            )
        )
        return {"step": self.step, "complete": False, "patients": len(self.charts)}

    def advance(self) -> dict:
        if not self.charts:
            self.reset()
        self.step = min(self.step + 1, 5)
        if self.step == 1:
            event = PerceptionEvent(
                ts=time.time(),
                modality="vision",
                kind="slump",
                confidence=0.72,
                track_id=101,
                meta={"duration_seconds": 74, "baseline_delta": 0.46},
            )
            title = "Posture change detected in seat A3"
            detail = "Vigil opens an accessible check-in before paging."
            self._ingest(event)
        elif self.step == 2:
            event = PerceptionEvent(
                ts=time.time(),
                modality="audio",
                kind="labored_breathing",
                confidence=0.84,
                track_id=101,
                meta={"seat_zone": "A3"},
            )
            title = "Second modality corroborates deterioration"
            detail = "Chart risk and two independent signals raise Maria Vega to ESI 2."
            self._ingest(event)
        elif self.step == 3:
            event = PerceptionEvent(
                ts=time.time(),
                modality="vision",
                kind="agitation",
                confidence=0.44,
                track_id=103,
                meta={"context": "animated phone conversation"},
            )
            title = "Low-confidence movement is routed to check-in"
            detail = "Jordan Park is not paged from a single ambiguous signal."
            self._ingest(event)
        elif self.step == 4:
            event = PerceptionEvent(
                ts=time.time(),
                modality="vision",
                kind="companion_alarm",
                confidence=0.91,
                track_id=102,
                meta={"seat_zone": "B1", "movement": "toward_staff"},
            )
            title = "Companion alarm routes help to seat B1"
            detail = "A family member seeking staff is treated as a hard safety signal."
            self._ingest(event)
        else:
            title = "Replay complete"
            detail = "Acknowledge or label alerts to close the loop and populate the audit chain."

        self.emit(
            BusEvent(
                type="demo_status",
                payload={"step": self.step, "title": title, "detail": detail},
            )
        )
        return {"step": self.step, "complete": self.step >= 5, "title": title, "detail": detail}

    def _ingest(self, event: PerceptionEvent) -> None:
        self.emit(BusEvent(type="perception", payload=event.model_dump(mode="json")))
        result = self.registry.ingest(event)
        if result.safety_only:
            if result.fused:
                self.emit(
                    BusEvent(
                        type="safety_alert",
                        payload={
                            "track_id": event.track_id,
                            "summary": result.fused.summary,
                            "clinical_context": None,
                        },
                    )
                )
            return
        if result.patient_id is None or result.fused is None:
            return

        chart = self.charts[result.patient_id]
        monitor = self.registry.monitor(result.patient_id)
        if monitor is None:
            return
        decision = decide_tier_zero(chart, monitor, result.fused)
        self.registry.apply_decision(decision)
        alert = self.registry.create_or_update_alert(decision, result.fused)
        self.emit(BusEvent(type="fused", payload=result.fused.model_dump(mode="json")))
        self.emit(
            BusEvent(
                type="reasoning_start",
                payload={
                    "patient": chart.name,
                    "patient_id": chart.patient_id,
                    "summary": result.fused.summary,
                    "prior_esi": decision.prior_esi,
                    "tier": decision.reasoning_tier,
                },
            )
        )
        self.emit(BusEvent(type="reasoning_delta", payload={"text": decision.rationale}))
        self.emit(BusEvent(type="decision", payload=decision.model_dump(mode="json")))
        self.emit(BusEvent(type="alert", payload=alert.model_dump(mode="json")))
        self.audit.append(
            actor="vigil-tier-0",
            role="system",
            action="retriage_decision",
            resource=f"patient:{chart.patient_id}",
            outcome="escalated" if decision.escalate else "held",
            metadata={
                "patient_id": chart.patient_id,
                "prior_esi": decision.prior_esi,
                "new_esi": decision.new_esi,
                "input_snapshot_hash": decision.input_snapshot_hash,
            },
        )
