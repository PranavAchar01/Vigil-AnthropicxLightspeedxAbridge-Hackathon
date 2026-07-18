"""Role-filtered command-center API and deterministic demo controls."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

from vigil.demo import DemoController
from vigil.events import BusEvent
from vigil.intake import IntakeRequest, assess_intake
from vigil.monitoring import MonitorRegistry
from vigil.security import (
    AuditChain,
    BreakGlassManager,
    Role,
    has_scope,
    parse_role,
    queue_item_for_role,
)
from vigil.server.bus import EventBus


class AckRequest(BaseModel):
    actor: str = "Charge RN"


class FeedbackRequest(BaseModel):
    actor: str = "Charge RN"
    outcome: str


class BindRequest(BaseModel):
    patient_id: str | None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    actor: str = "Charge RN"


class BreakGlassRequest(BaseModel):
    actor: str
    patient_id: str
    reason: str


class OverrideRequest(BaseModel):
    actor: str = "Charge RN"
    new_esi: int = Field(ge=1, le=5)


class AssistRequest(BaseModel):
    actor: str = Field(default="Front desk", min_length=2, max_length=80)
    seat: str = Field(min_length=1, max_length=12)
    reason: str = Field(default="Patient or companion requested clinical help", min_length=8, max_length=160)


class ReplayRequest(BaseModel):
    interval_seconds: float = Field(default=1.1, ge=0.15, le=10)


def build_command_router(
    registry: MonitorRegistry,
    audit: AuditChain,
    break_glass: BreakGlassManager,
    demo: DemoController,
    bus: EventBus,
) -> APIRouter:
    router = APIRouter(prefix="/api/v1")
    replay_tasks: set[asyncio.Task] = set()

    def role_from_header(value: str) -> Role:
        try:
            return parse_role(value)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    def queue_payload(role: Role) -> list[dict]:
        items: list[dict] = []
        for monitor, priority in registry.queue():
            chart = registry.chart(monitor.patient_id)
            if chart is None:
                continue
            items.append(
                queue_item_for_role(
                    monitor,
                    chart,
                    registry.active_alert(monitor.patient_id),
                    priority,
                    role,
                )
            )
        return items

    @router.get("/session")
    async def session(x_vigil_role: str = Header(default=Role.CHARGE_NURSE.value)):
        role = role_from_header(x_vigil_role)
        audit.append(
            actor=f"demo-{role.value}",
            role=role,
            action="queue_read",
            resource="waiting_room",
            outcome="success",
            metadata={
                "server_side_redaction": role in {Role.FRONT_DESK, Role.SECURITY, Role.COMPLIANCE}
            },
        )
        response = {
            "role": role.value,
            "scopes": sorted(has for has in _scopes(role)),
            "queue": queue_payload(role),
            "alert_budget": registry.alert_budget(),
            "demo": {"step": demo.step, "total_steps": 5},
            "audit_verified": audit.verify(),
        }
        if role in {Role.COMPLIANCE, Role.CHARGE_NURSE}:
            response["audit"] = audit.blocks(role, limit=24 if role == Role.CHARGE_NURSE else 100)
        return response

    @router.get("/queue")
    async def queue(x_vigil_role: str = Header(default=Role.CHARGE_NURSE.value)):
        role = role_from_header(x_vigil_role)
        return {"role": role.value, "patients": queue_payload(role)}

    @router.get("/patients/{patient_id}")
    async def patient(
        patient_id: str,
        x_vigil_role: str = Header(default=Role.CHARGE_NURSE.value),
    ):
        role = role_from_header(x_vigil_role)
        monitor = registry.monitor(patient_id)
        chart = registry.chart(patient_id)
        if monitor is None or chart is None:
            raise HTTPException(status_code=404, detail="unknown patient")
        if not has_scope(role, "chart:read"):
            audit.append(
                actor=f"demo-{role.value}",
                role=role,
                action="chart_read",
                resource=f"patient:{patient_id}",
                outcome="denied",
                metadata={"required_scope": "chart:read"},
            )
            raise HTTPException(status_code=403, detail="restricted_by_scope")
        audit.append(
            actor=f"demo-{role.value}",
            role=role,
            action="chart_read",
            resource=f"patient:{patient_id}",
            outcome="success",
        )
        return queue_item_for_role(
            monitor,
            chart,
            registry.active_alert(patient_id),
            next((score for item, score in registry.queue() if item.patient_id == patient_id), 0),
            role,
        )

    @router.post("/demo/reset")
    async def reset_demo(x_vigil_role: str = Header(default=Role.CHARGE_NURSE.value)):
        role = role_from_header(x_vigil_role)
        if role != Role.CHARGE_NURSE:
            raise HTTPException(status_code=403, detail="charge_nurse role required")
        return demo.reset()

    @router.post("/demo/advance")
    async def advance_demo(x_vigil_role: str = Header(default=Role.CHARGE_NURSE.value)):
        role = role_from_header(x_vigil_role)
        if role != Role.CHARGE_NURSE:
            raise HTTPException(status_code=403, detail="charge_nurse role required")
        return demo.advance()

    async def run_replay(interval: float) -> None:
        demo.reset()
        while demo.step < 5:
            await asyncio.sleep(interval)
            demo.advance()

    @router.post("/demo/replay")
    async def replay_demo(
        request: ReplayRequest,
        x_vigil_role: str = Header(default=Role.CHARGE_NURSE.value),
    ):
        role = role_from_header(x_vigil_role)
        if role != Role.CHARGE_NURSE:
            raise HTTPException(status_code=403, detail="charge_nurse role required")
        task = asyncio.create_task(run_replay(request.interval_seconds))
        replay_tasks.add(task)
        task.add_done_callback(replay_tasks.discard)
        return {"started": True, "steps": 5, "interval_seconds": request.interval_seconds}

    @router.post("/alerts/{alert_id}/acknowledge")
    async def acknowledge(
        alert_id: str,
        request: AckRequest,
        x_vigil_role: str = Header(default=Role.CHARGE_NURSE.value),
    ):
        role = role_from_header(x_vigil_role)
        if not has_scope(role, "escalate:ack"):
            raise HTTPException(status_code=403, detail="escalate:ack scope required")
        try:
            alert = registry.acknowledge(alert_id, request.actor)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="unknown alert") from exc
        audit.append(
            actor=request.actor,
            role=role,
            action="alert_acknowledge",
            resource=f"patient:{alert.patient_id}",
            outcome="success",
            metadata={"alert_id": alert.alert_id},
        )
        bus.publish(BusEvent(type="alert_acknowledged", payload=alert.model_dump(mode="json")))
        return alert

    @router.post("/alerts/{alert_id}/feedback")
    async def feedback(
        alert_id: str,
        request: FeedbackRequest,
        x_vigil_role: str = Header(default=Role.CHARGE_NURSE.value),
    ):
        role = role_from_header(x_vigil_role)
        if not has_scope(role, "escalate:ack"):
            raise HTTPException(status_code=403, detail="escalate:ack scope required")
        try:
            alert = registry.feedback(alert_id, request.outcome)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="unknown alert") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        audit.append(
            actor=request.actor,
            role=role,
            action="alert_feedback",
            resource=f"patient:{alert.patient_id}",
            outcome=request.outcome,
            metadata={"alert_id": alert.alert_id},
        )
        bus.publish(BusEvent(type="alert_resolved", payload=alert.model_dump(mode="json")))
        return alert

    @router.post("/tracks/{track_id}/bind")
    async def bind_track(
        track_id: int,
        request: BindRequest,
        x_vigil_role: str = Header(default=Role.CHARGE_NURSE.value),
    ):
        role = role_from_header(x_vigil_role)
        if role != Role.CHARGE_NURSE:
            raise HTTPException(status_code=403, detail="charge_nurse role required")
        binding = registry.bind_track(track_id, request.patient_id, confidence=request.confidence)
        audit.append(
            actor=request.actor,
            role=role,
            action="track_bind",
            resource=f"patient:{request.patient_id}" if request.patient_id else f"track:{track_id}",
            outcome=binding.state.value,
            metadata={"track_id": track_id, "patient_id": request.patient_id},
        )
        bus.publish(BusEvent(type="track_binding", payload=binding.model_dump(mode="json")))
        return binding

    @router.post("/patients/{patient_id}/esi")
    async def override_esi(
        patient_id: str,
        request: OverrideRequest,
        x_vigil_role: str = Header(default=Role.CHARGE_NURSE.value),
    ):
        role = role_from_header(x_vigil_role)
        if not has_scope(role, "esi:override"):
            raise HTTPException(status_code=403, detail="esi:override scope required")
        try:
            monitor = registry.override_esi(patient_id, request.new_esi)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="unknown patient") from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        audit.append(
            actor=request.actor,
            role=role,
            action="esi_override",
            resource=f"patient:{patient_id}",
            outcome="success",
            metadata={"new_esi": monitor.current_esi},
        )
        return monitor

    @router.post("/operations/medical-assist")
    async def request_medical_assist(
        request: AssistRequest,
        x_vigil_role: str = Header(default=Role.FRONT_DESK.value),
    ):
        role = role_from_header(x_vigil_role)
        if not has_scope(role, "operations:assist"):
            raise HTTPException(status_code=403, detail="operations:assist scope required")
        try:
            alert = registry.request_medical_assist(request.seat, request.actor)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="unknown waiting-room seat") from exc
        audit.append(
            actor=request.actor,
            role=role,
            action="medical_assist_request",
            resource=f"patient:{alert.patient_id}",
            outcome="routed",
            reason=request.reason,
            metadata={"seat": request.seat, "alert_id": alert.alert_id},
        )
        payload = {
            "action_id": alert.alert_id,
            "seat": request.seat,
            "state": "page_pending",
            "routed_to": "charge_nurse",
        }
        bus.publish(BusEvent(type="medical_assist_requested", payload=payload))
        return payload

    @router.post("/break-glass")
    async def create_break_glass(
        request: BreakGlassRequest,
        x_vigil_role: str = Header(default=Role.CHARGE_NURSE.value),
    ):
        role = role_from_header(x_vigil_role)
        if not has_scope(role, "break_glass:grant"):
            raise HTTPException(status_code=403, detail="break_glass:grant scope required")
        try:
            grant = break_glass.grant(request.actor, request.patient_id, request.reason)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        audit.append(
            actor=request.actor,
            role=role,
            action="break_glass",
            resource=f"patient:{request.patient_id}",
            outcome="granted",
            reason=request.reason,
            metadata={"expires_at": grant.expires_at, "grant_id": grant.grant_id},
        )
        bus.publish(
            BusEvent(
                type="compliance_alert",
                payload={
                    "action": "break_glass",
                    "actor": request.actor,
                    "patient_ref": request.patient_id,
                    "expires_at": grant.expires_at,
                },
            )
        )
        return grant

    @router.post("/intake")
    async def intake(
        request: IntakeRequest,
        x_vigil_role: str = Header(default=Role.TRIAGE_NURSE.value),
    ):
        role = role_from_header(x_vigil_role)
        if role not in {Role.TRIAGE_NURSE, Role.CHARGE_NURSE}:
            raise HTTPException(status_code=403, detail="triage scope required")
        result = assess_intake(request)
        audit.append(
            actor=f"demo-{role.value}",
            role=role,
            action="intake_floor",
            resource="patient:new",
            outcome="floor_applied" if result.floor_applied else "accepted",
            metadata={"initial_esi": result.initial_esi, "reasons": result.floor_reasons},
        )
        return result

    @router.get("/audit")
    async def get_audit(x_vigil_role: str = Header(default=Role.COMPLIANCE.value)):
        role = role_from_header(x_vigil_role)
        if role not in {Role.COMPLIANCE, Role.CHARGE_NURSE}:
            raise HTTPException(status_code=403, detail="audit scope required")
        return {"verification": audit.verify(), "blocks": audit.blocks(role)}

    @router.get("/audit/verify")
    async def verify_audit(x_vigil_role: str = Header(default=Role.COMPLIANCE.value)):
        role = role_from_header(x_vigil_role)
        if role not in {Role.COMPLIANCE, Role.CHARGE_NURSE}:
            raise HTTPException(status_code=403, detail="audit scope required")
        return audit.verify()

    return router


def _scopes(role: Role) -> set[str]:
    from vigil.security import ROLE_SCOPES

    return set(ROLE_SCOPES[role])
