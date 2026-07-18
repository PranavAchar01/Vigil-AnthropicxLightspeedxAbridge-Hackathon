"""Role scopes, field-level redaction, break-glass grants, and audit chaining."""

from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from vigil.chart import PatientChart
from vigil.monitoring import AlertRecord, PatientMonitor


class Role(str, Enum):
    CHARGE_NURSE = "charge_nurse"
    ATTENDING = "attending"
    TRIAGE_NURSE = "triage_nurse"
    FRONT_DESK = "front_desk"
    SECURITY = "security"
    COMPLIANCE = "compliance"


ROLE_SCOPES: dict[Role, frozenset[str]] = {
    Role.CHARGE_NURSE: frozenset(
        {"video:view", "chart:read", "reason:read", "escalate:ack", "esi:override", "queue:read"}
    ),
    Role.ATTENDING: frozenset({"chart:read", "reason:read", "esi:override", "queue:read"}),
    Role.TRIAGE_NURSE: frozenset({"chart:read", "queue:read"}),
    Role.FRONT_DESK: frozenset({"queue:read:limited"}),
    Role.SECURITY: frozenset({"alert:read:nonclinical"}),
    Role.COMPLIANCE: frozenset({"audit:read"}),
}


def parse_role(value: str | None) -> Role:
    try:
        return Role(value or Role.CHARGE_NURSE.value)
    except ValueError as exc:
        raise ValueError(f"unknown Vigil role: {value}") from exc


def has_scope(role: Role, scope: str) -> bool:
    return scope in ROLE_SCOPES[role]


def require_scope(role: Role, scope: str) -> None:
    if not has_scope(role, scope):
        raise PermissionError(f"{role.value} lacks {scope}")


def patient_ref(patient_id: str) -> str:
    return "pt_" + hashlib.sha256(patient_id.encode()).hexdigest()[:10]


def queue_item_for_role(
    monitor: PatientMonitor,
    chart: PatientChart,
    alert: AlertRecord | None,
    priority: float,
    role: Role,
    *,
    now: float | None = None,
) -> dict[str, Any]:
    current = now if now is not None else time.time()
    wait_minutes = max(0, round((current - monitor.wait_started_at) / 60))
    flagged = monitor.escalation_state.value in {"checkin", "page_pending"}

    if role == Role.COMPLIANCE:
        return {
            "patient_ref": patient_ref(monitor.patient_id),
            "wait_minutes": wait_minutes,
            "flagged": flagged,
            "decision_hash": None,
            "redacted": True,
        }
    if role == Role.SECURITY:
        return {
            "patient_ref": patient_ref(monitor.patient_id),
            "seat": monitor.seat,
            "medical_assist_needed": flagged,
            "alert_type": "medical assist" if flagged else None,
            "redacted": True,
        }
    if role == Role.FRONT_DESK:
        return {
            "patient_id": monitor.patient_id,
            "name": monitor.name,
            "seat": monitor.seat,
            "wait_minutes": wait_minutes,
            "flagged": flagged,
            "redacted": True,
        }

    base: dict[str, Any] = {
        "patient_id": monitor.patient_id,
        "name": monitor.name,
        "age": monitor.age,
        "gender": monitor.gender,
        "seat": monitor.seat,
        "visit": monitor.visit,
        "initial_esi": monitor.initial_esi,
        "current_esi": monitor.current_esi,
        "wait_minutes": wait_minutes,
        "priority": priority,
        "status": monitor.escalation_state.value,
        "latest_signal": monitor.latest_signal,
        "baseline_deviation": round(monitor.baseline_deviation, 2),
        "risk_factors": monitor.risk_factors,
        "reassessment_due": (current - monitor.last_assessed_at) / 60
        >= monitor.reassessment_target_minutes,
        "preferred_language": monitor.preferred_language,
        "accessibility_mode": monitor.accessibility_mode,
        "alert": alert.model_dump() if alert else None,
        "redacted": False,
    }
    if has_scope(role, "chart:read"):
        base["chart"] = {
            "conditions": chart.active_conditions,
            "medications": chart.active_medications,
            "vitals": {
                key: {"value": vital.value, "unit": vital.unit, "at": vital.at}
                for key, vital in chart.latest_vitals.items()
            },
        }
    if not has_scope(role, "reason:read") and base.get("alert"):
        base["alert"].pop("evidence", None)
    if not has_scope(role, "video:view"):
        base["video_mode"] = "skeleton_only"
    else:
        base["video_mode"] = "derived_pose"
    return base


class AuditBlock(BaseModel):
    index: int
    audit_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    ts: float = Field(default_factory=time.time)
    actor: str
    role: str
    action: str
    resource: str
    outcome: str
    reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    prev_hash: str
    hash: str


class AuditChain:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._blocks: list[AuditBlock] = []

    def clear(self) -> None:
        with self._lock:
            self._blocks.clear()

    def append(
        self,
        *,
        actor: str,
        role: Role | str,
        action: str,
        resource: str,
        outcome: str,
        reason: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AuditBlock:
        with self._lock:
            index = len(self._blocks)
            prev_hash = self._blocks[-1].hash if self._blocks else "GENESIS"
            body = {
                "index": index,
                "audit_id": uuid.uuid4().hex[:12],
                "ts": time.time(),
                "actor": actor,
                "role": role.value if isinstance(role, Role) else role,
                "action": action,
                "resource": resource,
                "outcome": outcome,
                "reason": reason,
                "metadata": metadata or {},
                "prev_hash": prev_hash,
            }
            digest = _audit_hash(body)
            block = AuditBlock(**body, hash=digest)
            self._blocks.append(block)
            return block.model_copy(deep=True)

    def blocks(self, role: Role, *, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            selected = self._blocks[-limit:]
            payload = [block.model_dump() for block in selected]
        if role == Role.COMPLIANCE:
            for block in payload:
                block["resource"] = _redact_resource(block["resource"])
                block["metadata"] = _redact_metadata(block["metadata"])
        return payload

    def verify(self) -> dict[str, Any]:
        with self._lock:
            blocks = [block.model_dump() for block in self._blocks]
        previous = "GENESIS"
        for index, block in enumerate(blocks):
            claimed = block.pop("hash")
            if block["index"] != index or block["prev_hash"] != previous:
                return {"valid": False, "blocks": len(blocks), "failed_at": index}
            if _audit_hash(block) != claimed:
                return {"valid": False, "blocks": len(blocks), "failed_at": index}
            previous = claimed
        return {"valid": True, "blocks": len(blocks), "head": previous}


class BreakGlassGrant(BaseModel):
    grant_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    actor: str
    patient_id: str
    reason: str
    granted_at: float = Field(default_factory=time.time)
    expires_at: float

    def active(self, now: float | None = None) -> bool:
        return (now if now is not None else time.time()) < self.expires_at


class BreakGlassManager:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._grants: dict[str, BreakGlassGrant] = {}

    def grant(
        self, actor: str, patient_id: str, reason: str, *, ttl_seconds: int = 900
    ) -> BreakGlassGrant:
        if len(reason.strip()) < 8:
            raise ValueError("a specific break-glass reason is required")
        grant = BreakGlassGrant(
            actor=actor,
            patient_id=patient_id,
            reason=reason.strip(),
            expires_at=time.time() + ttl_seconds,
        )
        with self._lock:
            self._grants[grant.grant_id] = grant
        return grant.model_copy(deep=True)

    def active(self, grant_id: str, *, now: float | None = None) -> BreakGlassGrant | None:
        with self._lock:
            grant = self._grants.get(grant_id)
            if grant and grant.active(now):
                return grant.model_copy(deep=True)
        return None


def _audit_hash(body: dict[str, Any]) -> str:
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _redact_resource(resource: str) -> str:
    if resource.startswith("patient:"):
        return "patient:" + patient_ref(resource.split(":", 1)[1])
    return resource


def _redact_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in metadata.items():
        if key in {"patient_id", "name"}:
            if key == "patient_id":
                safe["patient_ref"] = patient_ref(str(value))
            continue
        safe[key] = value
    return safe
