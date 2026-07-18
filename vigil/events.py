"""The typed event model that flows through Vigil's pipeline.

perception (vision + audio) -> fusion -> reasoning (triage) -> escalation (action)
Everything published to the dashboard event bus is one of these, serialized.
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


def _now() -> float:
    return time.time()


class Modality(str, Enum):
    VISION = "vision"
    AUDIO = "audio"


class Severity(str, Enum):
    HARD = "hard"  # unambiguous emergency (scream + fall) -> page immediately
    SOFT = "soft"  # ambiguous (prolonged stillness, slump) -> voice check-in first


class Action(str, Enum):
    PAGE_IMMEDIATELY = "page_immediately"
    VOICE_CHECKIN = "voice_checkin"
    HOLD = "hold"


# Discrete signals the perception layer can raise.
PerceptionKind = Literal[
    "fall",  # rapid drop to ground / horizontal (hard)
    "collapse",  # slump/drop then motionless — faint/syncope (hard)
    "seizure",  # rapid oscillatory convulsion (hard)
    "unresponsive",  # prolonged stillness — LOC / passed out (hard)
    "scream",  # audio distress vocalization (hard)
    "motionless",  # brief stillness — early soft signal
    "slump",  # sustained posture degradation (soft)
    "agitation",  # restlessness / pacing (soft)
    "chest_clutch",  # hands to chest / throat / head — distress gesture (soft)
]


class PerceptionEvent(BaseModel):
    """A single signal from one sensor at one moment."""

    ts: float = Field(default_factory=_now)
    modality: Modality
    kind: PerceptionKind
    confidence: float = Field(ge=0.0, le=1.0)
    track_id: int = 0
    meta: dict[str, Any] = Field(default_factory=dict)


class FusedEvent(BaseModel):
    """One or more perception signals fused into a triage-worthy event."""

    ts: float = Field(default_factory=_now)
    track_id: int = 0
    kinds: list[PerceptionKind]
    severity: Severity
    confidence: float = Field(ge=0.0, le=1.0)
    summary: str  # human-readable, e.g. "scream + collapse, on ground 3s"


class TriageDecision(BaseModel):
    """Claude's structured, chart-grounded re-triage decision. Monotonic.

    Grounded in the Emergency Severity Index (ESI) 4-decision-point algorithm
    (ESI Implementation Handbook, AHRQ / Emergency Nurses Association). Every
    grade reports WHICH decision point (A/B/C/D) and WHICH specific criterion
    drove it, so the call is auditable against the published rubric.
    """

    patient_id: str
    prior_esi: int = Field(ge=1, le=5)
    new_esi: int = Field(ge=1, le=5)
    escalate: bool
    action: Action
    # ESI v4 decision point that drove the grade: A=life-saving intervention,
    # B=high-risk/altered-mental-status/severe-distress, C=resource need, D=danger-zone vitals.
    esi_decision_point: str = ""
    esi_criteria: str = ""  # the one specific criterion matched, e.g. "charted HR 118 > 100"
    rationale: str  # cites the chart; shown in the reasoning trace
    spoken_summary: str  # <=10s of speech; what ElevenLabs says to the nurse

    def is_monotonic(self) -> bool:
        """ESI 1 is most acute, 5 least. Escalation must never raise the number."""
        return self.new_esi <= self.prior_esi


class InitialTriageDecision(BaseModel):
    """First-contact ESI grade computed from a spoken intake (+ optional chart).

    Unlike TriageDecision (re-triage, monotonic, only-escalates), this is the
    INITIAL acuity assignment: it runs the FULL ESI v4 algorithm including
    Decision C (resource prediction). It is decision-SUPPORT — a clinician
    confirms it (`needs_confirmation`). On model failure it fails SAFE to ESI 2
    (never silently assigns a low acuity to an ungraded patient).
    """

    patient_id: str = ""
    chief_complaint: str = ""
    esi: int = Field(ge=1, le=5)
    # ESI v4 decision point that assigned the level: A/B/C/D (see prompts.py).
    esi_decision_point: str = ""
    esi_criteria: str = ""
    predicted_resources: list[str] = Field(default_factory=list)  # Decision C
    danger_zone_vitals: bool = False  # Decision D
    red_flags: list[str] = Field(default_factory=list)
    rationale: str = ""
    spoken_summary: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    needs_confirmation: bool = True  # human-in-the-loop on the risky first grade
    transcript: str = ""  # the intake speech that was graded


class EscalationAction(BaseModel):
    """The act taken as a result of a decision."""

    ts: float = Field(default_factory=_now)
    kind: Literal["nurse_call", "patient_checkin", "none"]
    target: str = ""  # phone number for nurse_call
    message: str = ""  # what was spoken
    status: Literal["pending", "dialing", "connected", "completed", "failed", "skipped"] = "pending"
    provider_ref: str = ""  # ElevenLabs conversation/call id


class BusEvent(BaseModel):
    """Envelope broadcast to the dashboard over WebSocket."""

    ts: float = Field(default_factory=_now)
    # perception | fused | reasoning_start | reasoning_delta | decision |
    # call_status | escalation | note | status
    type: str
    payload: dict[str, Any]
