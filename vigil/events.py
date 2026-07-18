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
    "fall",
    "collapse",
    "scream",
    "motionless",
    "slump",
    "agitation",
    "chest_clutch",
    "gait_instability",
    "labored_breathing",
    "distress_phrase",
    "non_response",
    "companion_alarm",
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
    """Claude's structured, chart-grounded re-triage decision. Monotonic."""

    patient_id: str
    prior_esi: int = Field(ge=1, le=5)
    new_esi: int = Field(ge=1, le=5)
    escalate: bool
    action: Action
    rationale: str  # cites the chart; shown in the reasoning trace
    spoken_summary: str  # <=10s of speech; what ElevenLabs says to the nurse
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list)
    input_snapshot_hash: str = ""
    reasoning_tier: Literal["tier_0", "tier_1", "fail_safe"] = "tier_1"

    def is_monotonic(self) -> bool:
        """ESI 1 is most acute, 5 least. Escalation must never raise the number."""
        return self.new_esi <= self.prior_esi


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
