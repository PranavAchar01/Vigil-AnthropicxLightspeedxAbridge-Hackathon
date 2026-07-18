"""Deterministic intake safety floors for the integration-free demo path."""

from __future__ import annotations

from pydantic import BaseModel, Field


class IntakeRequest(BaseModel):
    name: str
    age: int | None = Field(default=None, ge=0, le=125)
    chief_complaint: str
    proposed_esi: int = Field(default=4, ge=1, le=5)
    symptoms: list[str] = Field(default_factory=list)
    medications: list[str] = Field(default_factory=list)
    conditions: list[str] = Field(default_factory=list)
    red_flags: list[str] = Field(default_factory=list)
    preferred_language: str = "English"
    accessibility_mode: str = "voice_and_text"
    consent: bool = True


class IntakeResult(BaseModel):
    name: str
    initial_esi: int
    proposed_esi: int
    floor_applied: bool
    floor_reasons: list[str]
    risk_factors: list[str]
    watch_list: list[str]
    preferred_language: str
    accessibility_mode: str
    consent: bool


def assess_intake(request: IntakeRequest) -> IntakeResult:
    text = " ".join(
        [request.chief_complaint]
        + request.symptoms
        + request.medications
        + request.conditions
        + request.red_flags
    ).lower()
    floor = 5
    reasons: list[str] = []

    def apply(level: int, reason: str) -> None:
        nonlocal floor
        floor = min(floor, level)
        reasons.append(reason)

    if any(term in text for term in ("anaphylaxis", "not breathing", "unresponsive")):
        apply(1, "Immediate life threat reported")
    if request.age is not None and request.age > 40 and "chest pain" in text:
        apply(2, "Chest pain in a patient older than 40")
    if any(
        term in text for term in ("face droop", "arm weakness", "slurred speech", "fast positive")
    ):
        apply(2, "Possible stroke symptom reported")
    anticoagulant = any(
        term in text for term in ("apixaban", "warfarin", "rivaroxaban", "blood thinner")
    )
    if anticoagulant and any(
        term in text for term in ("head strike", "hit my head", "head injury")
    ):
        apply(2, "Head strike while taking an anticoagulant")
    if any(
        term in text
        for term in ("cannot breathe", "can't breathe", "respiratory distress", "stridor")
    ):
        apply(2, "Respiratory distress reported")
    if any(term in text for term in ("syncope", "fainted", "passed out")):
        apply(3, "Syncope reported")

    initial_esi = min(request.proposed_esi, floor)
    risks: list[str] = []
    if anticoagulant:
        risks.append("anticoagulated")
    if any(term in text for term in ("coronary", "heart failure", "cardiac", "chest pain")):
        risks.append("cardiac_history")
    if any(term in text for term in ("copd", "asthma", "pneumonia", "shortness of breath")):
        risks.append("respiratory_risk")
    if any(term in text for term in ("diabetes", "insulin")):
        risks.append("diabetes")

    watch: list[str] = ["change from personal movement and posture baseline"]
    if "cardiac_history" in risks:
        watch.extend(["chest guarding", "postural decline", "reported chest symptoms"])
    if "respiratory_risk" in risks:
        watch.extend(["labored breathing", "speech interruption", "oxygenation concern"])
    if "anticoagulated" in risks:
        watch.extend(["fall", "head strike", "gait instability"])

    return IntakeResult(
        name=request.name,
        initial_esi=initial_esi,
        proposed_esi=request.proposed_esi,
        floor_applied=initial_esi < request.proposed_esi,
        floor_reasons=list(dict.fromkeys(reasons)),
        risk_factors=list(dict.fromkeys(risks)),
        watch_list=list(dict.fromkeys(watch)),
        preferred_language=request.preferred_language,
        accessibility_mode=request.accessibility_mode,
        consent=request.consent,
    )
