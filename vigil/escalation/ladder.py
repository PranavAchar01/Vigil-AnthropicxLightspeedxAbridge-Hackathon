"""The severity-aware escalation ladder — Vigil's false-positive defense.

The ladder never de-escalates; it only decides *how* to add attention:

  page_immediately  ->  call the charge nurse now (hard emergencies)
  voice_checkin     ->  talk to the PATIENT first; page only on a bad/absent
                        answer (ambiguous soft signals)
  hold              ->  keep watching

Call execution is injected (Protocol handlers) so this stays pure policy and
testable without touching ElevenLabs. escalation/elevenlabs_call.py provides the
concrete handlers.
"""

from __future__ import annotations

from typing import Protocol

from vigil.events import Action, EscalationAction, TriageDecision


class CheckinResult:
    """Outcome of a patient voice check-in."""

    def __init__(self, answered: bool, reassuring: bool, transcript: str = "") -> None:
        self.answered = answered
        self.reassuring = reassuring  # patient responded AND the answer was okay
        self.transcript = transcript

    @property
    def needs_escalation(self) -> bool:
        # bad answer, or no answer at all -> pull in a human
        return (not self.answered) or (not self.reassuring)


class EscalationHandlers(Protocol):
    def call_nurse(self, decision: TriageDecision) -> EscalationAction: ...
    def check_in_patient(self, decision: TriageDecision) -> CheckinResult: ...


def run_ladder(decision: TriageDecision, handlers: EscalationHandlers) -> list[EscalationAction]:
    """Execute the ladder for one decision. Returns the actions taken, in order."""
    actions: list[EscalationAction] = []

    if not decision.escalate or decision.action == Action.HOLD:
        actions.append(EscalationAction(kind="none", status="skipped"))
        return actions

    if decision.action == Action.PAGE_IMMEDIATELY:
        actions.append(handlers.call_nurse(decision))
        return actions

    if decision.action == Action.VOICE_CHECKIN:
        result = handlers.check_in_patient(decision)
        actions.append(
            EscalationAction(
                kind="patient_checkin",
                message=result.transcript,
                status="completed" if result.answered else "failed",
            )
        )
        if result.needs_escalation:
            # bad or absent answer -> escalate to the nurse
            actions.append(handlers.call_nurse(decision))
        return actions

    return actions
