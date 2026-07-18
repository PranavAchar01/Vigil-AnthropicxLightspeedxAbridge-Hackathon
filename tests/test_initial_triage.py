"""Initial ESI triage (Gap B): the code-enforced ESI-algorithm invariants and the
conservative fail-safe, driven through _finalize/_fail_safe with dict inputs — no
camera, no microphone, no live API (mirrors the re-triage tests in test_core.py)."""

from __future__ import annotations

import types

import pytest

from vigil.reasoning import initial_triage


def _raw(esi: int, point: str, danger: bool = False, **over) -> dict:
    d = {
        "esi": esi,
        "esi_decision_point": point,
        "esi_criteria": "c",
        "predicted_resources": [],
        "danger_zone_vitals": danger,
        "red_flags": [],
        "chief_complaint": "cc",
        "confidence": 0.8,
        "rationale": "r",
        "spoken_summary": "s",
    }
    d.update(over)
    return d


def test_decision_a_forces_esi_1():
    # Decision A is the only path to ESI 1 and always yields 1, even if the model
    # emitted a softer number alongside point A.
    d = initial_triage._finalize(_raw(3, "A"), "not breathing", None)
    assert d.esi == 1 and d.esi_decision_point == "A"


def test_danger_zone_uptriages_3_to_2():
    d = initial_triage._finalize(_raw(3, "C", danger=True), "fever, feels faint", None)
    assert d.esi == 2 and d.esi_decision_point == "D"


def test_danger_zone_only_applies_at_the_esi3_node():
    # Handbook fidelity: Decision D up-triages a would-be 3, not a 4/5.
    d = initial_triage._finalize(_raw(4, "C", danger=True), "sprained ankle", None)
    assert d.esi == 4  # unchanged — D is evaluated at the ESI-3 node only


def test_resource_grade_passes_through():
    d = initial_triage._finalize(
        _raw(3, "C", predicted_resources=["ECG", "troponin", "CT"]), "chest pain", None
    )
    assert d.esi == 3 and d.esi_decision_point == "C"
    assert d.predicted_resources == ["ECG", "troponin", "CT"]


def test_transcript_and_confirmation_flag_preserved():
    d = initial_triage._finalize(_raw(4, "C"), "mild sore throat for two days", None)
    assert d.transcript == "mild sore throat for two days"
    assert d.needs_confirmation is True  # human confirms the first grade


def test_fail_safe_is_conservative_and_flagged():
    d = initial_triage._fail_safe("garbled intake", None)
    assert d.esi == 2  # over-triage on failure, never silently low
    assert d.confidence == 0.0 and d.needs_confirmation is True
    assert d.red_flags  # surfaces that the grader was unavailable


def test_empty_transcript_rejected(monkeypatch):
    monkeypatch.setattr(initial_triage, "settings", types.SimpleNamespace(anthropic_api_key="k"))
    with pytest.raises(ValueError):
        initial_triage.grade("   ", None)


def test_grade_requires_api_key(monkeypatch):
    monkeypatch.setattr(initial_triage, "settings", types.SimpleNamespace(anthropic_api_key=""))
    with pytest.raises(initial_triage.ReasoningNotConfigured):
        initial_triage.grade("chest pain", None)
