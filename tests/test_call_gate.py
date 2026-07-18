"""The outbound-call rate limiter — a Twilio trial must never get spam-dialed."""

from __future__ import annotations

from vigil.escalation import elevenlabs_call as ec


def test_one_call_per_run(monkeypatch):
    monkeypatch.setattr(ec, "MAX_CALLS", 1)
    monkeypatch.setattr(ec, "COOLDOWN_S", 0.0)
    ec.reset_call_gate()
    ok, _ = ec._reserve_call("+15551234567")
    assert ok
    ok2, why = ec._reserve_call("+15551234567")
    assert not ok2 and "max" in why  # second call is suppressed


def test_failed_call_releases_slot(monkeypatch):
    monkeypatch.setattr(ec, "MAX_CALLS", 1)
    monkeypatch.setattr(ec, "COOLDOWN_S", 0.0)
    ec.reset_call_gate()
    assert ec._reserve_call("+15551234567")[0]
    ec._release_call("+15551234567")  # the dial failed → slot returned
    assert ec._reserve_call("+15551234567")[0]  # a later real incident can still call


def test_cooldown_spaces_repeat_calls(monkeypatch):
    monkeypatch.setattr(ec, "MAX_CALLS", 10)
    monkeypatch.setattr(ec, "COOLDOWN_S", 999.0)
    ec.reset_call_gate()
    assert ec._reserve_call("+15551234567")[0]
    ok, why = ec._reserve_call("+15551234567")
    assert not ok and "cooldown" in why
