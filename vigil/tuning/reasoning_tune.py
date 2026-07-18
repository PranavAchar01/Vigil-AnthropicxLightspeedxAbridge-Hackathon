"""Reasoning agent-loop evaluation: score Claude's re-triage against clinical
expectations, and measure latency. This is the eval that gates the agent loop —
it makes real (bounded) API calls. Cases pair a real chart with a fused event and
assert the decision is monotonic, escalates appropriately, and speaks a usable
handoff line.
"""

from __future__ import annotations

import time

from vigil.config import settings
from vigil.events import FusedEvent, Severity


def _fused(kinds, severity, summary, conf=0.9) -> FusedEvent:
    return FusedEvent(
        ts=time.time(), kinds=kinds, severity=severity, confidence=conf, summary=summary
    )


def build_cases():
    """(chart, fused, expectation) tuples from the demo cohort."""
    from vigil.chart import load_cohort

    if not settings.cohort_path.exists():
        return []
    charts = {c.name.split()[0].lower(): c for c in load_cohort(settings.cohort_path)}
    charts_list = list(charts.values())

    cases = []

    def add(chart, fused, expect):
        if chart is not None:
            cases.append((chart, fused, expect))

    # HARD events on acute charts → must escalate to high acuity.
    add(
        charts.get("pranav") or charts_list[0],
        _fused(["collapse"], Severity.HARD, "Collapse / faint — slumped then motionless"),
        {"escalate": True, "max_esi": 2},
    )
    add(
        charts.get("sahiel") or charts_list[0],
        _fused(["unresponsive"], Severity.HARD, "Unresponsive — prolonged motionlessness"),
        {"escalate": True, "max_esi": 2},
    )
    add(
        charts_list[0],
        _fused(["scream", "fall"], Severity.HARD, "Fall detected — patient on the ground + scream"),
        {"escalate": True, "max_esi": 2},
    )
    # SOFT event → monotonic, must not raise ESI number; a check-in is acceptable.
    add(
        charts_list[min(1, len(charts_list) - 1)],
        _fused(["slump"], Severity.SOFT, "Posture degraded / slumping"),
        {"escalate": None, "max_esi": 5},
    )
    add(
        charts_list[min(2, len(charts_list) - 1)],
        _fused(["chest_clutch"], Severity.SOFT, "Distress gesture — hand to chest"),
        {"escalate": None, "max_esi": 5},
    )
    return cases


def score_decision(chart, decision, expect) -> tuple[int, int, dict]:
    checks = {
        "monotonic": decision.new_esi <= chart.baseline_esi,
        "esi_ceiling": decision.new_esi <= expect["max_esi"],
        "rationale_nonempty": len(decision.rationale.strip()) > 10,
        "spoken_ok": 0 < len(decision.spoken_summary.strip()) <= 220,
    }
    if expect["escalate"] is True:
        checks["escalates"] = decision.escalate is True
    passed = sum(checks.values())
    return passed, len(checks), checks


def run_eval(limit: int | None = None) -> dict:
    """Run the real re-triage over the cases. Returns aggregate metrics."""
    from vigil.reasoning import triage

    cases = build_cases()
    if limit:
        cases = cases[:limit]
    if not cases:
        return {"status": "no cohort / cases", "cases": 0}

    total_p = total_c = 0
    latencies, details = [], []
    for chart, fused, expect in cases:
        t0 = time.time()
        try:
            decision = triage.decide(chart, fused)
        except Exception as e:  # noqa: BLE001
            details.append({"patient": chart.name, "error": repr(e)})
            total_c += 5
            continue
        dt = time.time() - t0
        latencies.append(dt)
        p, c, checks = score_decision(chart, decision, expect)
        total_p += p
        total_c += c
        details.append(
            {
                "patient": chart.name,
                "event": fused.summary,
                "esi": f"{decision.prior_esi}->{decision.new_esi}",
                "action": decision.action.value,
                "latency_s": round(dt, 2),
                "checks": checks,
            }
        )
    return {
        "cases": len(cases),
        "pass_rate": round(total_p / max(total_c, 1), 3),
        "checks_passed": total_p,
        "checks_total": total_c,
        "mean_latency_s": round(sum(latencies) / max(len(latencies), 1), 2),
        "max_latency_s": round(max(latencies), 2) if latencies else None,
        "details": details,
    }
