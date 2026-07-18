"""Slim Abridge's 25-record FHIR dataset into the cohort used in the live demo.

We pick a handful of patients whose charts make waiting-room deterioration
clinically plausible, plus one healthy control that proves Vigil holds instead of
crying wolf. Each gets a demo baseline ESI; the re-triage only ever lowers it.

Run:  uv run python scripts/extract_demo_cohort.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vigil.chart import build_chart, load_records  # noqa: E402
from vigil.config import settings  # noqa: E402

# (substring to match in visit_title, demo baseline ESI, one-line "why this patient")
DEMO_COHORT: list[tuple[str, int, str]] = [
    ("COVID-19 isolation", 3, "hero case: pneumonia + hypoxemia, visible decompensation"),
    ("post-sepsis recovery", 3, "sepsis relapse — a classic waiting-room death"),
    ("new hypertension and metabolic", 3, "cardiac plausibility for a chest-clutch collapse"),
    ("safety disclosure", 3, "behavioral/agitation escalation path"),
    ("Young adult preventive", 4, "healthy control — Vigil must HOLD, not escalate"),
]


def main() -> None:
    records = load_records(settings.dataset_path)
    by_title = {r.get("metadata", {}).get("visit_title", ""): r for r in records}

    cohort = []
    for needle, baseline, why in DEMO_COHORT:
        match = next((r for t, r in by_title.items() if needle.lower() in t.lower()), None)
        if match is None:
            print(f"  ! no record matched '{needle}' — skipping")
            continue
        chart = build_chart(match, baseline_esi=baseline)
        row = asdict(chart)
        row["demo_reason"] = why
        row["chief_complaint"] = chart.visit_title
        # the ambient intake conversation, trimmed — establishes why they're here
        row["intake_excerpt"] = (match.get("transcript", "") or "")[:900]
        cohort.append(row)
        print(f"  + {chart.name:28s} ESI {baseline}  — {why}")

    out = settings.cohort_path
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cohort, indent=2), encoding="utf-8")
    print(f"\nWrote {len(cohort)} patients -> {out}")


if __name__ == "__main__":
    main()
