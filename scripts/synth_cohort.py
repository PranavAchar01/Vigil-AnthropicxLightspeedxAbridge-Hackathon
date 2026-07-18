"""Write a SYNTHETIC demo cohort whose patient_ids MATCH the enrolled face gallery
(data/face_gallery.json), so face recognition -> chart binding works end to end:
the camera recognizes a face, resolves the patient_id, and this cohort supplies the
chart. Includes the 5 demo patients (mapped to their enrolled images/conditions)
plus the two real team members (Sahiel, Pranav) as consenting demo participants.

All clinical data is invented (no real records). Use extract_demo_cohort.py when
you have the real synthetic-ambient-fhir dataset instead.

Run:  uv run python scripts/synth_cohort.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vigil.config import settings  # noqa: E402


def _v(label: str, value: float, unit: str) -> dict:
    return {"label": label, "value": value, "unit": unit, "at": "2026-07-18T08:00:00Z"}


# patient_id values MUST match data/face_gallery.json exactly.
COHORT = [
    {
        "patient_id": "sahiel-bose",
        "name": "Sahiel Bose",
        "gender": "male",
        "age": 20,
        "visit_title": "COVID-19 isolation, worsening dyspnea",
        "active_conditions": ["Viral pneumonia (disorder)", "Asthma", "Hypoxemia"],
        "active_medications": ["albuterol inhaler", "apixaban 5 MG Oral Tablet"],
        "latest_vitals": {
            "spo2": _v("spo2", 88.0, "%"),
            "heart_rate": _v("heart_rate", 118.0, "/min"),
            "resp_rate": _v("resp_rate", 26.0, "/min"),
        },
        "baseline_esi": 3,
        "demo_reason": "hero deterioration case: pneumonia + hypoxemia",
    },
    {
        "patient_id": "pranav-achar",
        "name": "Pranav Achar",
        "gender": "male",
        "age": 20,
        "visit_title": "new-onset chest pain, cardiac workup",
        "active_conditions": ["Chest pain (finding)", "Hyperlipidemia", "Family history of MI"],
        "active_medications": ["aspirin 81 MG Oral Tablet", "atorvastatin 20 MG Oral Tablet"],
        "latest_vitals": {
            "heart_rate": _v("heart_rate", 104.0, "/min"),
            "systolic_bp": _v("systolic_bp", 158.0, "mmHg"),
            "diastolic_bp": _v("diastolic_bp", 96.0, "mmHg"),
        },
        "baseline_esi": 3,
        "demo_reason": "possible ACS — chest-clutch/collapse plausibility",
    },
    {
        "patient_id": "7bd9e5b0-5d4b-f10d-9579-f4813faf9cdc",
        "name": "Ariane Jan Runolfsson",
        "gender": "female",
        "age": 68,
        "visit_title": "COVID-19 isolation",
        "active_conditions": ["Pneumonia (disorder)", "COPD", "Coronary artery disease"],
        "active_medications": ["atenolol 50 MG Oral Tablet"],
        "latest_vitals": {"spo2": _v("spo2", 90.0, "%"), "heart_rate": _v("heart_rate", 112.0, "/min")},
        "baseline_esi": 3,
        "demo_reason": "covid.jpg — respiratory decompensation",
    },
    {
        "patient_id": "374e68b2-ee15-0852-cd48-3c7b6fd8e114",
        "name": "Dick Larson",
        "gender": "male",
        "age": 59,
        "visit_title": "post-sepsis recovery, febrile",
        "active_conditions": ["Sepsis (disorder)", "Type 2 diabetes mellitus"],
        "active_medications": ["insulin glargine", "metoprolol 25 MG Oral Tablet"],
        "latest_vitals": {
            "temp_c": _v("temp_c", 38.6, "Cel"),
            "heart_rate": _v("heart_rate", 108.0, "/min"),
            "systolic_bp": _v("systolic_bp", 98.0, "mmHg"),
        },
        "baseline_esi": 3,
        "demo_reason": "sepsis.jpg — sepsis relapse",
    },
    {
        "patient_id": "4b4735a2-ee12-ec86-041f-3ba4d5c81ec9",
        "name": "Elias Wisozk",
        "gender": "male",
        "age": 51,
        "visit_title": "new hypertension and chest tightness",
        "active_conditions": ["Essential hypertension", "Hyperlipidemia"],
        "active_medications": ["lisinopril 10 MG Oral Tablet", "atorvastatin 20 MG Oral Tablet"],
        "latest_vitals": {
            "systolic_bp": _v("systolic_bp", 170.0, "mmHg"),
            "diastolic_bp": _v("diastolic_bp", 100.0, "mmHg"),
            "heart_rate": _v("heart_rate", 90.0, "/min"),
        },
        "baseline_esi": 3,
        "demo_reason": "hypertension.jpg — cardiac plausibility",
    },
    {
        "patient_id": "256bea3d-7833-e7f8-74b4-0e4f7e299c73",
        "name": "Lala Kazuko Casper",
        "gender": "female",
        "age": 34,
        "visit_title": "behavioral health safety disclosure",
        "active_conditions": ["Anxiety disorder", "Depressive disorder"],
        "active_medications": ["sertraline 50 MG Oral Tablet"],
        "latest_vitals": {"heart_rate": _v("heart_rate", 96.0, "/min")},
        "baseline_esi": 3,
        "demo_reason": "safety.jpg — behavioral/agitation escalation",
    },
    {
        "patient_id": "966e9818-9cf6-8d71-74bc-36aef6643618",
        "name": "Delana Jamie Gutkowski",
        "gender": "female",
        "age": 23,
        "visit_title": "young adult preventive visit",
        "active_conditions": [],
        "active_medications": [],
        "latest_vitals": {"spo2": _v("spo2", 99.0, "%"), "heart_rate": _v("heart_rate", 66.0, "/min")},
        "baseline_esi": 4,
        "demo_reason": "young.jpg — healthy control, Vigil should HOLD",
    },
]


def main() -> None:
    out = settings.cohort_path
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(COHORT, indent=2), encoding="utf-8")
    for c in COHORT:
        print(f"  + {c['name']:24s} {c['patient_id']:38s} ESI {c['baseline_esi']}")
    print(f"\nWrote {len(COHORT)} patients (gallery-matched) -> {out}")


if __name__ == "__main__":
    main()
