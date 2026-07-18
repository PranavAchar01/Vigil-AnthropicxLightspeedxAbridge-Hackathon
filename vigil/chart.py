"""Load Abridge's synthetic FHIR dataset into the clinical priors Vigil reasons over.

The chart is what turns a body-pose event into a *clinical* decision: the same
collapse means something different for a patient with charted cardiac history and
hypoxemia than for a healthy 20-year-old. We extract exactly the acuity-relevant
slice: demographics, active conditions, active medications, and latest vitals.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

# LOINC codes for the vitals/labs that matter to acuity, mapped to short labels.
VITAL_LOINC: dict[str, str] = {
    "8310-5": "temp_c",
    "8867-4": "heart_rate",
    "9279-1": "resp_rate",
    "2708-6": "spo2",
    "59408-5": "spo2",
    "8480-6": "systolic_bp",
    "8462-4": "diastolic_bp",
    "8302-2": "height_cm",
    "29463-7": "weight_kg",
    "6690-2": "wbc",
    "718-7": "hemoglobin",
    "2339-0": "glucose",
}


@dataclass
class Vital:
    label: str
    value: float
    unit: str
    at: str  # ISO timestamp


@dataclass
class PatientChart:
    patient_id: str
    encounter_id: str
    name: str
    gender: str
    age: int | None
    visit_title: str
    active_conditions: list[str] = field(default_factory=list)
    active_medications: list[str] = field(default_factory=list)
    latest_vitals: dict[str, Vital] = field(default_factory=dict)
    baseline_esi: int = 3  # demo baseline; the re-triage only ever lowers this number

    def to_context(self) -> str:
        """Compact clinical summary injected into the re-triage prompt."""
        vitals = (
            ", ".join(f"{v.label} {v.value:g}{v.unit}" for v in self.latest_vitals.values())
            or "none on file"
        )
        conds = "; ".join(self.active_conditions[:12]) or "none listed"
        meds = "; ".join(self.active_medications[:12]) or "none listed"
        age = f"{self.age}yo" if self.age is not None else "age unknown"
        return (
            f"PATIENT: {self.name} ({age} {self.gender})\n"
            f"VISIT: {self.visit_title}\n"
            f"BASELINE ESI: {self.baseline_esi}\n"
            f"ACTIVE CONDITIONS: {conds}\n"
            f"ACTIVE MEDICATIONS: {meds}\n"
            f"LATEST VITALS: {vitals}"
        )


def _official_name(name_field: list[dict]) -> str:
    if not name_field:
        return "Unknown"
    entry = next((n for n in name_field if n.get("use") == "official"), name_field[0])
    given = " ".join(entry.get("given", []))
    return f"{given} {entry.get('family', '')}".strip() or "Unknown"


def _age(birth_date: str | None, ref: str | None) -> int | None:
    if not birth_date:
        return None
    try:
        b = date.fromisoformat(birth_date[:10])
        r = date.fromisoformat(ref[:10]) if ref else date.today()
        return r.year - b.year - ((r.month, r.day) < (b.month, b.day))
    except ValueError:
        return None


def _latest_vitals(observations: list[dict]) -> dict[str, Vital]:
    """Keep the most recent valueQuantity per acuity-relevant LOINC code."""
    latest: dict[str, Vital] = {}
    for obs in observations:
        vq = obs.get("valueQuantity")
        if not vq:
            continue
        for coding in obs.get("code", {}).get("coding", []):
            label = VITAL_LOINC.get(coding.get("code"))
            if not label:
                continue
            when = obs.get("effectiveDateTime", "")
            existing = latest.get(label)
            if existing is None or when > existing.at:
                latest[label] = Vital(
                    label=label,
                    value=round(float(vq.get("value", 0.0)), 2),
                    unit=vq.get("unit", ""),
                    at=when,
                )
    return latest


def build_chart(record: dict, baseline_esi: int = 3) -> PatientChart:
    md = record.get("metadata", {})
    pc = record.get("patient_context", {})
    patient = pc.get("patient", {})
    ls = pc.get("longitudinal_summary", {})
    related = record.get("encounter_fhir", {}).get("related_resources", {})

    return PatientChart(
        patient_id=md.get("patient_id", ""),
        encounter_id=md.get("encounter_id", ""),
        name=_official_name(patient.get("name", [])),
        gender=patient.get("gender", "unknown"),
        age=_age(patient.get("birthDate"), md.get("date")),
        visit_title=md.get("visit_title", ""),
        active_conditions=list(ls.get("condition_labels", [])),
        active_medications=list(ls.get("medication_labels", [])),
        latest_vitals=_latest_vitals(related.get("Observation", [])),
        baseline_esi=baseline_esi,
    )


def load_records(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def chart_from_dict(d: dict) -> PatientChart:
    """Rebuild a PatientChart from a serialized cohort row (data/demo_cohort.json)."""
    vitals = {
        k: Vital(label=v["label"], value=v["value"], unit=v["unit"], at=v.get("at", ""))
        for k, v in (d.get("latest_vitals") or {}).items()
    }
    return PatientChart(
        patient_id=d["patient_id"],
        encounter_id=d.get("encounter_id", ""),
        name=d["name"],
        gender=d.get("gender", "unknown"),
        age=d.get("age"),
        visit_title=d.get("visit_title", ""),
        active_conditions=list(d.get("active_conditions", [])),
        active_medications=list(d.get("active_medications", [])),
        latest_vitals=vitals,
        baseline_esi=int(d.get("baseline_esi", 3)),
    )


def load_cohort(path: Path) -> list[PatientChart]:
    charts = [chart_from_dict(row) for row in json.loads(Path(path).read_text())]
    return charts


if __name__ == "__main__":
    from vigil.config import settings

    recs = load_records(settings.dataset_path)
    print(f"Loaded {len(recs)} records from {settings.dataset_path}")
    chart = build_chart(recs[1])  # the COVID isolation / hypoxemia case
    print(chart.to_context())
