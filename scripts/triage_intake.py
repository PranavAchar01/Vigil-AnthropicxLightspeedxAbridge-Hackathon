"""Initial ESI triage from a spoken intake (Gap B) — end to end from the terminal.

Speak (or supply audio / text) a patient's reason-for-visit; Vigil transcribes it,
runs the full ESI v4 decision tree, and prints the assigned initial ESI with the
decision point, predicted resources, and rationale. Optionally ground it in a demo
patient's chart.

Run:
  uv run python scripts/triage_intake.py --mic --seconds 20
  uv run python scripts/triage_intake.py --audio /path/to/intake.wav
  uv run python scripts/triage_intake.py --text "chest pain radiating to my left arm"
  uv run python scripts/triage_intake.py --text "..." --patient <patient_id>
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vigil.chart import PatientChart, load_cohort  # noqa: E402
from vigil.config import settings  # noqa: E402
from vigil.reasoning import initial_triage, voice_intake  # noqa: E402


def _load_chart(patient_id: str) -> PatientChart | None:
    if not patient_id:
        return None
    if not settings.cohort_path.exists():
        print(f"  ! cohort not found at {settings.cohort_path}; grading without a chart")
        return None
    for c in load_cohort(settings.cohort_path):
        if c.patient_id == patient_id:
            return c
    print(f"  ! no cohort patient '{patient_id}'; grading without a chart")
    return None


def _get_transcript(args: argparse.Namespace) -> str:
    if args.text:
        return args.text
    if args.audio:
        return voice_intake.transcribe(args.audio)
    if args.mic:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            wav = voice_intake.record_wav(tf.name, seconds=args.seconds)
        print(f"  recorded → {wav}")
        return voice_intake.transcribe(wav)
    raise SystemExit("provide one of --text, --audio PATH, or --mic")


def main() -> None:
    ap = argparse.ArgumentParser(description="Initial ESI triage from a spoken intake.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--text", help="intake text (skip speech-to-text)")
    src.add_argument("--audio", help="path to an intake audio file")
    src.add_argument("--mic", action="store_true", help="record from the default microphone")
    ap.add_argument("--seconds", type=float, default=20.0, help="mic recording length")
    ap.add_argument("--patient", default="", help="cohort patient_id to ground the grade")
    args = ap.parse_args()

    transcript = _get_transcript(args)
    print(f"\n  INTAKE: “{transcript}”\n")

    chart = _load_chart(args.patient)
    d = initial_triage.grade(transcript, chart)

    bar = "─" * 56
    print(bar)
    print(f"  INITIAL ESI  {d.esi}   (decision point {d.esi_decision_point})")
    print(f"  criterion    {d.esi_criteria}")
    print(f"  complaint    {d.chief_complaint}")
    if d.predicted_resources:
        print(f"  resources    {', '.join(d.predicted_resources)}  (Decision C count)")
    if d.danger_zone_vitals:
        print("  vitals       DANGER ZONE (Decision D up-triage)")
    if d.red_flags:
        print(f"  red flags    {', '.join(d.red_flags)}")
    print(f"  confidence   {d.confidence:.2f}")
    print(f"  rationale    {d.rationale}")
    print(f"  spoken       “{d.spoken_summary}”")
    print(bar)
    print("  ⚠ decision support — a clinician confirms this grade." if d.needs_confirmation else "")


if __name__ == "__main__":
    main()
