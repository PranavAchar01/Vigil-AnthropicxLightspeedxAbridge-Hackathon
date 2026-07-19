# Vigil Intake — standalone voice triage

Speak a patient's reason for visit; get an **initial ESI level (1–5)** graded against
the **Emergency Severity Index (ESI) v4** rubric (AHRQ / Emergency Nurses Association).
Completely self-contained — no camera, no database, no other Vigil code.

- **Voice → text**: ElevenLabs Scribe (speech-to-text)
- **Text → ESI**: Claude, forced through the ESI v4 four-decision-point algorithm
- Optional **chart grounding** from `sample_cohort.json` (edit or replace it)

## Run locally

```bash
cd intake-app
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env        # then add your ANTHROPIC_API_KEY and ELEVENLABS_API_KEY

uvicorn app:app --port 8000
# open http://localhost:8000
```

Click **Record intake**, allow the mic, and speak (e.g. *"crushing chest pain going
down my left arm"*). It transcribes what it heard and shows the ESI, the decision
point (A/B/C/D), the criterion, and the predicted resources. You can also just type
symptoms and click **Grade**. Pick a patient from the dropdown to ground the grade in
their chart.

## What you need

- Python 3.10+
- An **Anthropic API key** (grading) and an **ElevenLabs API key** (speech-to-text),
  both in `.env`

## How the grade works

The grade is decision **support** — a clinician confirms it. Two ESI-algorithm rules
are enforced in code, not left to the model: Decision **A** (immediate life-saving
intervention) always yields **ESI 1**, and danger-zone vitals up-triage a would-be
ESI 3 to **ESI 2**. If the model is unreachable it fails safe to **ESI 2** and flags
for manual triage — it never silently assigns a low acuity.

## Files

| File | Purpose |
|---|---|
| `app.py` | FastAPI server: serves the page, `/patients`, `POST /intake` |
| `triage.py` | ESI v4 prompt + grading + code-enforced safety invariants |
| `voice.py` | ElevenLabs speech-to-text |
| `index.html` | The record-intake UI |
| `sample_cohort.json` | Demo patient charts for optional grounding |
