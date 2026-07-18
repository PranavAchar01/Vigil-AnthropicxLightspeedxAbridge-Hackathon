# Vigil — a continuous re-triage agent for the waiting room

> Triage happens **once**. Then a patient sits in a waiting room for 2–6 hours,
> unwatched, while their condition evolves. People deteriorate — and die — in
> waiting rooms; ~2M patients/year in the US leave without being seen. The
> failure isn't bad medicine. It's that **nobody is watching the queue.**

**Vigil is the agent that watches.** A camera and microphone observe the waiting
room; when a patient screams and collapses, Vigil detects it, re-triages that
patient against their **real clinical chart**, and — within seconds — places a
**real phone call to the charge nurse** with a spoken, chart-grounded summary of
what happened. The incident documents itself as a clinical note.

Built at the **Abridge hackathon** — *The Future of Agentic AI in Healthcare.*

---

## The agent loop (perceive → reason → act)

```
  ┌─────────────┐     ┌──────────────┐     ┌───────────────┐     ┌──────────────┐
  │  PERCEIVE   │     │   REASON     │     │     ACT       │     │  DOCUMENT    │
  │ webcam+mic  │ ──▶ │  Claude      │ ──▶ │ ElevenLabs    │ ──▶ │ Abridge-style│
  │ YOLO pose + │     │ re-triage vs │     │ real call to  │     │ SOAP note →  │
  │ scream det. │     │ FHIR chart   │     │ charge nurse  │     │ FHIR         │
  │  (FUSED)    │     │ ESI re-score │     │ (voice)       │     │ DocumentRef  │
  └─────────────┘     └──────────────┘     └───────────────┘     └──────────────┘
        skeletons only          monotonic:              severity-aware       on Abridge's
        (privacy)          escalate, never lower           ladder          ambient-doc thesis
```

**Zero text-to-text anywhere in the demo path:** vision + audio in, structured
clinical reasoning, speech out.

### Severity-aware escalation ladder (the false-positive defense)
- **Hard signal** (scream **+** fall): page + call the nurse **immediately**.
- **Soft signal** (prolonged stillness, posture decay): Vigil does a **voice
  check-in with the patient first** ("can you tell me your pain level right
  now?"); it only pages a human on a bad-or-absent answer.
- **Vigil never de-escalates.** It can only *add* attention, never remove it.

---

## Abridge integration

Abridge's product is **ambient clinical documentation**. Vigil sits on that
thesis at both ends:

- **Chart in.** Each patient is grounded in a real FHIR R4 chart from Abridge's
  synthetic ambient dataset — active conditions, medications, and latest vitals.
  That chart is what turns a body-pose event into a *clinical* decision: a
  collapse for a patient with charted cardiac history and hypoxemia is a
  different emergency than the same collapse for a healthy 20-year-old.
- **Note out.** The escalation and Vigil's voice interaction are transcribed
  into an **Abridge-style SOAP incident note** and written back as a FHIR
  `DocumentReference` linked to the patient — the transcript→note pattern from
  the dataset. **The escalation documents itself.**

---

## What we built at this hackathon

> Per hackathon rules, this section marks our original contributions vs.
> pre-existing libraries.

**Built here (our work):**
- The **fused perception layer** — combining YOLO-pose fall detection with local
  scream/distress audio detection into a single high-confidence event.
- The **chart-grounded re-triage agent** — Claude reasoning over FHIR priors +
  live events to produce a monotonic ESI re-score and escalation decision.
- The **severity-aware escalation ladder** and the **ElevenLabs outbound
  nurse-call** wiring with dynamic, chart-grounded scripting.
- The **Abridge-style ambient documentation** — incident → SOAP note → FHIR.
- The **live split-screen demo dashboard**.

**Pre-existing / libraries (not our contribution):** Ultralytics YOLO (pose
model), OpenCV, the audio classifier weights, the Anthropic and ElevenLabs SDKs,
FastAPI, and Abridge's provided FHIR dataset.

---

## Architecture

```
vigil/
  config.py            # env-driven settings, model + threshold config
  events.py            # typed event model (perception → fusion → decision → action)
  chart.py             # load Abridge FHIR dataset → PatientChart (conditions, meds, vitals)
  perception/
    vision.py          # YOLO-pose webcam → fall / slump / motionless events
    audio.py           # mic → scream / distress events
    fusion.py          # fuse vision + audio into a severity-tagged event
  reasoning/
    triage.py          # Claude chart-grounded re-triage (structured decision)
    prompts.py         # monotonic re-triage system prompt
  escalation/
    ladder.py          # severity-aware escalation policy
    elevenlabs_call.py # place the real outbound call to the nurse
  documentation/
    abridge_note.py    # incident → SOAP note → FHIR DocumentReference
  server/
    app.py             # FastAPI + WebSocket event bus, serves the dashboard
  dashboard/           # live split-screen UI
scripts/
  extract_demo_cohort.py  # slim the FHIR dataset into a demo cohort
data/
  demo_cohort.json     # the patients used in the live demo
```

---

## Run

```bash
uv sync
uv run python scripts/extract_demo_cohort.py      # FHIR dataset -> data/demo_cohort.json
uv run uvicorn vigil.server.app:app --port 8000   # ONE worker
# open http://localhost:8000
```

**It runs with zero credentials.** Every external dependency degrades gracefully,
so the full loop demos on a bare laptop and upgrades as you add keys:

| Layer | No creds / no hardware | With creds |
|---|---|---|
| Reasoning | chart-aware **heuristic** re-triage | Claude (`ANTHROPIC_API_KEY`) — opus→sonnet→haiku fallback |
| Nurse call | **simulated** call (dashboard banner + log) | real ElevenLabs outbound call (`ELEVENLABS_*`, `NURSE_PHONE_NUMBER`) |
| Vision | disabled (skeleton pane blank) | live YOLO-pose fall detection (webcam) |
| Audio | heuristic RMS+pitch fallback | YAMNet scream detection (`pip install '.[audio]'`) |
| FHIR | bundle written to `data/incidents/` | same bundle, POST-ready |

**Trigger an incident without staging a fall** (dev / demo backup):

```bash
curl -X POST 'http://localhost:8000/simulate/incident?severity=hard'   # scream+fall
curl -X POST 'http://localhost:8000/simulate/incident?severity=soft'   # motionless
curl 'http://localhost:8000/patients'                                  # list cohort
curl -X POST 'http://localhost:8000/active/<patient_id>'               # switch patient
```

**Enable the real nurse call** (before the demo):

```bash
cp .env.example .env    # fill ANTHROPIC_API_KEY, NURSE_PHONE_NUMBER
export ELEVENLABS_API_KEY=sk_...
uv run python scripts/setup_elevenlabs.py create-agent      # -> ELEVENLABS_AGENT_ID
uv run python scripts/setup_elevenlabs.py import-number \
    --phone +1... --sid AC... --token ...                   # -> ELEVENLABS_PHONE_NUMBER_ID
# put both ids in .env, then restart the server
```

## Test

```bash
uv run pytest        # safety logic: monotonic clamp, fusion, ladder, FHIR bundle shape
```

## Safety & privacy posture
- **Skeletons only** — no faces stored; waiting rooms already run CCTV.
- **Talks to the patient before paging a human** on ambiguous signals.
- **Monotonic** — adds attention, never removes it.
- **ESI is a real 5-level standard** — Vigil re-scores within it, it does not
  invent acuity.

*Everything in the dataset is synthetic (Synthea + LLM-generated). No real
patient data is used.*
