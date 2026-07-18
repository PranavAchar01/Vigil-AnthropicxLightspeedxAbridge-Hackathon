# Vigil — always-on, multimodal re-triage for the waiting room

> A patient is usually triaged once, then may spend hours in a waiting room while
> their condition continues to change. Vigil continuously watches each patient,
> connects new visual and vocal signals to their clinical context, and brings a
> clinician back into the loop when their urgency may have increased.

**Vigil is an identity-aware patient monitoring and continuous re-triage agent.**
It begins with a voice or video intake, creates an initial patient risk profile,
links the arriving patient to that profile, and then uses cameras and microphones
to monitor changes over time. When Vigil finds evidence of deterioration, it
re-assesses the patient's Emergency Severity Index (ESI), checks in with the
patient when appropriate, calls the charge nurse, and documents the incident as
a clinical note and FHIR record.

Vigil is designed as a conservative clinical decision-support layer. It does not
diagnose patients or replace nurses. It watches the gap between initial triage and
clinical care.

Built at the **Abridge hackathon — The Future of Agentic AI in Healthcare.**

---

## The problem

Triage is a snapshot. A waiting room is a timeline.

A patient's initial ESI may be appropriate when they arrive, but their condition
can deteriorate while they wait. Staff cannot continuously observe every patient,
and a conventional security camera has no understanding of the patient's medical
history, original complaint, or changing symptoms.

Vigil combines those missing pieces:

- **Identity:** Which monitored patient is this?
- **Clinical context:** Why are they here, and what makes them high risk?
- **Continuous perception:** How are their movement, posture, voice, and behavior
  changing over time?
- **Re-triage:** Does the accumulated evidence justify a more urgent ESI?
- **Action:** Should Vigil keep watching, speak to the patient, or immediately
  bring in a nurse?
- **Documentation:** Can the escalation document itself in a clinical format?

---

## Product vision

```text
PRE-ARRIVAL / CHECK-IN
Voice or video intake
        ↓
Symptoms + history + initial predicted ESI
        ↓
Patient monitoring profile
        ↓
Identity enrollment at arrival

CONTINUOUS WAITING-ROOM MONITORING
Face-to-patient association + persistent body tracking
        ↓
Video + audio + behavior over time
        ↓
Patient-specific deterioration assessment
        ↓
Hold, voice check-in, or immediate nurse escalation

CLOSE THE LOOP
Chart-grounded spoken summary to the nurse
        ↓
Ambient SOAP incident note
        ↓
FHIR transaction bundle
```

### 1. Intake and initial ESI

A voice or video agent collects the patient's name, chief complaint, symptoms,
history, medications, and relevant warning signs. That conversation produces:

- a patient ID and monitoring profile;
- a transcript and concise clinical summary;
- an initial predicted ESI;
- patient-specific risk factors to watch during the wait.

For the hackathon, all clinical records are synthetic. A video intake or a
consented check-in kiosk can also enroll the patient's identity for the demo.

### 2. Link the physical patient to the chart

At arrival, Vigil associates the person in the waiting room with their patient
profile. The intended identity pipeline is:

```text
Consented face enrollment → patient ID → clinical chart → initial ESI
                                ↓
                       live camera track ID
```

Face matching answers **who the patient is**. YOLO/ByteTrack answers **where the
patient is now**. Pose and audio models answer **what is changing**. The clinical
agent decides **whether that change affects urgency**.

The current demo performs on-device face recognition with InsightFace/ArcFace.
It stores 512-dimensional embeddings in a local gallery, periodically matches the
largest visible face, and automatically selects the corresponding synthetic
patient chart. Raw enrollment images are used only to build the local demo
gallery. Persistent multi-person face-to-track association remains planned.

### 3. Always-on multimodal monitoring

Vigil is not only a scream or fall detector. Those are high-visibility examples
inside a broader continuous monitoring loop. The intended signal catalog includes:

- falls, collapse, loss of balance, and seizure-like movement;
- progressive slumping, posture decay, or prolonged motionlessness;
- gait changes, agitation, pacing, confusion, or attempts to leave;
- chest clutching, guarding, repeated bending, or visible distress;
- screams, calls for help, groaning, coughing changes, or labored breathing;
- statements such as “I cannot breathe,” “my chest hurts,” or “I feel worse”;
- a meaningful deviation from that patient's own recent baseline.

The important unit is not a single video frame. Vigil maintains temporal evidence
and asks whether the patient's condition is changing across seconds and minutes.

No system can recognize literally every possible medical event. Vigil therefore
combines a defined set of safety signals with anomaly detection, confidence
thresholds, patient check-ins, and human escalation.

### 4. Patient-specific re-triage

The same visible behavior can mean different things for different patients. A
collapse in a patient with hypoxemia or cardiac disease is different from the same
motion in a healthy control. A fall in a patient taking an anticoagulant carries a
different bleeding risk.

The reasoning agent combines:

```text
initial ESI
+ intake conversation
+ active conditions and medications
+ latest charted vitals
+ current visual observations
+ current audio observations
+ change from the patient's recent baseline
+ previous check-in responses
```

It returns a structured decision containing the prior ESI, proposed new ESI,
supporting evidence, action, and a short spoken summary for the nurse.

### 5. Severity-aware escalation

- **Hard evidence:** Call or page the nurse immediately.
- **Ambiguous evidence:** Speak to the patient first; page a human if the answer
  is concerning or absent.
- **Reassuring evidence:** Continue monitoring and preserve the event history.
- **Monotonic safety:** Vigil may increase urgency but never decrease it. Because
  ESI 1 is most acute, the new ESI number may only stay the same or go down.

### 6. Ambient documentation

Every completed incident produces an Abridge-style SOAP note and a FHIR R4
transaction bundle. The bundle contains an Encounter, ESI Observation, detected
event Observation, SOAP DocumentReference, transcript DocumentReference, safety
Flag, and nurse Communication.

The escalation therefore creates its own structured audit trail.

---

## Current implementation

This repository implements the perceive, reason, act, document, and audit loop:

```text
Camera + microphone
        ↓
YOLO pose events + audio distress events
        ↓
Per-patient sliding-window event fusion
        ↓
Tier 0 rules, with optional Claude re-triage
        ↓
Code-enforced monotonic ESI safety rules
        ↓
ElevenLabs or Twilio nurse call
        ↓
SOAP note + local FHIR transaction bundle
        ↓
Ranked multi-patient command center and audit chain
```

Implemented today:

- webcam pose tracking with fall, slumping, and motionlessness detection;
- microphone scream/distress detection using YAMNet or a lightweight fallback;
- hard/soft event fusion with cooldown protection;
- independent fusion windows, baselines, and alert state for every monitored patient;
- consent-gated track binding with bound, stale, and unbound lifecycle states;
- chart extraction from the Abridge synthetic FHIR dataset;
- deterministic Tier 0 re-triage that works without external services;
- optional Claude-based structured Tier 1 re-triage;
- code-level monotonic ESI enforcement and fail-safe escalation;
- time-aware ranked re-triage queue across three simultaneous demo patients;
- six role scopes with server-side field redaction;
- nurse acknowledgement, confirmed-event, and false-alarm feedback controls;
- hash-chained access and decision audit blocks with one-pass verification;
- typed, expiring break-glass grants and permanent compliance events;
- rule-floored intake ESI estimation for high-risk complaints;
- patient voice check-in policy;
- ElevenLabs conversational calls with direct Twilio fallback;
- an ElevenLabs conversational agent that can securely request the active
  patient's live posture, motion, last event, and ESI status during a call;
- on-device face enrollment and face-to-chart selection using InsightFace;
- SOAP note generation and an eight-resource FHIR transaction bundle with Provenance;
- a local live dashboard plus a Next.js command center for patient context,
  perception, reasoning, escalation, and documentation;
- a judge overview plus three role-grouped dashboards for clinical response,
  waiting-room operations, and trust and audit;
- a deterministic stage replay that works even when the camera or edge tunnel is unavailable;
- optional Supabase event mirroring for remote observability;
- offline tests for safety policy, role redaction, audit integrity, identity binding,
  queue ranking, replay behavior, and FHIR bundle shape.

Still planned beyond the hackathon demo:

- voice/video intake and initial ESI estimation;
- longer temporal baselines and a broader visual/audio event catalog;
- live ASR for natural-language distress understanding;
- production JWT identity provider integration;
- multi-camera handoff and directional audio localization;
- optional submission of the generated bundle to a real FHIR endpoint.

The stage replay runs three independent patient monitors. Live camera events are
routed through persistent track bindings and the same monitoring registry. The
demo role picker uses a role header to make field-level redaction inspectable;
production deployment still requires a hospital identity provider.

---

## Architecture

```text
vigil/
  config.py            # environment-driven models, credentials, and thresholds
  events.py            # typed objects passed through the complete pipeline
  chart.py             # synthetic FHIR data → compact PatientChart
  monitoring.py        # multi-patient state, queue ranking, binding, alerts
  security.py          # role scopes, redaction, break-glass, hash audit
  intake.py            # deterministic initial ESI safety floors
  demo.py              # three-patient deterministic stage replay
  perception/
    vision.py          # YOLO pose + ByteTrack → fall/slump/motionless events
    audio.py           # microphone → scream/distress events
    fusion.py          # signals → severity-tagged FusedEvent
    faces.py           # local ArcFace embeddings → synthetic patient identity
  reasoning/
    prompts.py         # conservative re-triage policy + strict output schema
    triage.py          # Claude call + code-enforced ESI safety rules
    rules.py           # integration-free Tier 0 reasoning
  escalation/
    ladder.py          # hold, patient check-in, or immediate nurse call
    elevenlabs_call.py # ElevenLabs calls and check-in transcript evaluation
    twilio_call.py     # direct Twilio TTS fallback
  documentation/
    abridge_note.py    # SOAP note + FHIR R4 transaction Bundle
  server/
    app.py             # FastAPI orchestration and application endpoints
    command_api.py     # queue, replay, role, alert, intake, and audit API
    bus.py             # WebSocket event fan-out and isolated video frame buffer
    status.py          # live patient state exposed to the voice agent
    supabase_sink.py   # optional remote event mirror
  dashboard/
    index.html         # live patient, reasoning, call, and note interface
scripts/
  extract_demo_cohort.py # source FHIR dataset → small synthetic demo cohort
  enroll_faces.py        # consenting demo photos → local embedding gallery
  setup_elevenlabs.py    # create voice agent and import a Twilio number
supabase/
  schema.sql              # remote observability event table
web/                      # Next.js/Vercel command center
tests/
  test_core.py           # fusion, escalation, ESI, vision, and FHIR safety tests
  test_product_layer.py  # multi-patient, redaction, audit, intake, and replay tests
```

### Runtime event model

- `PerceptionEvent`: one visual or audio observation.
- `FusedEvent`: correlated signals that justify re-triage.
- `TriageDecision`: structured ESI decision and recommended action.
- `EscalationAction`: patient check-in, nurse call, or no action.
- `BusEvent`: real-time dashboard update.

### Server interfaces

- `GET /` — dashboard
- `GET /health` — configured capabilities and active patient
- `GET /video` — MJPEG camera stream
- `WS /events` — real-time JSON event stream
- `GET /patients` — loaded demo cohort
- `POST /active/{patient_id}` — select the active demo patient
- `GET /agent/patient-status` — token-protected live status for the voice agent
- `GET /agent/patient-status/{patient_id}` — token-protected status by patient
- `POST /webhooks/elevenlabs` — mirror completed call turns into the event log
- `GET /api/v1/session` — role-filtered command-center state
- `GET /api/v1/queue` — ranked, role-filtered re-triage queue
- `POST /api/v1/demo/reset` — reset the deterministic stage scenario
- `POST /api/v1/demo/advance` — advance one replay signal
- `POST /api/v1/demo/replay` — run the complete replay automatically
- `POST /api/v1/alerts/{id}/acknowledge` — close the nurse acknowledgement loop
- `POST /api/v1/alerts/{id}/feedback` — record confirmed or false alarm feedback
- `POST /api/v1/tracks/{id}/bind` — consent-aware manual track correction
- `POST /api/v1/patients/{id}/esi` — monotonic clinician urgency override
- `POST /api/v1/operations/medical-assist` — route a seat-level request into the clinical queue
- `POST /api/v1/intake` — apply deterministic ESI floors and monitoring watch lists
- `POST /api/v1/break-glass` — issue an audited 15-minute emergency grant
- `GET /api/v1/audit/verify` — verify the complete hash chain

Run the server with one Uvicorn worker because the event bus, camera frame buffer,
and patient state are currently in process.

---

## Abridge integration

Vigil extends Abridge's ambient documentation thesis across the complete incident:

- **Chart in:** Conditions, medications, demographics, and latest vitals from a
  synthetic FHIR R4 record ground the re-triage decision.
- **Ambient event in:** The waiting-room interaction becomes another clinical
  episode rather than disappearing when the patient reaches the front desk.
- **Note out:** Vigil generates a SOAP incident note and transcript.
- **FHIR out:** The note, ESI change, event, safety flag, and nurse communication
  are packaged as a transaction bundle linked to the patient.

The current implementation writes bundles to `data/incidents/`. They are
POST-ready but are not automatically submitted to an external FHIR server.

---

## Setup

### Requirements

- Python 3.12+
- `uv`
- webcam and microphone for live perception
- optional InsightFace dependencies for face-to-chart recognition
- Abridge synthetic FHIR dataset for the demo cohort
- Anthropic API key for re-triage reasoning
- ElevenLabs/Twilio configuration for real outbound calls
- optional Supabase project and Node.js/pnpm for the remote command center

### Install

```bash
uv sync
cp .env.example .env
```

Install local face recognition support when using the identity demo:

```bash
uv sync --extra faces
uv run python scripts/enroll_faces.py
```

Set `VIGIL_DATASET_PATH` in `.env` to the synthetic FHIR JSONL file, then build
the demo cohort:

```bash
uv run python scripts/extract_demo_cohort.py
```

Start Vigil:

```bash
uv run uvicorn vigil.server.app:app --port 8000
```

For a replay-only dashboard or cloud backend with no camera and microphone:

```bash
VIGIL_PERCEPTION_ENABLED=0 uv run uvicorn vigil.server.app:app --port 8000
```

Open <http://localhost:8000>.

Start the Next.js judge experience in a second terminal:

```bash
corepack pnpm --dir web dev
```

Open the four product routes:

- <http://localhost:3000/> — judge overview and system story
- <http://localhost:3000/clinical> — charge nurse, triage, and attending command center
- <http://localhost:3000/operations> — front desk and security coordination
- <http://localhost:3000/trust> — compliance, audit verification, and break-glass controls

Set `NEXT_PUBLIC_VIGIL_URL` in the frontend deployment to the persistent public
URL of the FastAPI service. Without it, the browser uses `http://localhost:8000`
and visibly identifies any unavailable backend as stage-safe preview mode.

### Capabilities and graceful degradation

| Layer | Requirement | Behavior when unavailable |
|---|---|---|
| Cohort | `data/demo_cohort.json` | Server starts without an active patient |
| Reasoning | `ANTHROPIC_API_KEY` | Incident stops before clinical re-triage |
| Vision | Webcam + OpenCV/Ultralytics | Vision thread is disabled |
| Audio | Microphone + `sounddevice` | Audio thread is disabled |
| Audio ML | `pip install '.[audio]'` | Uses energy/spectral heuristic fallback |
| Face identity | `uv sync --extra faces` + enrolled gallery | Uses the selected active patient |
| Nurse call | ElevenLabs or Twilio credentials | Action is recorded as failed/not configured |
| SOAP note | Anthropic key | Uses deterministic chart-grounded fallback |
| FHIR | No external credentials required | Bundle is written locally |
| Remote feed | Supabase credentials | Local dashboard and event bus continue working |

### Configure the nurse call

```bash
export ELEVENLABS_API_KEY=sk_...
uv run python scripts/setup_elevenlabs.py create-agent
uv run python scripts/setup_elevenlabs.py import-number \
  --phone +1... --sid AC... --token ...
```

Place the resulting IDs and the charge nurse's demo phone number in `.env`:

```text
ELEVENLABS_AGENT_ID=...
ELEVENLABS_PHONE_NUMBER_ID=...
NURSE_PHONE_NUMBER=+1...
```

Never commit `.env` or real patient information.

---

## Test

```bash
uv run pytest
```

The offline suite covers multimodal fusion, cooldown behavior, escalation policy,
the monotonic ESI clamp, fail-safe action correction, fall geometry, and FHIR
transaction bundle shape.

---

## Safety, privacy, and scope

- **Clinical decision support, not diagnosis:** Vigil recommends attention; a
  clinician makes the medical decision.
- **Monotonic re-triage:** Vigil can add urgency but cannot remove it.
- **Human escalation:** Ambiguous or serious evidence brings a person into the
  loop rather than autonomously treating the patient.
- **Synthetic demo data:** The hackathon cohort contains no real patient records.
- **Consent for identity:** Face enrollment must be explicit and limited to the
  monitoring episode.
- **Data minimization:** The target design uses face embeddings and derived pose
  events, with short retention and access controls, instead of treating raw video
  as a clinical record.
- **Honest perception limits:** Vigil detects supported signals and anomalies; it
  does not claim to recognize every possible emergency.
- **Prototype status:** This project is not a validated medical device and must
  not be used for real clinical care.

---

## Hackathon contribution

Original work in this repository includes:

- the fused visual/audio perception pipeline;
- the chart-grounded, monotonic re-triage agent;
- the severity-aware patient-check-in and nurse-escalation ladder;
- dynamic chart-grounded ElevenLabs/Twilio calling;
- ambient SOAP documentation and FHIR incident packaging;
- the live reasoning and escalation dashboard;
- the evolving identity-aware, always-on monitoring architecture described here.

Pre-existing technologies include Ultralytics YOLO, ByteTrack, OpenCV, YAMNet,
Anthropic and ElevenLabs APIs, Twilio, FastAPI, and the provided synthetic FHIR
dataset.

---

## Demo narrative

1. Reset the command center and show three independently monitored patients.
2. Advance once. Maria Vega's posture departs from baseline, so Vigil starts an
   accessible voice and text check-in without paging a nurse.
3. Advance again. A respiratory signal corroborates the visual change. Cardiac
   history and low charted oxygen saturation raise ESI 3 to ESI 2.
4. Acknowledge the alert, then label it confirmed or false alarm to show the
   evaluation feedback loop.
5. Switch to Front desk. The response contains name, seat, wait time, and flagged
   state, but no ESI, chart, evidence, or video fields.
6. Switch to Compliance. Patient names disappear while the full hash chain remains.
7. Continue the replay to show an ambiguous healthy-control signal and the
   companion alarm routed to a different patient.
8. Show the SOAP note, eight-resource FHIR bundle, source Provenance, and verified
   audit head.

**Vigil does not replace triage. It keeps triage from becoming stale.**
