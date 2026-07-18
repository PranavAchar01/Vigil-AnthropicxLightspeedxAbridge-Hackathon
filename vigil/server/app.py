"""Vigil server — one process, one uvicorn worker.

Wires the whole loop: perception threads publish PerceptionEvents -> fusion on the
event loop -> Claude re-triage -> severity-aware escalation (real nurse call) ->
Abridge SOAP note + FHIR bundle. Two transports leave the process: an MJPEG stream
(/video, skeleton frames) and a WebSocket (/events, JSON only).

Every action runs on real, sourced data: the FHIR chart, the live camera/mic, and
the real Claude + ElevenLabs APIs. There is no mock/simulation path — supply the
keys and it functions end to end. `/health` reports what is configured.

Run:  uv run uvicorn vigil.server.app:app --port 8000     (ONE worker)
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import (
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse

from vigil.chart import PatientChart, load_cohort
from vigil.config import settings
from vigil.documentation import abridge_note
from vigil.escalation.elevenlabs_call import (
    NurseCallHandler,
    checkin_configured,
    nurse_call_configured,
)
from vigil.escalation.ladder import run_ladder
from vigil.escalation.twilio_call import twilio_call_configured
from vigil.events import BusEvent, FusedEvent, PerceptionEvent
from vigil.perception.fusion import EventFuser
from vigil.reasoning import initial_triage, triage, voice_intake
from vigil.reasoning.triage import ReasoningNotConfigured
from vigil.server import status as pstatus
from vigil.server import supabase_sink as supa
from vigil.server.bus import EventBus, FrameBuffer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
log = logging.getLogger("vigil.server")

DASHBOARD_DIR = Path(__file__).resolve().parent.parent / "dashboard"
DASHBOARD = DASHBOARD_DIR / "index.html"
FONTS = DASHBOARD_DIR / "fonts.css"

bus = EventBus()
frames = FrameBuffer()
fuser = EventFuser(
    window_s=settings.fusion_window_seconds, cooldown_s=settings.fusion_cooldown_seconds
)


class State:
    charts: dict[str, PatientChart] = {}
    active_id: str = ""
    stop = threading.Event()
    pause = threading.Event()  # demo pause: freeze detection while the camera streams
    # One-call-per-issue: an incident keeps the SAME issue_id while its signals keep
    # re-firing; after nurse_issue_gap_s of quiet, the next hard event is a NEW issue.
    issue_id: int = 0
    last_hard_ts: float = 0.0


state = State()


def _load_cohort() -> None:
    if not settings.cohort_path.exists():
        log.warning(
            "cohort not found at %s — run scripts/extract_demo_cohort.py", settings.cohort_path
        )
        return
    charts = load_cohort(settings.cohort_path)
    state.charts = {c.patient_id: c for c in charts}
    # No patient is shown until the camera RECOGNIZES a face (face -> active -> chart).
    state.active_id = ""
    log.info("loaded %d patients; awaiting face recognition to bind a chart", len(charts))


def _active() -> PatientChart | None:
    return state.charts.get(state.active_id)


def _fallback_patient_id() -> str:
    """When a distress signal fires before a face is recognized, bind it to a patient
    anyway (first in the cohort) so the incident still shows. Face recognition, when it
    lands, overrides this via state.active_id."""
    return next(iter(state.charts), "")


class _Gallery:
    gallery = None
    loaded = False


def _face_gallery():
    """Lazy-load the enrolled face gallery once (for avatar lookup); None if absent."""
    if not _Gallery.loaded:
        _Gallery.loaded = True
        if settings.face_gallery_path.exists():
            try:
                from vigil.perception.faces import FaceGallery

                _Gallery.gallery = FaceGallery.load(settings.face_gallery_path)
            except Exception:  # noqa: BLE001
                _Gallery.gallery = None
    return _Gallery.gallery


def _avatar_for(patient_id: str) -> str | None:
    g = _face_gallery()
    return f"/faces/{patient_id}.jpg" if (g and g.image_for(patient_id)) else None


def _capabilities() -> dict[str, bool]:
    return {
        "reasoning": bool(settings.anthropic_api_key),
        "nurse_call": nurse_call_configured() or twilio_call_configured(),
        "patient_checkin": checkin_configured(),
        "cohort_loaded": bool(state.charts),
        "backend": supa.configured(),
        "fall_model": settings.fall_model_path.exists(),
        "intake_triage": bool(settings.anthropic_api_key),
        "voice_intake": bool(settings.elevenlabs_api_key),
    }


# Human-readable one-liner per event type, for the Supabase live feed.
def _mirror(event: BusEvent) -> None:
    p = event.payload
    t = event.type
    chart = _active()
    patient = chart.name if chart else None
    if t == "perception":
        summary = f"{p.get('modality')}: {p.get('kind')} ({p.get('confidence')})"
        source = str(p.get("modality"))
    elif t == "fused":
        summary = f"⚑ {p.get('summary')} · {str(p.get('severity')).upper()} [{'+'.join(p.get('kinds', []))}]"
        source = "vision"
    elif t == "decision":
        summary = (
            f"ESI {p.get('prior_esi')}→{p.get('new_esi')} · "
            f"{str(p.get('action')).replace('_', ' ')}"
        )
        source = "claude"
    elif t == "call_status":
        summary = f"nurse call: {p.get('status')}"
        source = "elevenlabs"
    elif t == "escalation":
        summary = f"{str(p.get('kind')).replace('_', ' ')} → {p.get('status')}"
        source = "vigil"
    elif t == "note":
        summary = "ambient SOAP note + FHIR bundle written"
        source = "claude"
    elif t == "initial_triage":
        summary = (
            f"voice intake → ESI {p.get('esi')} · decision {p.get('esi_decision_point')} · "
            f"{p.get('chief_complaint')}"
        )
        source = "claude"
    else:
        return  # skip noise: reasoning_delta, status/capabilities, patient
    supa.log_event(t, source=source, patient=patient, summary=summary, payload=p)


# --------------------------------------------------------------------------- #
# Perception -> fusion -> incident pipeline (all on the event loop)
# --------------------------------------------------------------------------- #


def _patient_name(pid: str) -> str | None:
    chart = state.charts.get(pid)
    return chart.name if chart else None


def _on_perception(ev: PerceptionEvent) -> None:
    """Runs on the loop thread (scheduled via call_soon_threadsafe)."""
    # Provisional fall signals are INSTANT UI feedback only: show the "validating…"
    # countdown the moment a fall is seen, but never fuse/reason/page on them. The nurse
    # is only paged once the fall validates into `fainted` (below).
    if ev.kind in ("fall_detected", "fall_cleared"):
        pid = state.active_id or _fallback_patient_id()
        bus.publish(
            BusEvent(
                type="fall_validation",
                payload={
                    "state": "detected" if ev.kind == "fall_detected" else "cleared",
                    "validate_s": float(ev.meta.get("validate_s", 0.0)),
                    "patient": _patient_name(pid),
                    "ts": ev.ts,
                },
            )
        )
        return

    bus.publish(BusEvent(type="perception", payload=ev.model_dump()))
    if ev.kind in ("fainted", "seizure", "scream"):
        pid = state.active_id or _fallback_patient_id()
        if pid:
            pstatus.mark_event(pid, ev.kind)
        if ev.kind == "fainted":  # the validation completed → flip the banner to confirmed
            bus.publish(
                BusEvent(
                    type="fall_validation",
                    payload={"state": "confirmed", "patient": _patient_name(pid), "ts": ev.ts},
                )
            )
    fused = fuser.add(ev)
    if fused is not None:
        asyncio.create_task(handle_incident(fused))


async def _stream_text(kind: str, text: str, delay: float = 0.025) -> None:
    for word in text.split(" "):
        bus.publish(BusEvent(type=kind, payload={"text": word + " "}))
        await asyncio.sleep(delay)


async def handle_incident(fused: FusedEvent) -> None:
    chart = _active()
    if chart is None:
        # A distress signal fired before a face was recognized. Bind it to a patient
        # anyway so the incident shows and the nurse still gets paged (fail toward care).
        pid = _fallback_patient_id()
        chart = state.charts.get(pid)
        if chart is None:
            log.warning("incident but no patients loaded; ignoring")
            return
        state.active_id = pid
        bus.publish(BusEvent(type="patient", payload=_patient_payload(chart)))
        log.info("incident with no recognized face — bound to %s (fallback)", chart.name)

    # One call per issue: keep the same issue_id while a single incident's signals keep
    # re-firing; only start a new issue (which may page again) after a quiet gap.
    now = time.time()
    if now - state.last_hard_ts > settings.nurse_issue_gap_s:
        state.issue_id += 1
    state.last_hard_ts = now

    bus.publish(BusEvent(type="fused", payload=fused.model_dump()))
    bus.publish(
        BusEvent(
            type="reasoning_start",
            payload={
                "patient": chart.name,
                "summary": fused.summary,
                "prior_esi": chart.baseline_esi,
            },
        )
    )

    try:
        decision = await asyncio.to_thread(triage.decide, chart, fused)
    except ReasoningNotConfigured as e:
        log.error("%s", e)
        bus.publish(
            BusEvent(
                type="status",
                payload={
                    "level": "error",
                    "message": "Set ANTHROPIC_API_KEY to enable re-triage reasoning.",
                },
            )
        )
        return

    await _stream_text("reasoning_delta", decision.rationale)
    bus.publish(BusEvent(type="decision", payload=decision.model_dump()))
    # record for the voice agent's get_patient_status tool (pre-computed; never in the call path)
    pstatus.update_retriage(
        chart.patient_id,
        decision.new_esi,
        decision.prior_esi,
        decision.rationale,
        decision.spoken_summary,
        chart.to_context(),
    )

    handler = NurseCallHandler(chart, bus=bus, issue_id=state.issue_id)
    actions = await asyncio.to_thread(run_ladder, decision, handler)
    for a in actions:
        bus.publish(BusEvent(type="escalation", payload=a.model_dump()))

    note, bundle_path = await asyncio.to_thread(
        abridge_note.write_incident, chart, fused, decision, actions
    )
    bus.publish(BusEvent(type="note", payload={"text": note, "bundle_path": bundle_path}))


# --------------------------------------------------------------------------- #
# Perception thread starters (lazy heavy imports)
# --------------------------------------------------------------------------- #


def _vision_status(track_id: int, posture: str, motion: str, moved: bool) -> None:
    # single-participant demo: bind whatever the camera sees to the active patient
    if state.active_id:
        pstatus.update_vision(state.active_id, posture, motion, moved)


def _vision_identify(pid: str, name: str, score: float) -> None:
    """A recognized face -> make that patient active and pull up their chart."""
    chart = state.charts.get(pid)
    if chart is None:
        return
    state.active_id = pid
    bus.publish_from_thread(BusEvent(type="patient", payload=_patient_payload(chart)))
    log.info("face recognized -> %s (%.3f)", chart.name, score)
    supa.log_event(
        "identify",
        source="vision",
        patient=chart.name,
        summary=f"face recognized → {chart.name} (match {score})",
        payload={"patient_id": pid, "score": score},
    )


def _start_vision(sink) -> None:
    try:
        from vigil.perception.vision import run_vision

        src = settings.video_source
        source = int(src) if src.lstrip("-").isdigit() else src  # camera index or file path
        run_vision(
            sink,
            frames,
            state.stop,
            source=source,
            status_sink=_vision_status,
            identify_sink=_vision_identify,
            pause_event=state.pause,
        )
    except Exception as e:  # noqa: BLE001
        log.warning("vision disabled (%r) — install ML deps and connect a camera", e)


def _start_audio(sink) -> None:
    import os

    if os.environ.get("VIGIL_DISABLE_AUDIO", "").lower() in ("1", "true", "yes"):
        log.info("audio detection PAUSED (VIGIL_DISABLE_AUDIO set)")
        return
    try:
        from vigil.perception.audio import run_audio

        run_audio(sink, state.stop, threshold=settings.scream_threshold)
    except Exception as e:  # noqa: BLE001
        log.warning("audio disabled (%r)", e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_running_loop()
    bus.bind_loop(loop)
    bus.set_tap(_mirror)  # mirror every event into the Supabase backend
    _load_cohort()
    # Publish our public URL so the Vercel page can discover this backend at load
    # time (set by scripts/serve_public.py once the tunnel is up).
    if settings.public_url:
        supa.set_backend_url(settings.public_url)

    def sink(ev: PerceptionEvent) -> None:  # called from perception threads
        loop.call_soon_threadsafe(_on_perception, ev)

    state.stop.clear()
    threading.Thread(target=_start_vision, args=(sink,), daemon=True).start()
    threading.Thread(target=_start_audio, args=(sink,), daemon=True).start()
    caps = _capabilities()
    log.info(
        "Vigil up on http://localhost:8000 — capabilities: %s",
        ", ".join(f"{k}={'on' if v else 'off'}" for k, v in caps.items()),
    )
    yield
    state.stop.set()


app = FastAPI(title="Vigil", lifespan=lifespan)

# Allow the Vercel command-center frontend (and any dashboard origin) to read the
# open endpoints cross-origin. Token-guarded routes stay protected by the header.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
async def index():
    if DASHBOARD.exists():
        return HTMLResponse(DASHBOARD.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Vigil</h1><p>dashboard/index.html missing</p>")


@app.get("/fonts.css")
async def fonts():
    if FONTS.exists():
        return FileResponse(FONTS, media_type="text/css")
    return JSONResponse({"error": "fonts.css missing"}, status_code=404)


@app.get("/intake", response_class=HTMLResponse)
async def intake_page():
    """Browser voice-intake tool: record a spoken intake -> POST /intake -> show the ESI."""
    page = DASHBOARD_DIR / "intake.html"
    if page.exists():
        return HTMLResponse(page.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Vigil</h1><p>dashboard/intake.html missing</p>")


@app.get("/health")
async def health():
    return JSONResponse({"capabilities": _capabilities(), "active_patient": state.active_id})


@app.post("/pause")
async def pause_toggle():
    """Demo pause: freeze detection (no events fire) while the camera keeps streaming."""
    if state.pause.is_set():
        state.pause.clear()
    else:
        state.pause.set()
    paused = state.pause.is_set()
    bus.publish(BusEvent(type="paused", payload={"paused": paused}))
    return {"paused": paused}


@app.get("/faces/{patient_id}.jpg")
def patient_avatar(patient_id: str):
    """Serve the enrolled face image for a patient (chart-card avatar)."""
    g = _face_gallery()
    img = g.image_for(patient_id) if g else None
    path = (settings.cohort_path.parent / "faces" / img) if img else None
    if path and path.exists():
        return FileResponse(path, media_type="image/jpeg")
    raise HTTPException(status_code=404, detail="no avatar")


@app.get("/agent/patient-status/{patient_id}")
def patient_status(patient_id: str, x_vigil_token: str = Header(default="")):
    """The voice agent's get_patient_status webhook tool hits this mid-call.

    Sync `def` on purpose: it runs in FastAPI's threadpool so the lock-guarded cache
    read never blocks the event loop (video/reasoning streams). No LLM, no I/O.
    """
    if settings.agent_token and x_vigil_token != settings.agent_token:
        raise HTTPException(status_code=401, detail="invalid or missing X-Vigil-Token")
    return JSONResponse(pstatus.snapshot(patient_id))


@app.get("/agent/patient-status")
def patient_status_active(x_vigil_token: str = Header(default="")):
    """Live status of the CURRENTLY active patient — the agent's tool hits this so it
    never has to guess a patient_id over the phone."""
    if settings.agent_token and x_vigil_token != settings.agent_token:
        raise HTTPException(status_code=401, detail="invalid or missing X-Vigil-Token")
    snap = pstatus.snapshot(state.active_id or "unknown")
    chart = _active()
    supa.log_event(
        "tool_call",
        source="agent",
        patient=chart.name if chart else None,
        summary=f"voice agent pulled live status → {snap.get('posture')}, "
        f"{snap.get('motion_level')}, ESI {snap.get('triage', {}).get('esi_level')}",
        payload=snap,
    )
    return JSONResponse(snap)


@app.post("/webhooks/elevenlabs")
async def elevenlabs_webhook(request: Request):
    """Post-call webhook: ElevenLabs sends the conversation transcript here when a
    call ends. We log each nurse question + agent answer so judges see the dialogue."""
    body = await request.json()
    data = body.get("data", body)
    transcript = data.get("transcript", []) or data.get("messages", []) or []
    chart = _active()
    patient = chart.name if chart else None
    for turn in transcript:
        role = turn.get("role", "")
        text = turn.get("message") or turn.get("text") or ""
        if not text:
            continue
        who = "nurse" if role in ("user", "human") else "agent"
        supa.log_event(
            "conversation_turn",
            source="elevenlabs",
            patient=patient,
            summary=f"{who}: {text[:120]}",
            payload={"role": who, "text": text, "conversation_id": data.get("conversation_id")},
        )
    return JSONResponse({"ok": True, "turns": len(transcript)})


@app.post("/agent/capture/{conversation_id}")
async def capture(conversation_id: str):
    """Manually capture an ElevenLabs conversation's transcript into the feed
    (e.g. after a 'Talk to agent' widget test — pass its conversation_id)."""
    from vigil.escalation.elevenlabs_call import capture_conversation_async

    chart = _active()
    capture_conversation_async(conversation_id, chart.name if chart else None)
    return {"ok": True, "capturing": conversation_id}


@app.get("/video")
def video():
    def gen():
        boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
        while not state.stop.is_set():
            jpg = frames.get()
            if jpg:
                yield boundary + jpg + b"\r\n"
            time.sleep(1 / 25)

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.websocket("/events")
async def events(ws: WebSocket):
    await ws.accept()
    q = bus.subscribe()
    chart = _active()
    if chart is not None:
        await ws.send_json({"type": "patient", "payload": _patient_payload(chart)})
    await ws.send_json({"type": "status", "payload": {"capabilities": _capabilities()}})
    try:
        while True:
            ev: BusEvent = await q.get()
            await ws.send_json(ev.model_dump())
    except WebSocketDisconnect:
        pass
    finally:
        bus.unsubscribe(q)


def _patient_payload(chart: PatientChart) -> dict:
    return {
        "id": chart.patient_id,
        "name": chart.name,
        "age": chart.age,
        "gender": chart.gender,
        "visit": chart.visit_title,
        "baseline_esi": chart.baseline_esi,
        "conditions": chart.active_conditions,
        "medications": chart.active_medications,
        "vitals": {k: f"{v.value:g}{v.unit}" for k, v in chart.latest_vitals.items()},
        "avatar": _avatar_for(chart.patient_id),
    }


@app.get("/patients")
async def patients():
    return JSONResponse(
        {
            "active": state.active_id,
            "patients": [_patient_payload(c) for c in state.charts.values()],
        }
    )


@app.post("/active/{patient_id}")
async def set_active(patient_id: str):
    if patient_id not in state.charts:
        return JSONResponse({"error": "unknown patient"}, status_code=404)
    state.active_id = patient_id
    chart = _active()
    bus.publish(BusEvent(type="patient", payload=_patient_payload(chart)))
    return {"active": patient_id, "name": chart.name}


@app.post("/intake")
async def intake(
    patient_id: str = Form(default=""),
    text: str = Form(default=""),
    audio: UploadFile | None = File(default=None),
):
    """Initial ESI triage from a spoken (or typed) patient intake — voice in, ESI out.

    Transcribes the audio (ElevenLabs Scribe), runs the full ESI v4 decision tree
    (initial_triage.grade), and — when the intake is bound to a known chart — sets that
    patient's baseline_esi so the continuous re-triage loop escalates from this grade.
    """
    transcript = (text or "").strip()
    if audio is not None:
        data = await audio.read()
        try:
            transcript = await asyncio.to_thread(
                voice_intake.transcribe,
                data,
                filename=audio.filename or "intake.webm",
                content_type=audio.content_type or "application/octet-stream",
            )
        except Exception as e:  # noqa: BLE001 — surface a real STT failure honestly
            raise HTTPException(status_code=502, detail=f"speech-to-text failed: {e}") from e
    if not transcript:
        raise HTTPException(status_code=400, detail="provide `text` or an `audio` file")

    chart = state.charts.get(patient_id) if patient_id else None
    try:
        decision = await asyncio.to_thread(initial_triage.grade, transcript, chart)
    except initial_triage.ReasoningNotConfigured as e:
        raise HTTPException(status_code=503, detail=str(e)) from e

    if chart is not None:
        chart.baseline_esi = decision.esi  # the re-triage loop now monitors from here
        bus.publish(BusEvent(type="patient", payload=_patient_payload(chart)))
    bus.publish(BusEvent(type="initial_triage", payload=decision.model_dump()))
    return JSONResponse(decision.model_dump())
