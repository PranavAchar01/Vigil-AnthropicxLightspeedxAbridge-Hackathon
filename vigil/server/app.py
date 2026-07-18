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
from dataclasses import replace
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse

from vigil.chart import PatientChart, load_cohort
from vigil.config import settings
from vigil.demo import DemoController
from vigil.documentation import abridge_note
from vigil.escalation.elevenlabs_call import (
    NurseCallHandler,
    checkin_configured,
    nurse_call_configured,
)
from vigil.escalation.ladder import run_ladder
from vigil.escalation.twilio_call import twilio_call_configured
from vigil.events import BusEvent, EscalationAction, FusedEvent, PerceptionEvent
from vigil.monitoring import MonitorRegistry
from vigil.perception.fusion import EventFuser
from vigil.reasoning import triage
from vigil.reasoning.rules import decide_tier_zero
from vigil.reasoning.triage import ReasoningNotConfigured
from vigil.security import AuditChain, BreakGlassManager
from vigil.server import status as pstatus
from vigil.server import supabase_sink as supa
from vigil.server.bus import EventBus, FrameBuffer
from vigil.server.command_api import build_command_router

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
log = logging.getLogger("vigil.server")

DASHBOARD_DIR = Path(__file__).resolve().parent.parent / "dashboard"
DASHBOARD = DASHBOARD_DIR / "index.html"
FONTS = DASHBOARD_DIR / "fonts.css"

bus = EventBus()
frames = FrameBuffer()
fuser = EventFuser(window_s=settings.fusion_window_seconds)
registry = MonitorRegistry(fusion_window_s=settings.fusion_window_seconds)
audit = AuditChain()
break_glass = BreakGlassManager()
demo = DemoController(registry, audit, bus.publish)


class State:
    charts: dict[str, PatientChart] = {}
    active_id: str = ""
    stop = threading.Event()


state = State()


def _load_cohort() -> None:
    if not settings.cohort_path.exists():
        log.warning(
            "cohort not found at %s — run scripts/extract_demo_cohort.py", settings.cohort_path
        )
        return
    charts = load_cohort(settings.cohort_path)
    state.charts = {c.patient_id: c for c in charts}
    hero = next(
        (c for c in charts if "covid" in c.visit_title.lower()), charts[0] if charts else None
    )
    state.active_id = hero.patient_id if hero else ""
    log.info(
        "loaded %d patients; active = %s",
        len(charts),
        state.charts.get(state.active_id, PatientChart("", "", "?", "", None, "")).name,
    )


def _active() -> PatientChart | None:
    return state.charts.get(state.active_id)


def _capabilities() -> dict[str, bool]:
    return {
        "reasoning": bool(settings.anthropic_api_key),
        "nurse_call": nurse_call_configured() or twilio_call_configured(),
        "patient_checkin": checkin_configured(),
        "cohort_loaded": bool(state.charts),
        "backend": supa.configured(),
        "multi_patient": True,
        "role_redaction": True,
        "audit_chain": True,
        "demo_replay": True,
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
        summary = (
            f"{p.get('summary')} / {str(p.get('severity')).upper()} "
            f"[{'+'.join(p.get('kinds', []))}]"
        )
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
    else:
        return  # skip noise: reasoning_delta, status/capabilities, patient
    supa.log_event(t, source=source, patient=patient, summary=summary, payload=p)


# --------------------------------------------------------------------------- #
# Perception -> fusion -> incident pipeline (all on the event loop)
# --------------------------------------------------------------------------- #


def _on_perception(ev: PerceptionEvent) -> None:
    """Runs on the loop thread (scheduled via call_soon_threadsafe)."""
    bus.publish(BusEvent(type="perception", payload=ev.model_dump()))
    if ev.kind in ("fall", "collapse", "scream") and state.active_id:
        pstatus.mark_event(state.active_id, ev.kind)
    # Route by persistent track binding. The first live track inherits the
    # currently recognized patient, then remains independent of later changes.
    binding = registry.binding(ev.track_id, now=ev.ts)
    if binding is None and state.active_id:
        registry.bind_track(ev.track_id, state.active_id, confidence=0.5, now=ev.ts)
    result = registry.ingest(ev)
    if result.patient_id and result.fused is not None:
        asyncio.create_task(handle_incident(result.fused, result.patient_id))
    elif result.safety_only and result.fused is not None:
        bus.publish(
            BusEvent(
                type="safety_alert",
                payload={
                    "track_id": ev.track_id,
                    "summary": result.fused.summary,
                    "clinical_context": None,
                },
            )
        )


async def _stream_text(kind: str, text: str, delay: float = 0.025) -> None:
    for word in text.split(" "):
        bus.publish(BusEvent(type=kind, payload={"text": word + " "}))
        await asyncio.sleep(delay)


async def handle_incident(fused: FusedEvent, patient_id: str | None = None) -> None:
    chart = state.charts.get(patient_id or "") or _active()
    if chart is None:
        log.warning("incident with no active patient; ignoring")
        return

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

    monitor = registry.monitor(chart.patient_id)
    if monitor is None:
        log.warning("incident for patient without monitor state: %s", chart.patient_id)
        return
    if settings.anthropic_api_key:
        # The optional Tier 1 call receives the current monotonic ESI, not the
        # original arrival value.
        current_chart = replace(chart, baseline_esi=monitor.current_esi)
        try:
            decision = await asyncio.to_thread(triage.decide, current_chart, fused)
        except ReasoningNotConfigured:
            decision = decide_tier_zero(chart, monitor, fused)
    else:
        decision = decide_tier_zero(chart, monitor, fused)

    await _stream_text("reasoning_delta", decision.rationale)
    bus.publish(BusEvent(type="decision", payload=decision.model_dump()))
    registry.apply_decision(decision)
    alert = registry.create_or_update_alert(decision, fused)
    bus.publish(BusEvent(type="alert", payload=alert.model_dump(mode="json")))
    audit.append(
        actor=f"vigil-{decision.reasoning_tier}",
        role="system",
        action="retriage_decision",
        resource=f"patient:{chart.patient_id}",
        outcome="escalated" if decision.escalate else "held",
        metadata={
            "patient_id": chart.patient_id,
            "prior_esi": decision.prior_esi,
            "new_esi": decision.new_esi,
            "input_snapshot_hash": decision.input_snapshot_hash,
        },
    )
    # record for the voice agent's get_patient_status tool (pre-computed; never in the call path)
    pstatus.update_retriage(
        chart.patient_id,
        decision.new_esi,
        decision.prior_esi,
        decision.rationale,
        decision.spoken_summary,
        chart.to_context(),
    )

    if nurse_call_configured() or twilio_call_configured() or checkin_configured():
        handler = NurseCallHandler(chart, bus=bus)
        actions = await asyncio.to_thread(run_ladder, decision, handler)
    else:
        kind = "nurse_call" if decision.action.value == "page_immediately" else "patient_checkin"
        actions = [
            EscalationAction(
                kind=kind,
                target="charge_nurse" if kind == "nurse_call" else "patient_kiosk",
                message=decision.spoken_summary,
                status="pending",
            )
        ]
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

        run_vision(
            sink,
            frames,
            state.stop,
            source=0,
            status_sink=_vision_status,
            identify_sink=_vision_identify,
        )
    except Exception as e:  # noqa: BLE001
        log.warning("vision disabled (%r) — install ML deps and connect a camera", e)


def _start_audio(sink) -> None:
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
    demo.reset()
    state.charts.update(demo.charts)
    if not state.active_id and demo.charts:
        state.active_id = next(iter(demo.charts))

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
app.include_router(build_command_router(registry, audit, break_glass, demo, bus))

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


@app.get("/health")
async def health():
    return JSONResponse({"capabilities": _capabilities(), "active_patient": state.active_id})


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
