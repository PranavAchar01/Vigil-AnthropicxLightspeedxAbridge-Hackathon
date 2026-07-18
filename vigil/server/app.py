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

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
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
from vigil.reasoning import triage
from vigil.reasoning.triage import ReasoningNotConfigured
from vigil.server.bus import EventBus, FrameBuffer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
log = logging.getLogger("vigil.server")

DASHBOARD_DIR = Path(__file__).resolve().parent.parent / "dashboard"
DASHBOARD = DASHBOARD_DIR / "index.html"
FONTS = DASHBOARD_DIR / "fonts.css"

bus = EventBus()
frames = FrameBuffer()
fuser = EventFuser(window_s=settings.fusion_window_seconds)


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
    }


# --------------------------------------------------------------------------- #
# Perception -> fusion -> incident pipeline (all on the event loop)
# --------------------------------------------------------------------------- #


def _on_perception(ev: PerceptionEvent) -> None:
    """Runs on the loop thread (scheduled via call_soon_threadsafe)."""
    bus.publish(BusEvent(type="perception", payload=ev.model_dump()))
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

    handler = NurseCallHandler(chart, bus=bus)
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


def _start_vision(sink) -> None:
    try:
        from vigil.perception.vision import run_vision

        run_vision(sink, frames, state.stop, source=0)
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
    _load_cohort()

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
