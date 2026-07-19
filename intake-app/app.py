"""Vigil Intake — standalone voice-triage app.

A self-contained FastAPI server: serves the record-intake page, transcribes spoken
intake via ElevenLabs, and grades the initial ESI (1-5) via Claude against the ESI v4
rubric. No camera, no database, no other Vigil code required.

Run:
    pip install -r requirements.txt
    cp .env.example .env      # add ANTHROPIC_API_KEY and ELEVENLABS_API_KEY
    uvicorn app:app --port 8000
    # open http://localhost:8000
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

import triage
import voice

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
log = logging.getLogger("intake.app")

HERE = Path(__file__).resolve().parent
PAGE = HERE / "index.html"
COHORT = HERE / "sample_cohort.json"

app = FastAPI(title="Vigil Intake")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def _cohort() -> list[dict]:
    if COHORT.exists():
        try:
            return json.loads(COHORT.read_text())
        except (ValueError, OSError):
            return []
    return []


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    if PAGE.exists():
        return HTMLResponse(PAGE.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Vigil Intake</h1><p>index.html missing</p>")


@app.get("/patients")
async def patients() -> JSONResponse:
    rows = [{"id": c["patient_id"], "name": c["name"]} for c in _cohort()]
    return JSONResponse({"patients": rows})


@app.get("/health")
async def health() -> JSONResponse:
    import os

    return JSONResponse(
        {
            "reasoning": bool(os.environ.get("ANTHROPIC_API_KEY")),
            "voice": bool(os.environ.get("ELEVENLABS_API_KEY")),
            "patients": len(_cohort()),
        }
    )


@app.post("/intake")
async def intake(
    patient_id: str = Form(default=""),
    text: str = Form(default=""),
    audio: UploadFile | None = File(default=None),
) -> JSONResponse:
    transcript = (text or "").strip()
    if audio is not None:
        data = await audio.read()
        try:
            transcript = await asyncio.to_thread(
                voice.transcribe,
                data,
                filename=audio.filename or "intake.webm",
                content_type=audio.content_type or "application/octet-stream",
            )
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=502, detail=f"speech-to-text failed: {e}") from e
    if not transcript:
        raise HTTPException(status_code=400, detail="provide `text` or an `audio` file")

    chart = (
        next((c for c in _cohort() if c["patient_id"] == patient_id), None) if patient_id else None
    )
    try:
        decision = await asyncio.to_thread(triage.grade, transcript, chart)
    except triage.NotConfigured as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    return JSONResponse(decision)
