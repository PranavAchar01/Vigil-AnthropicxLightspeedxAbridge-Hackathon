"""Preflight — verify THIS laptop is ready to serve Vigil publicly.

Run on the machine that will run the backend (e.g. Sahiel's laptop), from the repo root:

    uv run python scripts/preflight.py       (or:  .venv/bin/python scripts/preflight.py)

It checks Python deps, the required secrets (.env), the patient cohort, the ML
models, cloudflared, and Supabase config, then prints a checklist and a final
verdict. IMPORTANT: git does NOT carry your secrets or private data — .env,
data/demo_cohort.json, models/*.pt and the face gallery are gitignored and must be
COPIED to this laptop from a set-up machine. This script tells you which are missing.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

GREEN, RED, YEL = (
    "\033[32m",
    "\033[31m",
    "\033[33m",
)
RESET = "\033[0m"
rows: list[tuple[str, str, str]] = []  # (mark, label, detail)
blockers = 0


def add(ok: bool, label: str, ok_detail: str, bad_detail: str, warn: bool = False) -> bool:
    global blockers
    if ok:
        rows.append((f"{GREEN}✓{RESET}", label, ok_detail))
    elif warn:
        rows.append((f"{YEL}!{RESET}", label, bad_detail))
    else:
        rows.append((f"{RED}✗{RESET}", label, bad_detail))
        blockers += 1
    return ok


def has(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError):
        return False


# ---- Python dependencies ------------------------------------------------- #
DEPS = [
    ("cv2", "opencv — camera / vision", True),
    ("ultralytics", "YOLO pose + fall model", True),
    ("anthropic", "Claude reasoning + triage", True),
    ("fastapi", "web server", True),
    ("uvicorn", "ASGI server", True),
    ("requests", "HTTP — ElevenLabs STT", True),
    ("sounddevice", "microphone capture", True),
    ("multipart", "python-multipart — /intake upload", True),
    ("transformers", "AST scream model (audio extra)", False),
    ("insightface", "face recognition (faces extra)", False),
]
for mod, label, required in DEPS:
    add(
        has(mod),
        f"dep · {label}",
        "installed",
        "MISSING — run:  uv sync --extra audio --extra faces",
        warn=not required,
    )

# ---- Secrets (.env) ------------------------------------------------------ #
try:
    from vigil.config import settings

    add(
        bool(settings.anthropic_api_key),
        "secret · ANTHROPIC_API_KEY",
        "set",
        "MISSING — copy .env to this laptop",
    )
    add(
        bool(settings.elevenlabs_api_key),
        "secret · ELEVENLABS_API_KEY",
        "set",
        "MISSING — voice-intake speech-to-text needs it",
    )
    add(
        bool(settings.supabase_url),
        "secret · SUPABASE_URL",
        "set",
        "MISSING — Vercel↔backend rendezvous needs it",
    )
    add(
        bool(settings.supabase_secret_key),
        "secret · SUPABASE_SECRET_KEY",
        "set",
        "MISSING — publishing events/tunnel URL needs it",
    )
    add(
        bool(settings.supabase_publishable_key),
        "secret · SUPABASE_PUBLISHABLE_KEY",
        "set",
        "MISSING — used by the frontend",
        warn=True,
    )
    add(
        bool(settings.nurse_phone_number or settings.twilio_from_number),
        "config · nurse / Twilio phone",
        "set — nurse call live",
        "not set — nurse call disabled (rest still works)",
        warn=True,
    )
    cohort = settings.cohort_path
    n = 0
    if cohort.exists():
        try:
            n = len(json.loads(cohort.read_text()))
        except (ValueError, OSError):
            n = 0
    add(
        cohort.exists() and n > 0,
        f"data · {cohort.name}",
        f"{n} patients",
        "MISSING — copy data/demo_cohort.json (gitignored)",
    )
    add(
        settings.face_gallery_path.exists(),
        "data · face_gallery.json",
        "present — face→chart binding",
        "absent — face rec off, fallback patient used",
        warn=True,
    )
    add(
        (ROOT / "config" / "tuned.env").exists(),
        "config · tuned.env",
        "present",
        "absent — using default thresholds",
        warn=True,
    )
    add(
        settings.fall_model_path.exists(),
        f"model · {settings.fall_model_path.name}",
        "present",
        "absent — dedicated fall model; copy models/fall_yolov11.pt",
        warn=True,
    )
    add(
        (ROOT / "yolo11n-pose.pt").exists(),
        "model · yolo11n-pose.pt",
        "present",
        "absent — auto-downloads on first run",
        warn=True,
    )
except Exception as e:  # noqa: BLE001 — config import shouldn't hard-crash preflight
    add(False, "config · load .env / settings", "", f"ERROR loading config: {e}")

# ---- External tools ------------------------------------------------------ #
add(
    shutil.which("cloudflared") is not None,
    "tool · cloudflared",
    "installed",
    "MISSING — run:  brew install cloudflared",
)

# ---- Report -------------------------------------------------------------- #
print("\n  VIGIL PREFLIGHT — backend readiness\n  " + "─" * 60)
for mark, label, detail in rows:
    print(f"  {mark}  {label:<44}{detail}")
print("  " + "─" * 60)
if blockers == 0:
    print(f"  {GREEN}READY.{RESET} Start it with:  .venv/bin/python scripts/serve_public.py\n")
    sys.exit(0)
print(f"  {RED}NOT READY — {blockers} blocker(s) above must be fixed first.{RESET}\n")
sys.exit(1)
