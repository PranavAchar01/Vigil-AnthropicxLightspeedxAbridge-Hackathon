"""Thread-safe live patient-status store — the data the voice agent 'looks at'.

Two writers, no blocking readers:
  - the vision thread calls update_vision()/mark_event() every frame with the
    current posture + motion of the person on camera;
  - the re-triage loop calls update_retriage() right after Claude decides.

The agent's webhook tool reads snapshot() — a pure, lock-guarded cache copy. We
NEVER call Claude (or anything slow) in the read path: the spoken_summary and ESI
are pre-computed by the background loop, so a mid-call fetch is a dict copy.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone

_LOCK = threading.Lock()
_STATUS: dict[str, dict] = {}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _default(pid: str) -> dict:
    now = time.monotonic()
    return {
        "patient_id": pid,
        "posture": "unknown",
        "motion_level": "unknown",
        "last_frame_monotonic": now,
        "last_movement_monotonic": now,
        "fall_detected": False,
        "last_fall_ts": None,
        "scream_detected": False,
        "last_scream_ts": None,
        "esi_level": None,
        "esi_prev": None,
        "esi_changed": False,
        "triage_rationale": "",
        "chart_summary": "",
        "spoken_summary": "Monitoring; no re-triage event yet.",
    }


def update_vision(pid: str, posture: str, motion_level: str, moved: bool) -> None:
    """Per-frame continuous state from the vision thread."""
    now = time.monotonic()
    with _LOCK:
        s = _STATUS.setdefault(pid, _default(pid))
        s["posture"] = posture
        s["motion_level"] = motion_level
        s["last_frame_monotonic"] = now
        if moved:
            s["last_movement_monotonic"] = now


def mark_event(pid: str, kind: str) -> None:
    """Discrete perception event (fall / scream)."""
    with _LOCK:
        s = _STATUS.setdefault(pid, _default(pid))
        if kind in ("fall", "collapse"):
            s["fall_detected"] = True
            s["last_fall_ts"] = _now_iso()
            s["posture"] = "on the floor"
        elif kind == "scream":
            s["scream_detected"] = True
            s["last_scream_ts"] = _now_iso()


def update_retriage(
    pid: str,
    new_esi: int,
    prev_esi: int,
    rationale: str,
    spoken_summary: str,
    chart_summary: str = "",
) -> None:
    with _LOCK:
        s = _STATUS.setdefault(pid, _default(pid))
        s["esi_level"] = new_esi
        s["esi_prev"] = prev_esi
        s["esi_changed"] = new_esi != prev_esi
        s["triage_rationale"] = rationale
        s["spoken_summary"] = spoken_summary
        if chart_summary:
            s["chart_summary"] = chart_summary


def snapshot(pid: str) -> dict:
    """Pure cache read → the JSON the agent's webhook tool returns. No I/O, no LLM."""
    now = time.monotonic()
    with _LOCK:
        s = dict(_STATUS.get(pid) or _default(pid))
    esi, prev = s["esi_level"], s["esi_prev"]
    direction = (
        "worsening"
        if (s["esi_changed"] and prev is not None and esi is not None and esi < prev)
        else "stable"
    )
    feed_age_ms = round((now - s["last_frame_monotonic"]) * 1000)
    return {
        "patient_id": pid,
        "captured_at": _now_iso(),
        "in_view": feed_age_ms < 3000,
        "feed_age_ms": feed_age_ms,
        "seconds_since_last_movement": round(now - s["last_movement_monotonic"]),
        "posture": s["posture"],
        "motion_level": s["motion_level"],
        "fall_detected": s["fall_detected"],
        "last_fall_ts": s["last_fall_ts"],
        "scream_detected": s["scream_detected"],
        "last_scream_ts": s["last_scream_ts"],
        "triage": {
            "esi_level": esi,
            "esi_prev": prev,
            "esi_changed": s["esi_changed"],
            "direction": direction,
            "rationale": s["triage_rationale"],
        },
        "chart_summary": s["chart_summary"],
        "spoken_summary": s["spoken_summary"],
    }
