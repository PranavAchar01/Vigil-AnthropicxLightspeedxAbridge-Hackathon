"""Fire-and-forget event logging to the Supabase observability backend.

Every meaningful thing Vigil does — a perception input, the event the voice agent
is fed, a re-triage decision, an escalation, a tool call, a conversation turn — is
mirrored into one `vigil_events` table so judges can watch the whole system live in
Supabase / the Vercel view. Inserts run on a background worker thread and never
block the perception or reasoning loops; if Supabase is down we drop, never stall.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import urllib.request
from typing import Any

from vigil.config import settings

log = logging.getLogger("vigil.supabase")

_Q: queue.Queue[dict] = queue.Queue(maxsize=2000)
_worker_started = False
_lock = threading.Lock()


def configured() -> bool:
    return bool(settings.supabase_url and settings.supabase_secret_key)


def _headers() -> dict:
    return {
        "apikey": settings.supabase_secret_key,
        "Authorization": f"Bearer {settings.supabase_secret_key}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    }


def _post(row: dict) -> None:
    req = urllib.request.Request(
        f"{settings.supabase_url}/rest/v1/vigil_events",
        data=json.dumps(row).encode(),
        method="POST",
        headers=_headers(),
    )
    urllib.request.urlopen(req, timeout=10).read()


def _worker() -> None:
    while True:
        row = _Q.get()
        try:
            _post(row)
        except Exception as e:  # noqa: BLE001 — never let logging break the app
            log.debug("supabase insert failed: %r", e)
        finally:
            _Q.task_done()


def _ensure_worker() -> None:
    global _worker_started
    with _lock:
        if not _worker_started:
            threading.Thread(target=_worker, daemon=True).start()
            _worker_started = True


def set_backend_url(url: str) -> None:
    """Publish this backend's current public URL to the vigil_runtime rendezvous row
    (id='backend') so the public Vercel page discovers it at load time. Upsert on id;
    synchronous (called once at startup, not in a hot path)."""
    if not configured() or not url:
        return
    from datetime import datetime, timezone

    row = {"id": "backend", "url": url, "updated_at": datetime.now(timezone.utc).isoformat()}
    req = urllib.request.Request(
        f"{settings.supabase_url}/rest/v1/vigil_runtime?on_conflict=id",
        data=json.dumps(row).encode(),
        method="POST",
        headers={**_headers(), "Prefer": "resolution=merge-duplicates,return=minimal"},
    )
    try:
        urllib.request.urlopen(req, timeout=10).read()
        log.info("published backend url to Supabase: %s", url)
    except Exception as e:  # noqa: BLE001
        log.warning("could not publish backend url: %r", e)


def log_event(
    type: str,
    *,
    source: str | None = None,
    patient: str | None = None,
    summary: str | None = None,
    payload: dict[str, Any] | None = None,
) -> None:
    """Enqueue an event for Supabase. Non-blocking; safe to call from any thread."""
    if not configured():
        return
    _ensure_worker()
    row = {
        "type": type,
        "source": source,
        "patient": patient,
        "summary": summary,
        "payload": payload or {},
    }
    try:
        _Q.put_nowait(row)
    except queue.Full:
        log.debug("supabase queue full; dropping event %s", type)
