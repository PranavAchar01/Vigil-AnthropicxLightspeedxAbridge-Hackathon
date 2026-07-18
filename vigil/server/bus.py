"""In-process event bus + frame buffer for the single-worker demo server.

Two transports leave the process (see server/app.py):
  - a WebSocket carrying small JSON BusEvents (subscribers here), and
  - an HTTP MJPEG stream carrying only the annotated skeleton frames.

Frames NEVER go over the WebSocket (head-of-line blocking would stutter the
reasoning trace). Perception runs in daemon threads and must publish only via
`publish_from_thread`, which hops onto the event loop with call_soon_threadsafe.
"""

from __future__ import annotations

import asyncio
import threading

from vigil.events import BusEvent


class FrameBuffer:
    """Holds the latest annotated JPEG frame; written by the vision thread,
    read by the MJPEG generator."""

    def __init__(self) -> None:
        self._data: bytes = b""
        self._lock = threading.Lock()

    def set(self, jpeg: bytes) -> None:
        with self._lock:
            self._data = jpeg

    def get(self) -> bytes:
        with self._lock:
            return self._data


class EventBus:
    """Fan-out to per-subscriber bounded queues with drop-oldest backpressure."""

    def __init__(self) -> None:
        self._subs: set[asyncio.Queue[BusEvent]] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._tap = None  # optional observer: called with every event (e.g. Supabase logger)

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def set_tap(self, fn) -> None:
        """Register an observer invoked (non-blockingly) for every published event."""
        self._tap = fn

    def subscribe(self) -> asyncio.Queue[BusEvent]:
        q: asyncio.Queue[BusEvent] = asyncio.Queue(maxsize=64)
        self._subs.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[BusEvent]) -> None:
        self._subs.discard(q)

    def _emit(self, event: BusEvent) -> None:
        for q in list(self._subs):
            if q.full():
                try:
                    q.get_nowait()  # drop oldest
                except asyncio.QueueEmpty:
                    pass
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                pass
        if self._tap is not None:
            try:
                self._tap(event)
            except Exception:  # noqa: BLE001 — a logging failure must never break the bus
                pass

    def publish(self, event: BusEvent) -> None:
        """Publish from the event-loop thread (coroutines)."""
        self._emit(event)

    def publish_from_thread(self, event: BusEvent) -> None:
        """Publish from a perception daemon thread."""
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._emit, event)
