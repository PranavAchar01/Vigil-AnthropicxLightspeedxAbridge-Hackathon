"""Fuse vision + audio signals into one triage-worthy event.

The core insight of Vigil's false-positive defense: a scream *and* a fall in the
same short window is a far higher-confidence emergency than either alone (kills
the "someone just sat down fast" false positive). Fusion tags each event's
severity, which drives the escalation ladder downstream.
"""

from __future__ import annotations

from collections import deque

from vigil.events import FusedEvent, PerceptionEvent, PerceptionKind, Severity

# Signals that, on their own, already warrant a hard (page-now) response.
HARD_ALONE: set[PerceptionKind] = {"fall"}
# Signals that are ambiguous alone → voice check-in first.
SOFT_ALONE: set[PerceptionKind] = {"motionless", "slump", "agitation"}


class EventFuser:
    """Sliding-window fuser. Feed it PerceptionEvents; it emits FusedEvents.

    A cooldown prevents the same emergency from re-firing every frame.
    """

    def __init__(self, window_s: float = 4.0, cooldown_s: float = 8.0) -> None:
        self.window_s = window_s
        self.cooldown_s = cooldown_s
        self._buf: deque[PerceptionEvent] = deque(maxlen=64)
        self._last_fire_ts: float = 0.0
        self._last_severity: Severity | None = None

    def add(self, ev: PerceptionEvent) -> FusedEvent | None:
        self._buf.append(ev)
        self._prune(ev.ts)

        kinds = {e.kind for e in self._buf}
        conf_by_kind = {k: max(e.confidence for e in self._buf if e.kind == k) for k in kinds}

        severity, fused_kinds, summary = self._classify(kinds, conf_by_kind)
        if severity is None:
            return None

        # cooldown: don't spam the same-or-lower severity repeatedly
        if (
            ev.ts - self._last_fire_ts < self.cooldown_s
            and self._last_severity is not None
            and not _more_severe(severity, self._last_severity)
        ):
            return None

        self._last_fire_ts = ev.ts
        self._last_severity = severity
        confidence = max(conf_by_kind[k] for k in fused_kinds)
        return FusedEvent(
            ts=ev.ts,
            track_id=ev.track_id,
            kinds=fused_kinds,
            severity=severity,
            confidence=round(confidence, 3),
            summary=summary,
        )

    def _prune(self, now: float) -> None:
        while self._buf and now - self._buf[0].ts > self.window_s:
            self._buf.popleft()

    def _classify(
        self, kinds: set[PerceptionKind], conf: dict[PerceptionKind, float]
    ) -> tuple[Severity | None, list[PerceptionKind], str]:
        has_fall = "fall" in kinds
        has_scream = "scream" in kinds

        if has_fall and has_scream:
            return Severity.HARD, ["scream", "fall"], "Scream + collapse detected"
        if has_fall:
            return Severity.HARD, ["fall"], "Collapse detected"
        if has_scream and conf.get("scream", 0.0) >= 0.6:
            return Severity.HARD, ["scream"], "Loud scream / distress detected"
        if has_scream:
            return Severity.SOFT, ["scream"], "Possible distress vocalization"

        soft = [k for k in kinds if k in SOFT_ALONE]
        if soft:
            label = {
                "motionless": "Prolonged motionlessness",
                "slump": "Posture degraded / slumping",
                "agitation": "Agitation / pacing",
            }
            primary = soft[0]
            return Severity.SOFT, [primary], label[primary]

        return None, [], ""


def _more_severe(a: Severity, b: Severity) -> bool:
    order = {Severity.SOFT: 0, Severity.HARD: 1}
    return order[a] > order[b]
