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
HARD_ALONE: set[PerceptionKind] = {"fall", "collapse", "companion_alarm"}
# Signals that are ambiguous alone → voice check-in first.
SOFT_ALONE: set[PerceptionKind] = {
    "motionless",
    "slump",
    "agitation",
    "chest_clutch",
    "gait_instability",
    "labored_breathing",
    "distress_phrase",
    "non_response",
}


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
        has_collapse = "collapse" in kinds
        has_scream = "scream" in kinds
        if "companion_alarm" in kinds:
            return Severity.HARD, ["companion_alarm"], "Companion requested urgent help"

        visual_soft = kinds.intersection(
            {"motionless", "slump", "agitation", "chest_clutch", "gait_instability"}
        )
        audio_soft = kinds.intersection({"labored_breathing", "distress_phrase", "non_response"})
        if visual_soft and audio_soft:
            fused = [next(iter(visual_soft)), next(iter(audio_soft))]
            return Severity.HARD, fused, "Corroborated visual and audio deterioration"

        if "non_response" in kinds and ("scream" in kinds or "labored_breathing" in kinds):
            fused = [
                "non_response",
                "labored_breathing" if "labored_breathing" in kinds else "scream",
            ]
            return Severity.HARD, fused, "Distress signal followed by no response"
        down = has_fall or has_collapse
        down_kind: PerceptionKind = "collapse" if has_collapse else "fall"

        if down and has_scream:
            return Severity.HARD, ["scream", down_kind], "Scream + collapse detected"
        if has_collapse:
            return Severity.HARD, ["collapse"], "Patient collapsed and is motionless"
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
                "chest_clutch": "Repeated guarding at the chest",
                "gait_instability": "Gait instability detected",
                "labored_breathing": "Possible labored breathing",
                "distress_phrase": "Patient reported worsening symptoms",
                "non_response": "No response to directed check-in",
            }
            primary = soft[0]
            return Severity.SOFT, [primary], label[primary]

        return None, [], ""


def _more_severe(a: Severity, b: Severity) -> bool:
    order = {Severity.SOFT: 0, Severity.HARD: 1}
    return order[a] > order[b]
