"""Fusion tuning: labeled perception-event streams → correct severity + anti-spam.

Tunes the sliding-window fuser's window (how close two signals must be to fuse into
one event) and cooldown (how aggressively repeats are suppressed). Objective: fire
the right severity on each labeled stream, fuse scream+fall when co-occurring, and
never spam the same emergency.
"""

from __future__ import annotations

from vigil.events import Modality, PerceptionEvent, Severity
from vigil.perception.fusion import EventFuser
from vigil.tuning.optimizer import Param

A, V = Modality.AUDIO, Modality.VISION


def _ev(t, kind, conf, mod=V):
    return PerceptionEvent(ts=100.0 + t, modality=mod, kind=kind, confidence=conf)


def streams():
    return [
        (
            [_ev(0, "scream", 0.7, A), _ev(1.5, "fall", 0.8)],
            {"hard": (1, 1), "soft": (0, 0), "fuse_scream_fall": True},
        ),
        ([_ev(0, "fall", 0.8)], {"hard": (1, 1), "soft": (0, 0)}),
        ([_ev(0, "seizure", 0.85)], {"hard": (1, 1), "soft": (0, 0)}),
        ([_ev(0, "collapse", 0.85)], {"hard": (1, 1), "soft": (0, 0)}),
        ([_ev(0, "motionless", 0.6)], {"hard": (0, 0), "soft": (1, 1)}),
        ([_ev(0, "chest_clutch", 0.6)], {"hard": (0, 0), "soft": (1, 1)}),
        ([_ev(0, "slump", 0.55)], {"hard": (0, 0), "soft": (1, 1)}),
        ([_ev(0, "scream", 0.4, A)], {"hard": (0, 0), "soft": (1, 1)}),
        ([_ev(0, "scream", 0.7, A)], {"hard": (1, 1), "soft": (0, 0)}),
        ([_ev(i * 0.5, "fall", 0.8) for i in range(10)], {"hard": (1, 2)}),  # spam suppression
        ([], {"hard": (0, 0), "soft": (0, 0)}),
    ]


def space():
    return [Param("window_s", 1.0, 8.0), Param("cooldown_s", 2.0, 12.0)]


def make_eval(strms):
    def eval_fn(cand: dict):
        checks = ok = 0
        spam = 0
        for events, want in strms:
            fuser = EventFuser(window_s=cand["window_s"], cooldown_s=cand["cooldown_s"])
            fired = [f for f in (fuser.add(e) for e in events) if f is not None]
            n_hard = sum(f.severity == Severity.HARD for f in fired)
            n_soft = sum(f.severity == Severity.SOFT for f in fired)
            if "hard" in want:
                lo, hi = want["hard"]
                checks += 1
                ok += lo <= n_hard <= hi
                spam += max(0, n_hard - hi)
            if "soft" in want:
                lo, hi = want["soft"]
                checks += 1
                ok += lo <= n_soft <= hi
            if want.get("fuse_scream_fall"):
                checks += 1
                ok += any(
                    f.severity == Severity.HARD and {"scream", "fall"} <= set(f.kinds)
                    for f in fired
                )
        acc = ok / max(checks, 1)
        score = acc - 0.05 * spam
        return score, {"accuracy": round(acc, 3), "checks": checks, "passed": ok, "spam": spam}

    return eval_fn
