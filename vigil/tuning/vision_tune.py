"""Vision-detector tuning: a labeled synthetic benchmark + evaluator + search space.

We generate many variants of every patient scenario (varying scale, position, speed,
jitter, and framing) plus hard negatives (sitting, waving, reaching, standing,
walking, head-turns, brief-lean-and-recover). The evaluator runs the real detection
core (VisionDetector.analyze) under a candidate parameter set and scores it:
maximize recall on positives while punishing hard false-positives on negatives.
"""

from __future__ import annotations

import numpy as np

from vigil.perception.vision import (
    L_EAR,
    L_EYE,
    L_HIP,
    L_SH,
    L_WR,
    NOSE,
    R_EAR,
    R_EYE,
    R_HIP,
    R_SH,
    R_WR,
    Params,
    PersonState,
    VisionDetector,
)
from vigil.tuning.optimizer import Param

L_EL, R_EL = 7, 8
H_FR = 480.0
DT = 0.05
HARD = {"fall", "collapse", "seizure", "unresponsive"}


# --------------------------------------------------------------------------- #
# Pose synthesis
# --------------------------------------------------------------------------- #
def pose(
    sx, sy, s=120.0, head_dx=0.0, head_dy=-0.7, wrist="side", hips=False, jitter=0.0, rng=None
):
    kp = np.zeros((17, 2), float)
    kc = np.zeros(17)

    def setp(i, x, y):
        j = rng.normal(0, jitter, 2) if (rng is not None and jitter) else 0.0
        kp[i] = (x, y) + j
        kc[i] = 0.9

    nx, ny = sx + head_dx * s, sy + head_dy * s
    setp(NOSE, nx, ny)
    setp(L_EYE, nx - 0.1 * s, ny - 0.05 * s)
    setp(R_EYE, nx + 0.1 * s, ny - 0.05 * s)
    setp(L_EAR, nx - 0.2 * s, ny)
    setp(R_EAR, nx + 0.2 * s, ny)
    setp(L_SH, sx - 0.5 * s, sy)
    setp(R_SH, sx + 0.5 * s, sy)
    setp(L_EL, sx - 0.6 * s, sy + 0.6 * s)
    setp(R_EL, sx + 0.6 * s, sy + 0.6 * s)
    if wrist == "side":
        setp(L_WR, sx - 0.7 * s, sy + 1.2 * s)
        setp(R_WR, sx + 0.7 * s, sy + 1.2 * s)
    elif wrist == "chest":
        setp(L_WR, sx - 0.12 * s, sy + 0.4 * s)
        setp(R_WR, sx + 0.12 * s, sy + 0.4 * s)
    elif wrist == "up":  # waving / reaching — hands high but off-center
        setp(L_WR, sx - 0.9 * s, sy - 0.6 * s)
        setp(R_WR, sx + 0.9 * s, sy - 0.6 * s)
    if hips:
        setp(L_HIP, sx - 0.35 * s, sy + 1.6 * s)
        setp(R_HIP, sx + 0.35 * s, sy + 1.6 * s)
    pts = kp[kc >= 0.5]
    x1, y1 = pts.min(0)
    x2, y2 = pts.max(0)
    return np.array([x1 - 8, y1 - 8, x2 + 8, y2 + 8]), kp, kc


def _seq(builder, seconds, **kw):
    return [builder(**kw) for _ in range(int(seconds / DT))]


# --------------------------------------------------------------------------- #
# Labeled benchmark  (list of (frames, expected_kind_or_None))
# --------------------------------------------------------------------------- #
def build_dataset(seed: int = 0) -> list[tuple[list, str | None]]:
    rng = np.random.default_rng(seed)
    items: list[tuple[list, str | None]] = []

    def variants(n):
        for _ in range(n):
            yield {
                "s": rng.uniform(95, 210),
                "cx": rng.uniform(240, 400),
                "cy": rng.uniform(120, 175),
                "jit": rng.uniform(0.4, 2.2),
            }

    # ---- POSITIVES -------------------------------------------------------- #
    for v in variants(8):  # FALL
        s, cx, cy0, jit = v["s"], v["cx"], v["cy"], v["jit"]
        f = [pose(cx, cy0, s, jitter=jit, rng=rng) for _ in range(20)]
        drop_px = rng.uniform(220, 300)
        for i in range(11):
            f.append(pose(cx, cy0 + drop_px * i / 10, s, jitter=jit, rng=rng))
        f += _seq(lambda: pose(cx, cy0 + drop_px, s, jitter=jit, rng=rng), 0.6)
        items.append((f, "fall"))

    for v in variants(8):  # COLLAPSE (slump head-down, then still)
        s, cx, cy0, jit = v["s"], v["cx"], v["cy"], v["jit"]
        f = [pose(cx, cy0, s, jitter=jit, rng=rng) for _ in range(10)]
        for i in range(10):
            f.append(pose(cx, cy0 + 2, s, head_dy=-0.7 + 0.11 * i, jitter=jit, rng=rng))
        f += _seq(lambda: pose(cx, cy0 + 2, s, head_dy=0.4, jitter=jit * 0.4, rng=rng), 3.4)
        items.append((f, "collapse"))

    for v in variants(8):  # SEIZURE
        s, cx, cy0, jit = v["s"], v["cx"], v["cy"], v["jit"]
        f = [pose(cx, cy0, s, jitter=jit, rng=rng) for _ in range(6)]
        amp = rng.uniform(12, 20)
        for i in range(46):
            sgn = -1 if i % 2 else 1
            f.append(pose(cx + amp * sgn, cy0 + amp * 0.5 * sgn, s, jitter=jit, rng=rng))
        items.append((f, "seizure"))

    for v in variants(6):  # UNRESPONSIVE (prolonged stillness)
        s, cx, cy0, jit = v["s"], v["cx"], v["cy"], v["jit"] * 0.3
        items.append((_seq(lambda: pose(cx, cy0, s, jitter=jit, rng=rng), 11.0), "unresponsive"))

    for v in variants(6):  # CHEST_CLUTCH
        s, cx, cy0, jit = v["s"], v["cx"], v["cy"], v["jit"]
        items.append(
            (
                _seq(lambda: pose(cx, cy0, s, wrist="chest", jitter=jit, rng=rng), 2.2),
                "chest_clutch",
            )
        )

    for v in variants(6):  # SLUMP (sustained lean)
        s, cx, cy0, jit = v["s"], v["cx"], v["cy"], v["jit"]
        items.append(
            (_seq(lambda: pose(cx, cy0, s, head_dx=0.85, jitter=jit, rng=rng), 5.4), "slump")
        )

    # ---- NEGATIVES (must NOT fire a hard event) --------------------------- #
    for v in variants(6):  # normal sitting, slow side-to-side
        s, cx, cy0, jit = v["s"], v["cx"], v["cy"], v["jit"]
        f = []
        x = cx
        for i in range(120):
            x += 5 if (i // 20) % 2 == 0 else -5
            f.append(pose(x, cy0, s, jitter=jit, rng=rng))
        items.append((f, None))

    for v in variants(5):  # waving / arms-up with a gentle sway — must not fire hard
        s, cx, cy0, jit = v["s"], v["cx"], v["cy"], v["jit"]
        f = []
        for i in range(80):
            x = cx + 8.0 * np.sin(i * 0.25)  # low-frequency sway (few reversals ≠ seizure)
            f.append(pose(x, cy0, s, wrist="up", jitter=jit, rng=rng))
        items.append((f, None))

    for v in variants(5):  # reaching down (bends briefly then back up) — not a fall
        s, cx, cy0, jit = v["s"], v["cx"], v["cy"], v["jit"]
        f = [pose(cx, cy0, s, jitter=jit, rng=rng) for _ in range(10)]
        for i in range(8):
            f.append(pose(cx, cy0 + 10 * i, s, jitter=jit, rng=rng))  # gentle bend (<low_cy)
        for i in range(8):
            f.append(pose(cx, cy0 + 80 - 10 * i, s, jitter=jit, rng=rng))  # back up
        f += [pose(cx, cy0, s, jitter=jit, rng=rng) for _ in range(10)]
        items.append((f, None))

    for v in variants(4):  # brief lean then recover — not a slump/collapse
        s, cx, cy0, jit = v["s"], v["cx"], v["cy"], v["jit"]
        f = [pose(cx, cy0, s, head_dx=0.85, jitter=jit, rng=rng) for _ in range(40)]  # 2s < slump_s
        f += [pose(cx, cy0, s, jitter=jit, rng=rng) for _ in range(20)]
        items.append((f, None))

    return items


# --------------------------------------------------------------------------- #
# Search space + evaluator
# --------------------------------------------------------------------------- #
def space() -> list[Param]:
    return [
        Param("drop_frac", 0.06, 0.20),
        Param("fall_confirm_s", 0.15, 0.55),
        Param("low_cy", 0.50, 0.75),
        Param("torso_horiz_deg", 45.0, 70.0),
        Param("ar_wide", 0.90, 1.40),
        Param("still_motion", 0.30, 1.00),
        Param("faint_s", 1.5, 4.5),
        Param("motionless_s", 2.0, 5.0),
        Param("unresponsive_s", 6.0, 13.0),
        Param("seizure_motion", 1.2, 3.8),
        Param("seizure_osc", 3, 9, integer=True),
        Param("seizure_s", 0.6, 1.8),
        Param("gesture_s", 0.8, 2.8),
        Param("slump_min_deg", 18.0, 34.0),
        Param("slump_s", 3.0, 7.5),
    ]


def _detector(params: Params):
    det = object.__new__(VisionDetector)
    det.model = None
    det.device = None
    det.emitted = []
    det.emit = det.emitted.append
    det.emit_status = lambda *a: None
    det.p = params
    det.st = PersonState()
    det._last_status_ts = 0.0
    det.metrics = {}
    det.flash = None
    return det


def _run(params: Params, frames) -> tuple[set, int | None]:
    det = _detector(params)
    ts = 100.0
    first_at = {}
    for fi, (box, kp, kc) in enumerate(frames):
        det.analyze(box, kp, kc, H_FR, ts)
        for e in det.emitted:
            first_at.setdefault(e.kind, fi)
        ts += DT
    return {e.kind for e in det.emitted}, first_at


def make_eval(dataset):
    pos = [(f, k) for f, k in dataset if k is not None]
    neg = [f for f, k in dataset if k is None]

    def eval_fn(cand: dict):
        params = Params(**cand)
        tp, lat = 0, []
        per = {}
        for frames, kind in pos:
            kinds, first = _run(params, frames)
            hit = kind in kinds
            tp += hit
            per[kind] = per.get(kind, [0, 0])
            per[kind][0] += hit
            per[kind][1] += 1
            if hit:
                lat.append(first[kind] * DT)
        fp = 0
        for frames in neg:
            kinds, _ = _run(params, frames)
            if kinds & HARD:
                fp += 1
        recall = tp / max(len(pos), 1)
        fpr = fp / max(len(neg), 1)
        mean_lat = float(np.mean(lat)) if lat else 0.0
        score = recall - 1.6 * fpr - 0.02 * mean_lat
        metrics = {
            "recall": round(recall, 3),
            "fp_rate": round(fpr, 3),
            "detected": tp,
            "n_pos": len(pos),
            "n_neg_fp": fp,
            "mean_latency_s": round(mean_lat, 2),
            "per_scenario": {k: f"{v[0]}/{v[1]}" for k, v in per.items()},
        }
        return score, metrics

    return eval_fn
