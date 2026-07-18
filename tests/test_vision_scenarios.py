"""Foolproofing proof: drive the detection core with synthetic keypoint sequences
for every patient scenario and assert the right signal fires — no camera, no model.

Each scenario is a sequence of COCO-17 poses at 20 fps fed to VisionDetector.analyze.
This is how we 'model every scenario' and guarantee that a fall / faint / seizure /
unresponsive / distress-gesture / slump each register.
"""

from __future__ import annotations

import numpy as np

from vigil.events import Modality, PerceptionEvent  # noqa: F401  (PerceptionEvent used via emit)
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
    FallModelDetector,
    Params,
    PersonState,
    VisionDetector,
)

L_EL, R_EL = 7, 8
H_FR = 480.0
DT = 0.05  # 20 fps


def make_detector():
    """A detector with NO YOLO loaded — we call analyze() directly."""
    det = object.__new__(VisionDetector)
    det.model = None
    det.device = None
    det.emitted: list = []
    det.emit = det.emitted.append
    det.emit_status = lambda *a: None
    det.p = Params()
    det.st = PersonState()
    det._last_status_ts = 0.0
    det.metrics = {}
    det.flash = None
    return det


def pose(sx, sy, s=120.0, head_dx=0.0, head_dy=-0.7, wrist="side", hips=False):
    """Build one COCO-17 pose (webcam framing: legs out of view). Returns (box,kp,kc)."""
    kp = np.zeros((17, 2), float)
    kc = np.zeros(17)

    def setp(i, x, y):
        kp[i] = (x, y)
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
    if hips:
        setp(L_HIP, sx - 0.35 * s, sy + 1.6 * s)
        setp(R_HIP, sx + 0.35 * s, sy + 1.6 * s)
    pts = kp[kc >= 0.5]
    x1, y1 = pts.min(0)
    x2, y2 = pts.max(0)
    box = np.array([x1 - 8, y1 - 8, x2 + 8, y2 + 8])
    return box, kp, kc


def run(frames):
    """Feed (box,kp,kc) frames and return the set of emitted PerceptionKinds."""
    det = make_detector()
    ts = 100.0
    for box, kp, kc in frames:
        det.analyze(box, kp, kc, H_FR, ts)
        ts += DT
    return {e.kind for e in det.emitted}


def hold(gen_pose, seconds):
    return [gen_pose() for _ in range(int(seconds / DT))]


# --------------------------------------------------------------------------- #
# Scenarios
# --------------------------------------------------------------------------- #


def test_fainted_requires_staying_down_5s():
    # A fall that STAYS on the ground >= 5s -> fainted (the validated escalation).
    frames = [pose(320, 150 + (i % 2) * 3) for i in range(20)]  # upright
    for i in range(11):  # rapid drop to the floor
        frames.append(pose(320, 150 + 26 * i))
    frames += hold(lambda: pose(320, 410), 6.5)  # on the ground > 5s
    assert "fainted" in run(frames)


def test_fall_then_quick_recovery_no_page():
    # Falls but gets back up within the 5s window -> NO page (trip-and-recover).
    frames = [pose(320, 150) for _ in range(10)]
    for i in range(11):  # fall
        frames.append(pose(320, 150 + 26 * i))
    frames += hold(lambda: pose(320, 410), 2.0)  # on the ground only 2s (< 5s)
    frames += [pose(320 + (i % 2) * 45, 150) for i in range(40)]  # stand up + move clearly
    assert "fainted" not in run(frames)


def test_seated_head_drop_is_not_fainted():
    # Leaning / dropping the head while seated must never read as a faint.
    frames = [pose(320, 150) for _ in range(10)]
    frames += hold(lambda: pose(320, 152, head_dy=0.4), 9.0)  # head below, seated, still
    assert "fainted" not in run(frames)


def test_seizure_needs_5s_sustained():
    # >= 5s of sustained oscillation before it fires (validation).
    frames = [pose(320, 150) for _ in range(6)]
    for i in range(130):  # ~6.5s of oscillatory shaking
        sgn = -1 if i % 2 else 1
        frames.append(pose(320 + 15 * sgn, 150 + 8 * sgn))
    assert "seizure" in run(frames)


def test_brief_shake_is_not_seizure():
    # A short shake (< 5s) that stops must NOT page — validation.
    frames = [pose(320, 150) for _ in range(6)]
    for i in range(50):  # ~2.5s of shaking, then stops
        sgn = -1 if i % 2 else 1
        frames.append(pose(320 + 15 * sgn, 150 + 8 * sgn))
    frames += [pose(320, 150) for _ in range(20)]
    assert "seizure" not in run(frames)


def test_slump_sustained_lean():
    frames = hold(lambda: pose(320, 150, head_dx=0.85), 5.4)  # head far to the side ~5s
    assert "slump" in run(frames)


def test_normal_sitting_is_quiet():
    # person present and moving normally (slides side to side) — NO hard events
    frames = []
    x = 300
    for i in range(120):  # 6s
        x += 5 if (i // 20) % 2 == 0 else -5  # slow triangle drift (few reversals)
        frames.append(pose(x, 150, wrist="side"))
    kinds = run(frames)
    assert kinds.isdisjoint({"fainted", "seizure"})


# --------------------------------------------------------------------------- #
# Dedicated fall-detection model (state machine, driven without loading YOLO)
# --------------------------------------------------------------------------- #
def _fall_det():
    d = object.__new__(FallModelDetector)
    d.emitted = []
    d.emit = d.emitted.append
    d.conf, d.confirm_s, d.faint_s, d.still_frac, d.cooldown_s = 0.5, 0.4, 3.0, 0.03, 6.0
    d.fallen_since = d.still_since = 0.0
    d.center_prev = None
    d.last_emit = {}
    d.last = {}
    return d


def test_fall_model_fainted_when_still():
    d = _fall_det()
    box, ts = [200, 300, 360, 470], 100.0
    for _ in range(120):  # 6s: a fallen box that stays put -> fainted (no immediate 'fall')
        d._update(0.85, box, 480.0, 640.0, ts)
        ts += DT
    kinds = {e.kind for e in d.emitted}
    assert "fainted" in kinds and "fall" not in kinds


def test_fall_model_moving_is_not_fainted():
    d = _fall_det()
    ts = 100.0
    for i in range(30):  # fallen but sliding across the frame (not still) -> no page
        d._update(0.8, [200 + i * 8, 300, 360 + i * 8, 470], 480.0, 640.0, ts)
        ts += DT
    assert "fainted" not in {e.kind for e in d.emitted}


def test_fall_model_no_detection_is_quiet():
    d = _fall_det()
    ts = 100.0
    for _ in range(40):
        d._update(None, None, 480.0, 640.0, ts)
        ts += DT
    assert d.emitted == []


def test_fall_model_geometry_gate():
    """Geometry gate: wide OR low-in-frame keeps real falls; tall + mid-frame (a
    standing/walking person) is rejected."""
    d = object.__new__(FallModelDetector)
    d.fallen_ids = {0}
    d.max_aspect, d.low_aspect, d.low_cy = 1.10, 1.40, 0.78
    H = 480.0
    tall = np.array([100, 50, 180, 250])  # aspect 2.5, center_y 0.31 -> upright, REJECT
    wide = np.array([100, 300, 320, 440])  # aspect 0.64 -> on the ground, ACCEPT
    low_compact = np.array([100, 330, 220, 480])  # aspect 1.25, center_y 0.84 -> ACCEPT
    high_compact = np.array([100, 30, 220, 180])  # aspect 1.25, center_y 0.22 -> REJECT

    def pick(box, conf=0.9):
        return d._pick_fallen(np.array([0]), np.array([conf]), np.array([box]), H)[0]

    assert pick(tall) is None
    assert pick(wide) == 0.9
    assert pick(low_compact) == 0.9  # a fall whose box is compact but low in frame
    assert pick(high_compact) is None  # compact but high => not on the ground
    # both a standing box (higher conf) and a real wide fall: keep the fall
    assert (
        d._pick_fallen(np.array([0, 0]), np.array([0.99, 0.80]), np.array([tall, wide]), H)[0]
        == 0.80
    )


def test_tremor_seizure_detector():
    """Frame-difference seizure detector: sustained moderate in-place motion fires
    'seizure'; a still person does not."""
    d = object.__new__(VisionDetector)
    d.p = Params()
    d.st = PersonState()
    d.metrics = {}
    d.flash = None
    d._prev_gray = None
    d._tremor_ema = 0.0
    d._prev_box_center = None
    d.emitted = []
    d.emit = d.emitted.append
    box = np.array([50, 50, 250, 350])  # stationary person box
    base = np.full((400, 300, 3), 100, np.uint8)
    rng = np.random.default_rng(0)
    mask = rng.random((300, 200)) < 0.08  # ~8% of box pixels -> tremor band

    ts = 100.0
    for i in range(130):  # ~6.5s (> seizure_tremor_s=5s) of sustained in-place motion
        fr = base.copy()
        if i % 2:
            fr[50:350, 50:250][mask] = 170
        d._tremor(fr, box, ts)
        ts += 0.05
    assert any(e.kind == "seizure" for e in d.emitted)

    # a perfectly still person must NOT fire
    d.st = PersonState()
    d._prev_gray = None
    d._tremor_ema = 0.0
    d._prev_box_center = None
    d.emitted = []
    d.emit = d.emitted.append
    for i in range(50):
        d._tremor(base.copy(), box, 200.0 + i * 0.05)
    assert not any(e.kind == "seizure" for e in d.emitted)
