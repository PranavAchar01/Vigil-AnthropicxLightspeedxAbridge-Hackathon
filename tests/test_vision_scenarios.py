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


def test_fall_off_chair():
    frames = []
    frames += [pose(320, 150 + (i % 2) * 3) for i in range(20)]  # ~1s upright
    for i in range(11):  # ~0.5s rapid drop to the floor
        frames.append(pose(320, 150 + 26 * i))
    frames += hold(lambda: pose(320, 410), 0.6)  # landed low
    assert "fall" in run(frames)


def test_collapse_faint_in_place():
    # slump the head below the shoulders (small vertical drop) then go still ~3s
    frames = [pose(320, 150) for _ in range(10)]
    for i in range(10):  # head rotates down to below shoulder line
        frames.append(pose(320, 152, head_dy=-0.7 + 0.11 * i))
    frames += hold(lambda: pose(320, 152, head_dy=0.4), 3.4)  # head_below + motionless
    kinds = run(frames)
    assert "collapse" in kinds


def test_seizure_convulsion():
    frames = [pose(320, 150) for _ in range(6)]
    for i in range(44):  # ~2.2s of oscillatory shaking
        sgn = -1 if i % 2 else 1
        frames.append(pose(320 + 15 * sgn, 150 + 8 * sgn))
    assert "seizure" in run(frames)


def test_unresponsive_prolonged_stillness():
    frames = hold(lambda: pose(320, 150), 11.0)  # dead still 11s
    kinds = run(frames)
    assert "unresponsive" in kinds
    assert "motionless" in kinds  # the earlier soft signal fired too


def test_chest_clutch_distress_gesture():
    frames = hold(lambda: pose(320, 150, wrist="chest"), 2.2)  # hands at chest > gesture_s
    assert "chest_clutch" in run(frames)


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
    assert kinds.isdisjoint({"fall", "collapse", "seizure", "unresponsive"})


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


def test_fall_model_confirms_fall_then_collapse():
    d = _fall_det()
    box, ts = [200, 300, 360, 470], 100.0
    for _ in range(120):  # 6s: a fallen box that stays put
        d._update(0.85, box, 480.0, 640.0, ts)
        ts += DT
    kinds = {e.kind for e in d.emitted}
    assert "fall" in kinds and "collapse" in kinds


def test_fall_model_moving_is_fall_not_collapse():
    d = _fall_det()
    ts = 100.0
    for i in range(30):  # fallen but sliding across the frame (not still)
        d._update(0.8, [200 + i * 8, 300, 360 + i * 8, 470], 480.0, 640.0, ts)
        ts += DT
    kinds = {e.kind for e in d.emitted}
    assert "fall" in kinds and "collapse" not in kinds


def test_fall_model_no_detection_is_quiet():
    d = _fall_det()
    ts = 100.0
    for _ in range(40):
        d._update(None, None, 480.0, 640.0, ts)
        ts += DT
    assert d.emitted == []
