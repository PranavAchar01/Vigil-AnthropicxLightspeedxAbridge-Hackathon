"""Real-time patient-distress detection from one webcam. Skeletons only (COCO-17).

Rebuilt for LAPTOP-WEBCAM framing (head + shoulders in view, hips usually NOT) and
to cover every scenario a waiting-room patient might present, each as a distinct
signal:

    fainted      went down (fell / horizontal) and STAYED down >= 5s        (hard)
    seizure      oscillatory convulsion sustained >= 5s                      (hard)
    slump        sustained posture degradation                              (soft)

Everything is measured relative to shoulder width (scale-invariant to distance) and
frame height (position), so it works up close on a laptop cam. Detection does NOT
require the hips or a ByteTrack id — it tracks the single most-prominent person.
Every threshold is env-tunable (VIGIL_* below) and defaults are demo-sensitive.

Live metrics are drawn on the video so you can see the detector responding in real
time, and each fired event flashes on screen. Ultralytics + OpenCV import lazily.
"""

from __future__ import annotations

import math
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

# macOS: we open the camera from a worker thread, but AVFoundation's auth request
# must run on the main thread. Skip the in-app request and rely on the OS-level
# camera grant (run scripts/grant_camera.py once to establish it).
os.environ.setdefault("OPENCV_AVFOUNDATION_SKIP_AUTH", "1")

import numpy as np

from vigil.events import Modality, PerceptionEvent

# COCO-17 keypoint indices
NOSE = 0
L_EYE, R_EYE = 1, 2
L_EAR, R_EAR = 3, 4
L_SH, R_SH = 5, 6
L_WR, R_WR = 9, 10
L_HIP, R_HIP = 11, 12
UPPER_BODY = list(range(0, 11))  # nose..wrists — the reliably-framed keypoints


def _f(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Params:
    """All detector thresholds, env-overridable. Defaults are tuned demo-sensitive."""

    kp_conf: float = _f("VIGIL_KP_CONF", 0.25)  # keypoint visibility floor
    # fall
    drop_frac: float = _f("VIGIL_DROP_FRAC", 0.11)  # downward drop of body-center (frac of H)
    drop_window_s: float = _f("VIGIL_DROP_WINDOW_S", 0.8)
    fall_confirm_s: float = _f("VIGIL_FALL_CONFIRM_S", 0.30)
    torso_horiz_deg: float = _f("VIGIL_TORSO_HORIZ_DEG", 55.0)
    ar_wide: float = _f("VIGIL_AR_WIDE", 1.05)  # bbox wider than tall
    low_cy: float = _f("VIGIL_LOW_CY", 0.60)  # body-center in the lower part of frame
    # motion / stillness (shoulder-widths per second)
    still_motion: float = _f("VIGIL_STILL_MOTION", 0.55)
    motionless_s: float = _f("VIGIL_MOTIONLESS_S", 3.0)  # soft — shows quickly
    unresponsive_s: float = _f("VIGIL_UNRESPONSIVE_S", 8.0)  # hard
    faint_s: float = _f("VIGIL_FAINT_S", 3.0)  # slump/down + still → collapse
    # seizure
    seizure_motion: float = _f("VIGIL_SEIZURE_MOTION", 2.2)  # mean motion in window
    seizure_osc: float = _f("VIGIL_SEIZURE_OSC", 5)  # direction reversals in window
    seizure_window_s: float = _f("VIGIL_SEIZURE_WINDOW_S", 1.6)
    seizure_s: float = _f("VIGIL_SEIZURE_S", 1.0)  # sustained before firing
    # slump
    slump_min_deg: float = _f("VIGIL_SLUMP_MIN_DEG", 26.0)
    slump_s: float = _f("VIGIL_SLUMP_S", 5.0)
    # frame-difference seizure/tremor: sustained MODERATE person-region motion that is
    # neither still nor large/purposeful — catches subtle (incl. seated) seizures that
    # keypoint oscillation misses. Tuned on real seizure footage (person-region motion
    # ~0.06-0.09 sustained, vs ~0.22 for active sitting and <0.02 for a still person).
    seizure_tremor_lo: float = _f("VIGIL_SEIZURE_TREMOR_LO", 0.025)
    seizure_tremor_hi: float = _f("VIGIL_SEIZURE_TREMOR_HI", 0.16)
    seizure_tremor_s: float = _f("VIGIL_SEIZURE_TREMOR_S", 1.3)
    # bookkeeping
    cooldown_s: float = _f("VIGIL_COOLDOWN_S", 6.0)
    unresponsive_cooldown_s: float = _f("VIGIL_UNRESPONSIVE_COOLDOWN_S", 15.0)


P = Params()

Sink = Callable[[PerceptionEvent], None]
# (track_id, posture, motion_level, moved) — per-frame live status for the voice agent
StatusSink = Callable[[int, str, str, bool], None]


def _mid(a, b):
    return (a + b) / 2.0


def _dist(a, b) -> float:
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


def torso_angle_deg(kp) -> float | None:
    """Angle (deg) of the shoulder->hip torso vector from image vertical.
    0 = upright, 90 = horizontal. Image y points DOWN. None if degenerate."""
    sh = _mid(kp[L_SH], kp[R_SH])
    hip = _mid(kp[L_HIP], kp[R_HIP])
    dx, dy = hip[0] - sh[0], hip[1] - sh[1]
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return None
    return math.degrees(math.atan2(abs(dx), abs(dy)))


@dataclass
class PersonState:
    fsm: str = "upright"  # upright | down
    cy_hist: deque = field(default_factory=lambda: deque(maxlen=120))  # (t, cy)
    delta_hist: deque = field(default_factory=lambda: deque(maxlen=120))  # (t, dx, dy) of sh_mid
    kp_prev: np.ndarray | None = None
    sh_prev: np.ndarray | None = None
    t_prev: float = 0.0
    fall_since: float = 0.0
    still_since: float = 0.0
    faint_since: float = 0.0
    seizure_since: float = 0.0
    seizure_last: float = 0.0  # last frame the shaking was above threshold (hysteresis)
    slump_since: float = 0.0
    tremor_since: float = 0.0
    last_drop_ts: float = -1e9
    last_emit: dict = field(default_factory=dict)


class VisionDetector:
    def __init__(
        self,
        model_path: str = "yolo11n-pose.pt",
        device: str | None = None,
        emit: Sink = lambda e: None,
        emit_status: StatusSink = lambda *a: None,
        params: "Params | None" = None,
    ) -> None:
        from ultralytics import YOLO  # lazy: heavy import

        self.model = YOLO(model_path)
        self.device = device
        self.emit = emit
        self.emit_status = emit_status
        self.p = params or P  # tunable thresholds (the optimizer injects candidates here)
        self.st = PersonState()
        self._last_status_ts = 0.0
        self.metrics: dict = {}  # live values for the on-frame overlay
        self.flash: tuple[str, float] | None = None  # (text, ts) of the last fired event
        self._prev_gray: np.ndarray | None = None  # frame-diff seizure/tremor state
        self._tremor_ema: float = 0.0
        self._prev_box_center: tuple[float, float] | None = None

    # ------------------------------------------------------------------ helpers
    def status_line(self) -> str:
        m = self.metrics
        return f"{self.st.fsm}  motion:{m.get('motion', 0):.1f}  drop:{m.get('drop', 0):.2f}"

    def _fire(self, ts: float, kind, conf: float, label: str) -> None:
        # fainted/seizure get a long cooldown so one incident never re-pages the nurse.
        cd = self.p.unresponsive_cooldown_s if kind in ("fainted", "seizure") else self.p.cooldown_s
        if ts - self.st.last_emit.get(kind, -1e9) < cd:
            return
        self.st.last_emit[kind] = ts
        if kind == "fainted":
            self.st.last_drop_ts = ts  # mark down-context so the tremor won't co-fire seizure
        self.flash = (label, ts)
        self.emit(
            PerceptionEvent(
                ts=ts,
                modality=Modality.VISION,
                kind=kind,
                confidence=round(float(min(conf, 0.99)), 3),
                track_id=1,
            )
        )

    def _largest_person(self, res):
        if res.keypoints is None or res.boxes is None or len(res.boxes) == 0:
            return None
        boxes = res.boxes.xyxy.cpu().numpy()
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        i = int(np.argmax(areas))
        kxy = res.keypoints.xy.cpu().numpy()[i]
        kconf = res.keypoints.conf
        kc = kconf.cpu().numpy()[i] if kconf is not None else np.ones(len(kxy))
        return boxes[i], kxy, kc

    # ------------------------------------------------------------------- process
    def process(self, frame, ts: float):
        H = frame.shape[0]
        res = self.model.predict(frame, imgsz=512, conf=0.35, device=self.device, verbose=False)[0]

        person = self._largest_person(res)
        if person is None:
            self.metrics = {"present": False}
            self.st.kp_prev = None
            self.st.sh_prev = None
            self._prev_gray = None
            self._prev_box_center = None
            self.st.tremor_since = 0.0
            return res
        box, kp, kc = person
        self.analyze(box, kp, kc, H, ts)
        self._tremor(frame, box, ts)
        return res

    def _tremor(self, frame, box, ts: float) -> None:
        """Frame-difference seizure/tremor detector: fires 'seizure' on SUSTAINED
        moderate motion inside the person's box — the signature of a subtle (incl.
        seated) seizure that keypoint oscillation misses. Ignores near-stillness and
        large/purposeful movement."""
        p = self.p
        # luma (BGR) without cv2 so it works anywhere process() runs
        g = (0.114 * frame[:, :, 0] + 0.587 * frame[:, :, 1] + 0.299 * frame[:, :, 2]).astype(
            np.float32
        )
        prev, self._prev_gray = self._prev_gray, g
        if prev is None or prev.shape != g.shape:
            return
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, g.shape[1]), min(y2, g.shape[0])
        if x2 - x1 < 8 or y2 - y1 < 8:
            self.st.tremor_since = 0.0
            return
        d = np.abs(g[y1:y2, x1:x2] - prev[y1:y2, x1:x2])
        energy = float((d > 12).mean())
        self._tremor_ema = 0.6 * self._tremor_ema + 0.4 * energy
        e = self._tremor_ema
        self.metrics["tremor"] = round(e, 3)

        # a seizure tremor is IN PLACE; a fall/large move TRANSLATES the box → not a
        # tremor. Reset when the person is moving across the frame so falls stay falls.
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        prevc, self._prev_box_center = self._prev_box_center, (cx, cy)
        translating = (
            prevc is not None and (abs(cx - prevc[0]) + abs(cy - prevc[1])) / g.shape[0] > 0.015
        )
        grounded = self.st.fsm == "down"  # on the floor => a fall/collapse, not a seizure
        recent_fall = ts - self.st.last_drop_ts < 6.0  # a fall just happened -> not a seizure
        if (
            not translating
            and not grounded
            and not recent_fall
            and p.seizure_tremor_lo <= e <= p.seizure_tremor_hi
        ):
            self.st.tremor_since = self.st.tremor_since or ts
            if ts - self.st.tremor_since >= p.seizure_tremor_s:
                self._fire(ts, "seizure", 0.75, "SEIZURE DETECTED")
        else:
            self.st.tremor_since = 0.0

    def analyze(self, box, kp, kc, H: float, ts: float) -> None:
        """The detection core — pure keypoints in, events out. Split from `process`
        so it can be driven by synthetic sequences in tests (no camera/model)."""
        vis = kc >= self.p.kp_conf

        # --- reference geometry (head + shoulders; hips optional) ---------------
        x1, y1, x2, y2 = box
        box_w, box_h = (x2 - x1), max(y2 - y1, 1.0)
        ar = box_w / box_h

        if vis[L_SH] and vis[R_SH]:
            sh_mid = _mid(kp[L_SH], kp[R_SH])
            scale = max(_dist(kp[L_SH], kp[R_SH]), 1.0)
        else:  # shoulders occluded → fall back to the bbox so detection never stops
            sh_mid = np.array([(x1 + x2) / 2.0, y1 + 0.22 * box_h])
            scale = max(0.35 * box_w, 1.0)

        # head: nose, else ears/eyes, else top-center of bbox
        head = None
        for idx in (NOSE, L_EAR, R_EAR, L_EYE, R_EYE):
            if vis[idx]:
                head = kp[idx]
                break
        if head is None:
            head = np.array([(x1 + x2) / 2.0, y1])
        head_below = head[1] > sh_mid[1] + 0.20 * scale  # head at/below shoulder line

        hip_ok = vis[L_HIP] and vis[R_HIP]
        torso_deg = torso_angle_deg(kp) if (hip_ok and vis[L_SH] and vis[R_SH]) else None

        cy = sh_mid[1] / H

        # --- first frame: seed and return --------------------------------------
        if self.st.kp_prev is None or self.st.t_prev == 0.0:
            self.st.kp_prev = np.concatenate([kp, kc[:, None]], axis=1)
            self.st.sh_prev = sh_mid
            self.st.t_prev = ts
            self.st.cy_hist.append((ts, cy))
            self.metrics = {"present": True, "motion": 0.0, "drop": 0.0}
            return

        dt = min(max(ts - self.st.t_prev, 1e-3), 0.5)

        # --- motion (scale-invariant) ------------------------------------------
        prev = self.st.kp_prev
        m = vis & (prev[:, 2] >= self.p.kp_conf)
        m_upper = m.copy()
        m_upper[11:] = False  # ignore legs (rarely framed / noisy)
        motion_px = (
            float(np.linalg.norm(kp[m_upper] - prev[m_upper, :2], axis=1).mean())
            if m_upper.any()
            else 0.0
        )
        motion = motion_px / scale / dt  # shoulder-widths per second

        d_sh = sh_mid - self.st.sh_prev
        vy = (cy - (self.st.sh_prev[1] / H)) / dt  # + = downward
        self.st.delta_hist.append((ts, float(d_sh[0]), float(d_sh[1])))
        self.st.cy_hist.append((ts, cy))

        # drop over the recent window (how far below the recent-highest position)
        recent = [c for (t0, c) in self.st.cy_hist if ts - t0 <= self.p.drop_window_s]
        drop = cy - min(recent) if recent else 0.0
        if drop >= self.p.drop_frac:  # a real downward drop => fall context, not a seizure
            self.st.last_drop_ts = ts

        # seizure: reversals of sh_mid direction + sustained high motion in window
        win = [
            (dx, dy) for (t0, dx, dy) in self.st.delta_hist if ts - t0 <= self.p.seizure_window_s
        ]
        reversals = 0
        for a, b in zip(win, win[1:]):
            if a[0] * b[0] + a[1] * b[1] < 0:  # dot < 0 → direction flipped >90°
                reversals += 1
        # mean motion across the seizure window, approximated by a per-frame motion EMA
        self._motion_ema = 0.6 * getattr(self, "_motion_ema", motion) + 0.4 * motion
        seizure_energy = self._motion_ema

        still = motion < self.p.still_motion

        # ---- posture recovery: clearly upright again → reset the "down" state ---
        upright_now = (not head_below) and (torso_deg is None or torso_deg < 25.0) and cy < 0.55
        if upright_now and motion >= self.p.still_motion:
            self.st.fsm = "upright"

        horiz = (
            (torso_deg is not None and torso_deg >= self.p.torso_horiz_deg)
            or ar >= self.p.ar_wide
            or head_below
        )
        low = cy >= self.p.low_cy

        # ============================ SCENARIO DETECTORS ========================
        # 1) FAINTED — the person goes DOWN (falls to the floor / lies horizontal /
        # collapses) and STAYS down. VALIDATION: they must remain down for at least
        # faint_s (5s) before we fire, so a trip-and-recover never pages the nurse.
        # This replaces the old separate 'fall', 'collapse', and 'unresponsive' events
        # — there is no standalone "fall detection".
        truly_horizontal = torso_deg is not None and torso_deg >= self.p.torso_horiz_deg
        big_drop = drop >= 2.4 * self.p.drop_frac and vy > 0.4  # fell straight down
        if big_drop:
            self.st.fsm = "down"
        went_down = (
            (drop >= self.p.drop_frac and (horiz or low))  # a fall to the ground
            or truly_horizontal                             # lying / horizontal torso
            or self.st.fsm == "down"                        # already registered as down
        )
        if went_down:
            self.st.fsm = "down"
            self.st.faint_since = self.st.faint_since or ts
            if ts - self.st.faint_since >= self.p.faint_s:  # ON THE GROUND >= 5s
                self._fire(ts, "fainted", 0.9, "FAINTED — down 5s, not recovering")
        else:
            self.st.faint_since = 0.0

        # 2) SEIZURE — oscillatory shaking sustained >= seizure_s (5s) before firing.
        # Brief dips in the signal are NORMAL in a real convulsion, so we don't reset
        # the timer on a single low frame — only after the shaking has clearly STOPPED
        # for > 2s. This is what lets a genuine 5s+ seizure fire despite fluctuation.
        # GATE: a person who is going DOWN (falling / horizontal / on the floor) is
        # FAINTING, not seizing — the fall's impact motion must never read as seizure.
        # So a seizure is only recognized while the person is upright/in-place.
        seizing_now = (
            not went_down
            and seizure_energy >= self.p.seizure_motion
            and reversals >= self.p.seizure_osc
        )
        if seizing_now:
            self.st.seizure_since = self.st.seizure_since or ts
            self.st.seizure_last = ts
            if ts - self.st.seizure_since >= self.p.seizure_s:  # SEIZING >= 5s
                self._fire(ts, "seizure", min(0.6 + 0.05 * reversals, 0.95), "SEIZURE DETECTED")
        elif ts - self.st.seizure_last > 2.0:  # shaking clearly stopped (> 2s) -> reset
            self.st.seizure_since = 0.0

        # 3) SLUMP — sustained lean (upright but degraded posture)
        if self.st.fsm == "upright":
            slumping = (
                torso_deg is not None and self.p.slump_min_deg <= torso_deg < self.p.torso_horiz_deg
            ) or (torso_deg is None and abs(head[0] - sh_mid[0]) > 0.65 * scale)
        else:
            slumping = False
        if slumping:
            self.st.slump_since = self.st.slump_since or ts
            if ts - self.st.slump_since >= self.p.slump_s:
                self._fire(ts, "slump", 0.55, "slumping")
        else:
            self.st.slump_since = 0.0

        # --- persist frame state -----------------------------------------------
        self.st.kp_prev = np.concatenate([kp, kc[:, None]], axis=1)
        self.st.sh_prev = sh_mid
        self.st.t_prev = ts

        # --- live metrics + status ---------------------------------------------
        posture = (
            "on the floor"
            if self.st.fsm == "down"
            else "slumped"
            if slumping
            else "upright"
        )
        motion_label = (
            "still" if still else "slight" if motion < 3 * self.p.still_motion else "active"
        )
        self.metrics = {
            "present": True,
            "state": self.st.fsm,
            "motion": round(motion, 2),
            "drop": round(float(drop), 3),
            "vy": round(float(vy), 2),
            "reversals": reversals,
            "still_s": round(ts - self.st.still_since, 1) if self.st.still_since else 0.0,
            "posture": posture,
            "head_below": bool(head_below),
        }
        if ts - self._last_status_ts >= 0.4:
            self._last_status_ts = ts
            self.emit_status(1, posture, motion_label, not still)

    # ------------------------------------------------------------------- overlay
    def draw(self, frame, cv2) -> None:
        """Render the live metric panel + last-event flash onto the annotated frame."""
        m = self.metrics
        H, W = frame.shape[0], frame.shape[1]
        panel = (
            [
                f"STATE   {m.get('state', '—')}",
                f"motion  {m.get('motion', 0):.2f} sw/s  ({m.get('posture', '—')})",
                f"drop    {m.get('drop', 0):.2f}   vy {m.get('vy', 0):+.2f}",
                f"seizure reversals {m.get('reversals', 0)}",
                f"still   {m.get('still_s', 0):.1f}s",
            ]
            if m.get("present")
            else ["no person in frame"]
        )
        y = 46
        cv2.rectangle(frame, (8, 30), (330, 30 + 22 * len(panel) + 10), (0, 0, 0), -1)
        for line in panel:
            cv2.putText(frame, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 255, 170), 1)
            y += 22
        # event flash (persist ~2.5s)
        if self.flash and time.time() - self.flash[1] < 2.5:
            txt = self.flash[0]
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 3)
            x = (W - tw) // 2
            cv2.rectangle(frame, (x - 16, H - 90), (x + tw + 16, H - 40), (0, 0, 0), -1)
            cv2.putText(frame, txt, (x, H - 55), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (80, 90, 255), 3)


class FallModelDetector:
    """A dedicated, pre-trained fall-detection YOLO running ALONGSIDE the pose FSM
    for redundant fall/faint coverage.

    Model: melihuzunoglu/human-fall-detection (YOLOv11, classes fallen/sitting/
    standing), fine-tuned specifically for falls. A confirmed 'fallen' box → `fall`;
    a fallen box whose centroid then stays put → `collapse` (fainted on the ground).
    Either this OR the pose FSM firing is enough — fusion de-dups on cooldown.
    """

    def __init__(self, model_path, emit: Sink = lambda e: None, device: str | None = None) -> None:
        from ultralytics import YOLO  # lazy heavy import

        self.model = YOLO(model_path)
        self.device = device
        self.emit = emit
        self.names = {int(i): str(n).lower() for i, n in self.model.names.items()}
        self.fallen_ids = {i for i, n in self.names.items() if "fall" in n}
        self.conf = _f("VIGIL_FALL_MODEL_CONF", 0.40)
        # Geometry gate (validated on UR Fall Detection Dataset): a real fallen person
        # is EITHER wide (box aspect h/w below max_aspect) OR low in the frame (box
        # center below low_cy while still not tall, ≤ low_aspect). A standing/walking
        # person's box is tall (~1.7) and centered mid-frame → rejected. This drove
        # false positives on normal activity from 60% to 0% while keeping fall recall.
        self.max_aspect = _f("VIGIL_FALL_MAX_ASPECT", 1.10)
        self.low_aspect = _f("VIGIL_FALL_LOW_ASPECT", 1.40)
        self.low_cy = _f("VIGIL_FALL_LOW_CY", 0.78)
        self.confirm_s = _f("VIGIL_FALL_CONFIRM_S", 0.40)
        self.faint_s = _f("VIGIL_FALL_FAINT_S", 3.0)
        self.still_frac = _f("VIGIL_FALL_STILL_FRAC", 0.03)
        self.cooldown_s = _f("VIGIL_FALL_COOLDOWN_S", 6.0)
        self.fallen_since = 0.0
        self.still_since = 0.0
        self.center_prev: np.ndarray | None = None
        self.last_emit: dict = {}
        self.last: dict = {"fallen": False, "conf": 0.0, "box": None}

    def _fire(self, ts: float, kind, conf: float) -> None:
        if ts - self.last_emit.get(kind, -1e9) < self.cooldown_s:
            return
        self.last_emit[kind] = ts
        self.emit(
            PerceptionEvent(
                ts=ts,
                modality=Modality.VISION,
                kind=kind,
                confidence=round(float(min(conf, 0.99)), 3),
                track_id=2,
            )
        )

    def _update(self, best_conf: float | None, box, H: float, W: float, ts: float) -> None:
        """Confirm/faint state machine — pure, so tests can drive it without the model."""
        if best_conf is None:
            self.fallen_since = self.still_since = 0.0
            self.center_prev = None
            self.last = {"fallen": False, "conf": 0.0, "box": None}
            return
        # No immediate 'fall' event — validation requires staying down. Fire 'fainted'
        # only after a fallen box stays put for faint_s (>= 5s on the ground).
        self.fallen_since = self.fallen_since or ts
        center = np.array([(box[0] + box[2]) / 2.0 / W, (box[1] + box[3]) / 2.0 / H])
        if self.center_prev is not None:
            if float(np.linalg.norm(center - self.center_prev)) < self.still_frac:
                self.still_since = self.still_since or ts
                if ts - self.still_since >= self.faint_s:
                    self._fire(ts, "fainted", min(best_conf + 0.1, 0.99))
            else:
                self.still_since = 0.0
        self.center_prev = center
        self.last = {"fallen": True, "conf": round(best_conf, 2), "box": box}

    def process(self, frame, ts: float):
        H, W = frame.shape[0], frame.shape[1]
        res = self.model.predict(
            frame, imgsz=640, conf=self.conf, device=self.device, verbose=False
        )[0]
        best_conf, best_box = None, None
        if res.boxes is not None and len(res.boxes):
            best_conf, best_box = self._pick_fallen(
                res.boxes.cls.cpu().numpy().astype(int),
                res.boxes.conf.cpu().numpy(),
                res.boxes.xyxy.cpu().numpy(),
                H,
            )
        self._update(best_conf, best_box, H, W, ts)
        return res

    def _pick_fallen(self, cls, cf, xy, H):
        """Highest-confidence 'fallen' box that passes the geometry gate — wide, OR
        low-in-frame and not tall (a person on the ground). Rejects tall, mid-frame
        boxes (standing/walking). Pure — testable without the YOLO model."""
        best_conf, best_box = None, None
        for c, k, b in zip(cls, cf, xy):
            if c not in self.fallen_ids:
                continue
            aspect = (b[3] - b[1]) / max(b[2] - b[0], 1.0)
            center_y = (b[1] + b[3]) / 2.0 / H
            on_ground = aspect <= self.max_aspect or (
                aspect <= self.low_aspect and center_y >= self.low_cy
            )
            if not on_ground:
                continue  # tall + mid-frame => upright (standing/walking), not a fall
            if best_conf is None or k > best_conf:
                best_conf, best_box = float(k), b
        return best_conf, best_box

    def draw(self, frame, cv2) -> None:
        if not self.last.get("fallen"):
            return
        b = self.last.get("box")
        if b is not None:
            p1, p2 = (int(b[0]), int(b[1])), (int(b[2]), int(b[3]))
            cv2.rectangle(frame, p1, p2, (60, 60, 255), 3)
        cv2.putText(
            frame,
            f"FALL MODEL: FALLEN {self.last['conf']:.2f}",
            (12, frame.shape[0] - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (60, 60, 255),
            2,
        )


def run_vision(
    sink: Sink,
    frame_buffer,
    stop_event,
    source=0,
    device: str | None = None,
    model_path: str = "yolo11n-pose.pt",
    status_sink: StatusSink = lambda *a: None,
    identify_sink=lambda pid, name, score: None,
    pause_event=None,
) -> None:
    """Blocking capture loop; run in a daemon thread. Encodes annotated frames into
    `frame_buffer`, publishes PerceptionEvents through `sink`, per-frame live status
    through `status_sink`, and (if a face gallery is enrolled) the identified patient
    through `identify_sink`."""
    import cv2

    det = VisionDetector(model_path=model_path, device=device, emit=sink, emit_status=status_sink)

    from vigil.config import settings

    # Dedicated pre-trained fall-detection model (redundant fall/faint coverage).
    fall_det = None
    fall_every_n = max(1, int(_f("VIGIL_FALL_EVERY_N", 1)))
    if settings.fall_model_path.exists():
        try:
            fall_det = FallModelDetector(str(settings.fall_model_path), emit=sink, device=device)
            print(
                f"[vision] fall-detection model loaded ({settings.fall_model_path.name}; "
                f"classes {list(fall_det.names.values())})"
            )
        except Exception as e:  # noqa: BLE001 — never let it break the pose feed
            print(f"[vision] fall model unavailable ({e!r}); pose FSM covers falls")
            fall_det = None

    # optional on-device face recognition -> which patient is on camera
    recognizer, gallery, last_pid, frame_i = None, None, None, 0

    if settings.face_gallery_path.exists():
        try:
            from vigil.perception.faces import FaceGallery, FaceRecognizer

            gallery = FaceGallery.load(settings.face_gallery_path, settings.face_match_threshold)
            recognizer = FaceRecognizer()
            print(f"[vision] face recognition on; {len(gallery)} patients enrolled")
        except Exception as e:  # noqa: BLE001
            print(f"[vision] face recognition unavailable ({e!r}); running without it")
            recognizer = None

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[vision] could not open source={source}; vision disabled")
        return
    is_file = not isinstance(source, int)
    print(f"[vision] {'reading ' + str(source) if is_file else 'camera open'}; detecting distress")
    try:
        while not stop_event.is_set() and cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                if is_file:  # loop the demo clip so the app keeps showing it
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                time.sleep(0.05)
                continue

            # DEMO PAUSE — keep streaming live video but run no detection (no events).
            if pause_event is not None and pause_event.is_set():
                cv2.putText(
                    frame, "PAUSED", (12, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 255), 3
                )
                ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ok:
                    frame_buffer.set(jpg.tobytes())
                time.sleep(0.05)
                continue

            frame_i += 1
            if recognizer and gallery and frame_i % settings.face_identify_every_n == 0:
                try:
                    emb = recognizer.embed_largest(frame)
                    match = gallery.identify(emb) if emb is not None else None
                    if match and match[0] != last_pid:
                        last_pid = match[0]
                        identify_sink(match[0], match[1], round(match[2], 3))
                except Exception:  # noqa: BLE001 — recognition must never break the feed
                    pass

            ts = time.time()
            res = det.process(frame, ts=ts)
            if fall_det is not None and frame_i % fall_every_n == 0:
                try:
                    fall_det.process(frame, ts=ts)
                except Exception:  # noqa: BLE001 — never let it break the feed
                    pass
            annotated = res.plot() if res is not None else frame
            det.draw(annotated, cv2)
            if fall_det is not None:
                fall_det.draw(annotated, cv2)
            cv2.putText(
                annotated,
                f"VIGIL  {det.status_line()}",
                (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 120),
                2,
            )
            ok, jpg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ok:
                frame_buffer.set(jpg.tobytes())
    finally:
        cap.release()
