"""Real-time fall + soft-signal detection from one webcam. Skeletons only (COCO-17).

Adapted from the integration-research skeleton. The fall detector requires >=2 of
3 posture votes (torso horizontal, wide bbox, head-below-hip) PLUS a sudden
centroid drop PLUS a 2s time-on-ground confirm — that combination is what rejects
a fast sit as a fall. Ultralytics + OpenCV are imported lazily so the server can
boot in simulation mode without the heavy deps installed.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from vigil.events import Modality, PerceptionEvent

# COCO-17 keypoint indices
NOSE = 0
L_SH, R_SH = 5, 6
L_HIP, R_HIP = 11, 12
KP_CONF = 0.30

# tunable thresholds
TORSO_HORIZ_DEG = 60.0
AR_WIDE = 1.0
DROP_FRAC = 0.15
DROP_WINDOW_S = 0.6
GROUND_CONFIRM_S = 2.0
POSTURE_DEBOUNCE = 3
MOTIONLESS_EPS = 0.012
SLUMP_DEG = 32.0
SLUMP_S = 12.0
COOLDOWN_S = 8.0
# Faint/collapse: a person already down/slumped who goes still this long (research
# COLLAPSE_MOTIONLESS state). Fast on purpose — this is the "fainted" signal.
FAINT_SECONDS = 6.0

Sink = Callable[[PerceptionEvent], None]
# (track_id, posture, motion_level, moved) — per-frame live status for the voice agent
StatusSink = Callable[[int, str, str, bool], None]


def _mid(a, b):
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def torso_angle_deg(kp) -> float | None:
    """Angle (deg) of the shoulder->hip torso vector from image vertical.
    0 = upright, 90 = horizontal. Image y points DOWN."""
    sh, hip = _mid(kp[L_SH], kp[R_SH]), _mid(kp[L_HIP], kp[R_HIP])
    dx, dy = hip[0] - sh[0], hip[1] - sh[1]
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return None
    return math.degrees(math.atan2(abs(dx), abs(dy)))


@dataclass
class TrackState:
    state: str = "upright"  # upright | on_ground | fallen
    cy_hist: deque = field(default_factory=lambda: deque(maxlen=90))
    kp_prev: np.ndarray | None = None
    ground_frames: int = 0
    ground_since: float = 0.0
    still_since: float = 0.0
    slump_since: float = 0.0
    faint_since: float = 0.0
    last_emit: dict = field(default_factory=lambda: defaultdict(float))


class VisionDetector:
    def __init__(
        self,
        model_path: str = "yolo11n-pose.pt",
        device: str | None = None,
        emit: Sink = lambda e: None,
        emit_status: StatusSink = lambda *a: None,
    ) -> None:
        from ultralytics import YOLO  # lazy: heavy import

        self.model = YOLO(model_path)
        self.device = device
        self.emit = emit
        self.emit_status = emit_status
        self.tracks: dict[int, TrackState] = {}
        self._last_status_ts = 0.0

    def status_line(self) -> str:
        states = [t.state for t in self.tracks.values()]
        fallen = sum(s == "fallen" for s in states)
        return f"people:{len(states)}  fallen:{fallen}"

    def _fire(self, ts: float, kind, track_id: int, confidence: float) -> None:
        tk = self.tracks[track_id]
        if ts - tk.last_emit[kind] < COOLDOWN_S:
            return
        tk.last_emit[kind] = ts
        self.emit(
            PerceptionEvent(
                ts=ts,
                modality=Modality.VISION,
                kind=kind,
                confidence=round(float(confidence), 3),
                track_id=int(track_id),
            )
        )

    def _horizontal_posture(self, kp, box) -> tuple[bool, int]:
        x1, y1, x2, y2 = box
        w, h = (x2 - x1), (y2 - y1)
        ar = w / max(h, 1e-6)
        ang = torso_angle_deg(kp)
        hip_y = _mid(kp[L_HIP], kp[R_HIP])[1]
        head_below_hip = kp[NOSE][1] > hip_y + 0.02 * h
        votes = int((ang is not None and ang >= TORSO_HORIZ_DEG) + (ar >= AR_WIDE) + head_below_hip)
        return votes >= 2, votes

    def process(self, frame, ts: float):
        H = frame.shape[0]
        res = self.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            imgsz=480,
            conf=0.4,
            device=self.device,
            verbose=False,
        )[0]
        if res.boxes is None or res.boxes.id is None:
            return res
        ids = res.boxes.id.int().cpu().tolist()
        boxes = res.boxes.xyxy.cpu().numpy()
        kxy = res.keypoints.xy.cpu().numpy()
        kconf = res.keypoints.conf.cpu().numpy()

        for tid, box, kp, kc in zip(ids, boxes, kxy, kconf):
            tk = self.tracks.setdefault(tid, TrackState())
            visible = kc >= KP_CONF
            if not (visible[L_SH] and visible[R_SH] and visible[L_HIP] and visible[R_HIP]):
                continue

            cy = _mid(_mid(kp[L_SH], kp[R_SH]), _mid(kp[L_HIP], kp[R_HIP]))[1] / H
            tk.cy_hist.append((ts, cy))
            drop = 0.0
            for t0, cy0 in tk.cy_hist:
                if ts - t0 <= DROP_WINDOW_S:
                    drop = max(drop, cy - cy0)  # positive = downward in image
            sudden_drop = drop >= DROP_FRAC

            horiz, votes = self._horizontal_posture(kp, box)
            ang = torso_angle_deg(kp) or 0.0

            # fall state machine
            if tk.state == "upright":
                if sudden_drop and horiz:
                    tk.ground_frames += 1
                    if tk.ground_frames >= POSTURE_DEBOUNCE:
                        tk.state, tk.ground_since = "on_ground", ts
                else:
                    tk.ground_frames = 0
            elif tk.state == "on_ground":
                if horiz:
                    if ts - tk.ground_since >= GROUND_CONFIRM_S:
                        conf = 0.6 + 0.2 * (votes >= 3) + 0.2 * sudden_drop
                        self._fire(ts, "fall", tid, min(conf, 0.99))
                        tk.state = "fallen"
                else:
                    tk.state, tk.ground_frames = "upright", 0
            elif tk.state == "fallen":
                if not horiz:
                    tk.state, tk.ground_frames = "upright", 0

            # soft signal: motionlessness (disp also drives the live motion level)
            disp = 0.0
            if tk.kp_prev is not None:
                m = visible & (tk.kp_prev[:, 2] >= KP_CONF)
                if m.any():
                    disp = np.linalg.norm(kp[m] - tk.kp_prev[m, :2], axis=1).mean() / H
                    if disp < MOTIONLESS_EPS:
                        if tk.still_since == 0.0:
                            tk.still_since = ts
                        elif ts - tk.still_since >= _motionless_s():
                            self._fire(ts, "motionless", tid, 0.7)
                    else:
                        tk.still_since = 0.0
            tk.kp_prev = np.concatenate([kp, kc[:, None]], axis=1)

            # soft signal: posture slump (sustained lean while upright)
            slumping = False
            if tk.state == "upright" and SLUMP_DEG <= ang < TORSO_HORIZ_DEG:
                slumping = True
                if tk.slump_since == 0.0:
                    tk.slump_since = ts
                elif ts - tk.slump_since >= SLUMP_S:
                    self._fire(ts, "slump", tid, 0.6)
            else:
                tk.slump_since = 0.0

            # collapse / faint: down or slumped AND motionless for FAINT_SECONDS
            if (tk.state in ("on_ground", "fallen") or slumping) and disp < MOTIONLESS_EPS:
                if tk.faint_since == 0.0:
                    tk.faint_since = ts
                elif ts - tk.faint_since >= FAINT_SECONDS:
                    self._fire(ts, "collapse", tid, 0.85)
            else:
                tk.faint_since = 0.0

            # live status for the voice agent (throttled ~2/sec)
            if ts - self._last_status_ts >= 0.4:
                self._last_status_ts = ts
                posture = (
                    "on the floor"
                    if tk.state in ("on_ground", "fallen")
                    else "slumped"
                    if slumping
                    else "upright"
                )
                moved = disp >= MOTIONLESS_EPS
                motion = (
                    "still"
                    if disp < MOTIONLESS_EPS
                    else "slight"
                    if disp < 3 * MOTIONLESS_EPS
                    else "active"
                )
                self.emit_status(tid, posture, motion, moved)

        return res


def _motionless_s() -> float:
    from vigil.config import settings

    return settings.motionless_seconds


def run_vision(
    sink: Sink,
    frame_buffer,
    stop_event,
    source=0,
    device: str | None = None,
    model_path: str = "yolo11n-pose.pt",
    status_sink: StatusSink = lambda *a: None,
    identify_sink=lambda pid, name, score: None,
) -> None:
    """Blocking capture loop; run in a daemon thread. Encodes annotated frames
    into `frame_buffer`, publishes PerceptionEvents through `sink`, per-frame live
    status through `status_sink`, and (if a face gallery is enrolled) the identified
    patient through `identify_sink`."""
    import cv2

    det = VisionDetector(model_path=model_path, device=device, emit=sink, emit_status=status_sink)

    # optional on-device face recognition -> which patient is on camera
    recognizer, gallery, last_pid, frame_i = None, None, None, 0
    from vigil.config import settings

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
        print(f"[vision] could not open camera source={source}; vision disabled")
        return
    print("[vision] camera open; detecting falls (skeletons only)")
    try:
        while not stop_event.is_set() and cap.isOpened():
            ok, frame = cap.read()
            if not ok:
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

            res = det.process(frame, ts=time.time())
            annotated = res.plot() if res is not None else frame
            cv2.putText(
                annotated,
                f"VIGIL  {det.status_line()}",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 120),
                2,
            )
            ok, jpg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ok:
                frame_buffer.set(jpg.tobytes())
    finally:
        cap.release()
