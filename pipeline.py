"""
Impact-Net computer vision engine.

Streamlit-/FastAPI-agnostic crash triage pipeline: YOLOv8 vehicle tracking,
dense optical flow physics, Kinetic Severity Index (KSI), H.264 annotation,
and structured 911 dispatch telemetry.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import cv2
import imageio.v2 as imageio
import numpy as np
import pandas as pd
from ultralytics import YOLO

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VEHICLE_CLASS_IDS: frozenset[int] = frozenset(
    {2, 3, 5, 6, 7, 8}
)  # car, motorcycle, bus, train, truck, boat (wreckage often → boat)
VEHICLE_CLASS_NAMES: dict[int, str] = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    6: "train",
    7: "truck",
    8: "boat",
}

MAX_PROCESS_WIDTH = 640
DEFAULT_MODEL = "yolov8n.pt"
OUTPUT_DIR_NAME = "outputs"

# --- Performance (CPU-friendly triage without dropping crash recall) ---
TARGET_ANALYSIS_FPS = 12.0  # run YOLO/physics at ~12 Hz; fill other frames for playback
YOLO_IMGSZ = 416  # detector input size (speed/accuracy tradeoff)
MOTION_WIDTH = 160  # tiny grayscale for cheap scene-motion score
SOFT_DETECT_EVERY = 8  # reserved
PROGRESS_EVERY = 8  # UI progress update cadence
DEBRIS_SCORE_THRESH = 0.22  # elevated dust/smoke plume (0–1)
CRITICAL_PLUME_THRESH = 0.62  # high-energy impact plume → allow Level 4–5
MODERATE_AFTERMATH_MAX_DV = 30.0  # vehicle-damage-only cap (~KSI ~2)
IMPACT_BURST_MOTION = 1.15  # cheap-motion spike vs rolling baseline

# Optical-flow / physics heuristics (pixel-space → km/h-equivalent impact units).
# Keep this conservative: dense-flow noise on asphalt otherwise saturates Δv.
PX_PER_FRAME_TO_KMH = 5.5
SUDDEN_DECEL_KMH = 12.0  # Δv spike threshold for anomaly flagging
PROXIMITY_IOU_COLLISION = 0.02
PROXIMITY_CENTER_FRAC = 0.28  # centers within this fraction of frame diagonal
SCENE_FLOW_BASELINE_FRAMES = 15
AFTERMATH_MIN_AREA_FRAC = 0.08  # wreckage covering ≥8% of frame
IMPACT_ANGLE_FLOOR_DEG = 55.0  # keep sin(θ) from collapsing noisy headings
MAX_SPEED_KMH = 64.0  # clamp impossible track teleports from ID switches
MAX_DELTA_V_KMH = 55.0  # KSI saturates near 5 around Δv≈47 with sin≈1
MAX_CENTER_JUMP_PX = 48.0  # treat larger center jumps as re-acquisitions, not motion
VELOCITY_EMA_ALPHA = 0.4  # smooth box velocities to suppress flow flicker
SCENE_BURST_GAIN = 1.4  # scene flow → Δv gain (kept mild for POV driving)

ProgressCallback = Callable[[float, str], None]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class TrackPhysics:
    """Per-track velocity history for acceleration / impact analysis."""

    track_id: int
    class_name: str
    velocities: list[tuple[float, float]] = field(default_factory=list)  # (vx, vy) km/h
    magnitudes: list[float] = field(default_factory=list)
    accelerations: list[float] = field(default_factory=list)  # Δ|v| per frame (signed)
    delta_vs: list[float] = field(default_factory=list)  # ||v_t − v_{t−1}|| km/h
    boxes: list[tuple[int, int, int, int]] = field(default_factory=list)
    frames: list[int] = field(default_factory=list)

    def push(
        self,
        frame_idx: int,
        box: tuple[int, int, int, int],
        vx: float,
        vy: float,
    ) -> float:
        # EMA smooth to suppress optical-flow flicker on textured asphalt
        if self.velocities:
            prev_vx, prev_vy = self.velocities[-1]
            a = VELOCITY_EMA_ALPHA
            vx = a * vx + (1.0 - a) * prev_vx
            vy = a * vy + (1.0 - a) * prev_vy
        mag = float(math.hypot(vx, vy))
        accel = 0.0
        delta_v = 0.0
        if self.velocities:
            prev_vx, prev_vy = self.velocities[-1]
            accel = mag - self.magnitudes[-1]
            delta_v = float(math.hypot(vx - prev_vx, vy - prev_vy))
            # Cold-start: first real motion after a zero/unknown velocity is acquisition, not impact
            if self.magnitudes[-1] < 1.0 and mag > 6.0:
                delta_v = 0.0
            # Ignore one-frame spikes that immediately settle (tracker jitter)
            if len(self.delta_vs) >= 1 and self.delta_vs[-1] < 4.0 and delta_v > 18.0:
                delta_v *= 0.35
        self.frames.append(frame_idx)
        self.boxes.append(box)
        self.velocities.append((vx, vy))
        self.magnitudes.append(mag)
        self.accelerations.append(accel)
        self.delta_vs.append(clamp_delta_v(delta_v))
        return self.delta_vs[-1]


@dataclass
class CollisionEvent:
    frame_idx: int
    timestamp_sec: float
    delta_v: float
    impact_angle_deg: float
    ksi: float
    severity_level: int
    track_ids: tuple[int, ...]
    confidence: float


# ---------------------------------------------------------------------------
# Geometry & physics helpers
# ---------------------------------------------------------------------------


def _clamp_even(value: int) -> int:
    """H.264 yuv420p requires even width/height."""
    return value if value % 2 == 0 else value - 1


def scale_to_max_width(frame: np.ndarray, max_width: int = MAX_PROCESS_WIDTH) -> tuple[np.ndarray, float]:
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame, 1.0
    scale = max_width / float(w)
    new_w = _clamp_even(int(round(w * scale)))
    new_h = _clamp_even(int(round(h * scale)))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def box_center(box: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def box_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def vector_angle_deg(v1: tuple[float, float], v2: tuple[float, float]) -> float:
    """Acute-to-obtuse angle (0–180°) between two 2D vectors."""
    n1 = math.hypot(v1[0], v1[1])
    n2 = math.hypot(v2[0], v2[1])
    if n1 < 1e-6 or n2 < 1e-6:
        return 90.0  # unknown → assume orthogonal contribution
    cos_t = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
    return math.degrees(math.acos(cos_t))


def kinetic_severity_index(delta_v: float, theta_impact_rad: float) -> float:
    """
    KSI = min(5.0, 0.5 * (Δv / 15)^2 * sin(θ_impact))

    Δv in km/h-equivalent impact units. θ_impact in radians.
    """
    sin_term = max(abs(math.sin(theta_impact_rad)), math.sin(math.radians(IMPACT_ANGLE_FLOOR_DEG)))
    raw = 0.5 * ((max(0.0, delta_v) / 15.0) ** 2) * sin_term
    return float(min(5.0, raw))


def ksi_to_level(ksi: float) -> int:
    """Map continuous KSI ∈ [0, 5] to discrete severity levels 1–5."""
    if ksi < 1.0:
        return 1
    if ksi < 2.0:
        return 2
    if ksi < 3.0:
        return 3
    if ksi < 4.0:
        return 4
    return 5


def severity_label(level: int) -> str:
    if level <= 2:
        return "Minor (Fender Bender)"
    if level == 3:
        return "Moderate (Lane Blockage)"
    return "Critical / Severe (Immediate Trauma Dispatch)"


def triage_recommendation(level: int) -> str:
    if level <= 2:
        return "Monitor incident; standard roadside assistance if needed."
    if level == 3:
        return "Dispatch traffic control; assess injuries; possible ambulance standby."
    return "IMMEDIATE trauma dispatch; clear corridor; prioritize EMS / fire."


def box_area(box: tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = box
    return float(max(0, x2 - x1) * max(0, y2 - y1))


def pixels_to_kmh(dx: float, dy: float, fps: float) -> tuple[float, float, float]:
    """Convert pixel displacement per frame into km/h-equivalent impact units."""
    scale = (fps / 30.0) * PX_PER_FRAME_TO_KMH
    vx, vy = dx * scale, dy * scale
    mag = float(math.hypot(vx, vy))
    if mag > MAX_SPEED_KMH and mag > 1e-6:
        scale_c = MAX_SPEED_KMH / mag
        vx, vy, mag = vx * scale_c, vy * scale_c, MAX_SPEED_KMH
    return vx, vy, mag


def clamp_delta_v(delta_v: float) -> float:
    return float(max(0.0, min(MAX_DELTA_V_KMH, delta_v)))

# ---------------------------------------------------------------------------
# Optical flow / motion (fast path)
# ---------------------------------------------------------------------------


def compute_farneback_flow(prev_gray: np.ndarray, gray: np.ndarray) -> np.ndarray:
    """Gunnar-Farneback dense optical flow → HxWx2 float32 (dx, dy) in pixels."""
    return cv2.calcOpticalFlowFarneback(
        prev_gray,
        gray,
        None,
        pyr_scale=0.5,
        levels=2,
        winsize=11,
        iterations=2,
        poly_n=5,
        poly_sigma=1.1,
        flags=0,
    )


def cheap_scene_motion(prev_small: np.ndarray, small: np.ndarray) -> float:
    """
    Fast scene-motion score from downscaled frame differencing.

    Scaled to roughly match the old Farneback p90 magnitude range used by
    aftermath gating (~0.1–2.0 on typical dashcam), without dense flow cost.
    """
    diff = cv2.absdiff(prev_small, small)
    return float(np.mean(diff)) / 12.0


def debris_plume_score(frame_bgr: np.ndarray) -> float:
    """
    Score elevated dust / smoke plumes from high-energy impacts.

    Ignores ground-level cues (asphalt glare, detached bumpers, crumpled panels)
    that belong to moderate property-damage crashes. Only mid/upper-frame
    desaturated gray masses with vertical extent count.
    """
    h, w = frame_bgr.shape[:2]
    tw = 160
    th = max(1, int(round(h * (tw / float(w)))))
    small = cv2.resize(frame_bgr, (tw, th), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    mask = (sat < 45) & (val > 155) & (val < 235)

    # Elevated band only — exclude road / bumper debris in the lower frame
    y0, y1 = int(th * 0.12), int(th * 0.58)
    band = mask[y0:y1, :]
    if band.size == 0:
        return 0.0

    frac = float(np.mean(band))
    col_means = band.mean(axis=0)
    if float(np.std(col_means)) < 0.06 and frac > 0.10:
        return float(min(0.10, frac * 0.4))

    band_u8 = (band.astype(np.uint8) * 255)
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(band_u8, connectivity=8)
    best = 0.0
    band_area = float(max(1, band.size))
    band_h = max(1, band.shape[0])
    band_w = max(1, band.shape[1])
    for i in range(1, n_labels):
        area = float(stats[i, cv2.CC_STAT_AREA]) / band_area
        bw = float(stats[i, cv2.CC_STAT_WIDTH]) / band_w
        bh = float(stats[i, cv2.CC_STAT_HEIGHT]) / band_h
        cy = float(stats[i, cv2.CC_STAT_TOP]) + 0.5 * float(stats[i, cv2.CC_STAT_HEIGHT])
        cy_norm = cy / band_h
        if bw > 0.75 and bh < 0.35:
            continue
        if bh < 0.18 and bw > 0.45:
            continue
        elev = 1.0 if 0.15 <= cy_norm <= 0.90 else 0.55
        compact = area * (0.7 + min(bh, 0.9)) * elev
        best = max(best, compact)

    return float(min(1.0, frac * 1.35 + best * 1.7))


def is_critical_plume(debris: float) -> bool:
    """High-energy airborne dust/smoke cloud (Level 4–5 eligible)."""
    return float(debris) >= CRITICAL_PLUME_THRESH


def impact_burst_delta_v(
    scene_motion: float,
    motion_history: list[float],
    debris: float,
    vehicle_count: int,
    *,
    debris_history: Optional[list[float]] = None,
) -> float:
    """
    Map sudden visual shock → Δv proxy.

    Critical airborne plumes → Level 4–5. Motion spikes without a critical plume
    stay at moderate Δv. Ground debris / crumpled panels alone do not escalate.
    """
    if len(motion_history) < 6:
        baseline = 0.35
        p80 = 0.5
    else:
        window = motion_history[-SCENE_FLOW_BASELINE_FRAMES:]
        baseline = float(np.median(window))
        p80 = float(np.percentile(window, 80))

    floor = max(baseline * 2.2, p80 * 1.5, 0.55)
    burst = max(0.0, scene_motion - floor)
    strong_burst = burst >= IMPACT_BURST_MOTION or scene_motion >= max(2.0, baseline * 3.5)

    debris_jump = debris
    if debris_history and len(debris_history) >= 4:
        debris_jump = debris - float(np.median(debris_history[-8:]))

    critical = is_critical_plume(debris) and (
        debris >= 0.70 or debris_jump >= 0.15 or strong_burst
    )
    if not critical and not (strong_burst and vehicle_count >= 1):
        return 0.0
    if not critical and vehicle_count == 0:
        return 0.0

    delta = 0.0
    if critical:
        delta = max(delta, 36.0 + debris * 22.0)
        if vehicle_count >= 1:
            delta += 6.0
    if strong_burst and critical:
        delta = max(delta, 30.0 + burst * 14.0)
        delta += 8.0
    elif strong_burst and vehicle_count >= 1 and not critical:
        delta = min(MODERATE_AFTERMATH_MAX_DV, 20.0 + burst * 10.0)
    return clamp_delta_v(delta)


def mean_flow_in_box(
    flow: np.ndarray,
    box: tuple[int, int, int, int],
    fps: float,
) -> tuple[float, float, float]:
    """Median dense-flow inside bbox → velocity (vx, vy, mag) in km/h-equivalent units."""
    h, w = flow.shape[:2]
    x1, y1, x2, y2 = box
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0, 0.0, 0.0

    patch = flow[y1:y2, x1:x2]
    if patch.size == 0:
        return 0.0, 0.0, 0.0

    dx = float(np.median(patch[..., 0]))
    dy = float(np.median(patch[..., 1]))
    return pixels_to_kmh(dx, dy, fps)


def bbox_center_velocity(
    prev_box: tuple[int, int, int, int],
    box: tuple[int, int, int, int],
    fps: float,
    *,
    frame_span: int = 1,
) -> tuple[float, float, float]:
    """Track-center displacement → per-frame velocity (supports analysis stride)."""
    span = max(1, int(frame_span))
    cx0, cy0 = box_center(prev_box)
    cx1, cy1 = box_center(box)
    dx, dy = (cx1 - cx0) / span, (cy1 - cy0) / span
    jump_limit = MAX_CENTER_JUMP_PX * span
    if math.hypot(cx1 - cx0, cy1 - cy0) > jump_limit:
        # Likely tracker ID reuse / re-acquisition — do not treat as physical Δv
        return 0.0, 0.0, 0.0
    return pixels_to_kmh(dx, dy, fps)


def estimate_track_velocity(
    flow: Optional[np.ndarray],
    phys: TrackPhysics,
    box: tuple[int, int, int, int],
    fps: float,
    *,
    frame_span: int = 1,
) -> tuple[float, float, float]:
    """
    Prefer bbox-center motion (fast + stable). Dense flow is optional fallback
    when a track is brand-new and has no prior box.
    """
    if phys.boxes:
        return bbox_center_velocity(phys.boxes[-1], box, fps, frame_span=frame_span)

    if flow is not None:
        return mean_flow_in_box(flow, box, fps)
    return 0.0, 0.0, 0.0


def scene_flow_stats(flow: np.ndarray) -> tuple[float, float]:
    """Return (mean, p90) flow magnitude for the frame."""
    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    return float(np.mean(mag)), float(np.percentile(mag, 90))


def analysis_stride_for_fps(fps: float) -> int:
    """Choose frame stride so analysis runs near TARGET_ANALYSIS_FPS."""
    fps = float(fps) if fps and fps > 0 else 24.0
    return max(1, int(round(fps / TARGET_ANALYSIS_FPS)))


def resize_gray(gray: np.ndarray, width: int) -> np.ndarray:
    h, w = gray.shape[:2]
    if w <= width:
        return gray
    new_h = max(1, int(round(h * (width / float(w)))))
    return cv2.resize(gray, (width, new_h), interpolation=cv2.INTER_AREA)

# ---------------------------------------------------------------------------
# Detection / tracking
# ---------------------------------------------------------------------------


def load_yolo_model(model_path: str = DEFAULT_MODEL) -> YOLO:
    try:
        model = YOLO(model_path)
        return model
    except Exception as exc:
        raise RuntimeError(f"Failed to load YOLO model '{model_path}': {exc}") from exc


def parse_yolo_tracks(result) -> list[dict]:
    """Extract vehicle tracks from a single Ultralytics result."""
    detections: list[dict] = []
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return detections

    xyxy = boxes.xyxy.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)
    conf = boxes.conf.cpu().numpy() if boxes.conf is not None else np.ones(len(xyxy))
    ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else np.arange(len(xyxy))

    for i in range(len(xyxy)):
        class_id = int(cls[i])
        if class_id not in VEHICLE_CLASS_IDS:
            continue
        x1, y1, x2, y2 = map(int, xyxy[i])
        detections.append(
            {
                "track_id": int(ids[i]),
                "class_id": class_id,
                "class_name": VEHICLE_CLASS_NAMES.get(class_id, "vehicle"),
                "confidence": float(conf[i]),
                "box": (x1, y1, x2, y2),
            }
        )
    return detections


# ---------------------------------------------------------------------------
# Collision / KSI analysis
# ---------------------------------------------------------------------------


def tracks_are_interacting(
    box_a: tuple[int, int, int, int],
    box_b: tuple[int, int, int, int],
    frame_shape: tuple[int, int],
) -> bool:
    if box_iou(box_a, box_b) >= PROXIMITY_IOU_COLLISION:
        return True
    h, w = frame_shape
    diag = math.hypot(w, h)
    ca, cb = box_center(box_a), box_center(box_b)
    return math.hypot(ca[0] - cb[0], ca[1] - cb[1]) <= PROXIMITY_CENTER_FRAC * diag


def _event_from_delta(
    frame_idx: int,
    timestamp_sec: float,
    delta_v: float,
    theta_deg: float,
    track_ids: tuple[int, ...],
    confidence_bias: float = 0.5,
) -> CollisionEvent:
    delta_v = clamp_delta_v(delta_v)
    # Strong jerks are treated as near-orthogonal impacts so KSI isn't muted by
    # noisy heading estimates (still respects the formula + angle floor).
    if delta_v >= 28.0:
        theta_deg = max(float(theta_deg), 80.0)
    else:
        theta_deg = max(float(theta_deg), IMPACT_ANGLE_FLOOR_DEG)
    ksi = kinetic_severity_index(delta_v, math.radians(theta_deg))
    level = ksi_to_level(ksi)
    conf = min(0.99, confidence_bias + delta_v / 50.0 + ksi / 8.0)
    return CollisionEvent(
        frame_idx=frame_idx,
        timestamp_sec=timestamp_sec,
        delta_v=float(delta_v),
        impact_angle_deg=float(theta_deg),
        ksi=float(ksi),
        severity_level=int(level),
        track_ids=track_ids,
        confidence=float(conf),
    )


def evaluate_frame_collisions(
    frame_idx: int,
    timestamp_sec: float,
    active: list[dict],
    physics: dict[int, TrackPhysics],
    frame_shape: tuple[int, int],
    *,
    sudden_decel_kmh: float = SUDDEN_DECEL_KMH,
    scene_delta_v: float = 0.0,
) -> list[CollisionEvent]:
    """Detect sudden Δv, multi-vehicle proximity, and scene-flow impact bursts."""
    events: list[CollisionEvent] = []
    n = len(active)
    decel_floor = float(sudden_decel_kmh)

    # Visual impact burst (debris plume / shockwave) — critical for real gallery
    # crashes where tracks vanish inside dust and aftermath stillness never holds.
    if scene_delta_v >= max(decel_floor * 1.2, 26.0):
        tids = tuple(sorted(d["track_id"] for d in active)) if active else (-1,)
        events.append(
            _event_from_delta(
                frame_idx,
                timestamp_sec,
                scene_delta_v,
                90.0,
                tids,
                confidence_bias=0.72,
            )
        )

    if n == 0:
        return events

    # Per-track vector Δv (captures impact jerks, not only scalar slowdowns)
    for det in active:
        tid = det["track_id"]
        phys = physics.get(tid)
        if not phys or not phys.delta_vs:
            continue
        delta_v = phys.delta_vs[-1]
        # Also credit strong decelerations of speed magnitude
        if phys.accelerations:
            delta_v = max(delta_v, abs(min(0.0, phys.accelerations[-1])))
        # Single-track needs a clearer spike than multi-vehicle interaction
        if len(phys.delta_vs) < 3:
            continue
        if delta_v < decel_floor * 1.35:
            continue
        theta_deg = 90.0
        if len(phys.velocities) >= 2:
            theta_deg = vector_angle_deg(phys.velocities[-2], phys.velocities[-1])
        events.append(
            _event_from_delta(
                frame_idx,
                timestamp_sec,
                delta_v,
                theta_deg,
                (tid,),
                confidence_bias=0.5,
            )
        )

    # Pairwise vehicle–vehicle: require a real jerk/decel, not steady relative motion.
    # Dashcam traffic routinely has large apparent closing speeds without any crash.
    for i in range(n):
        for j in range(i + 1, n):
            a, b = active[i], active[j]
            if not tracks_are_interacting(a["box"], b["box"], frame_shape):
                continue
            pa, pb = physics.get(a["track_id"]), physics.get(b["track_id"])
            if not pa or not pb:
                continue

            jerk = 0.0
            if pa.delta_vs:
                jerk = max(jerk, pa.delta_vs[-1])
            if pb.delta_vs:
                jerk = max(jerk, pb.delta_vs[-1])
            decel = 0.0
            if pa.accelerations:
                decel = max(decel, abs(min(0.0, pa.accelerations[-1])))
            if pb.accelerations:
                decel = max(decel, abs(min(0.0, pb.accelerations[-1])))
            kinematic = max(jerk, decel)

            if kinematic < decel_floor * 0.85:
                continue

            delta_v = kinematic
            if pa.velocities and pb.velocities:
                rvx = pa.velocities[-1][0] - pb.velocities[-1][0]
                rvy = pa.velocities[-1][1] - pb.velocities[-1][1]
                closing = math.hypot(rvx, rvy)
                # Mild boost only — closing alone must never create a Level 4–5 event
                delta_v = min(MAX_DELTA_V_KMH, delta_v + min(closing * 0.15, 8.0))

            iou = box_iou(a["box"], b["box"])
            if iou >= 0.15:
                delta_v = min(MAX_DELTA_V_KMH, delta_v + iou * 8.0)

            if delta_v < decel_floor * 1.1:
                continue

            # Need a short track history so cold-start pairs don't escalate
            if len(pa.delta_vs) < 2 or len(pb.delta_vs) < 2:
                continue
            va = pa.velocities[-1] if pa.velocities else (1.0, 0.0)
            vb = pb.velocities[-1] if pb.velocities else (-1.0, 0.0)
            theta_deg = vector_angle_deg(va, vb)
            events.append(
                _event_from_delta(
                    frame_idx,
                    timestamp_sec,
                    delta_v,
                    theta_deg,
                    (a["track_id"], b["track_id"]),
                    confidence_bias=0.62,
                )
            )
    return events


def evaluate_aftermath_severity(
    detections: list[dict],
    physics: dict[int, TrackPhysics],
    frame_shape: tuple[int, int],
    frame_idx: int,
    timestamp_sec: float,
    scene_motion: float,
    *,
    debris: float = 0.0,
) -> Optional[CollisionEvent]:
    """
    Static / low-motion wreckage triage.

    Sample dashcam clips often show the *aftermath* (tree on car, jackknifed
    trailer) rather than the impact impulse. Requires clear wreckage cues so
    ordinary traffic / large trucks in motion are not auto-escalated.
    Active debris plumes relax the stillness gate (real crashes are chaotic).
    """
    if not detections and debris < DEBRIS_SCORE_THRESH:
        return None

    h, w = frame_shape
    frame_area = float(max(1, h * w))
    areas = [box_area(d["box"]) for d in detections] if detections else [0.0]
    area_frac = float(sum(areas) / frame_area)
    max_frac = float(max(areas) / frame_area) if areas else 0.0
    mean_speed = 0.0
    speeds = []
    for d in detections:
        phys = physics.get(d["track_id"])
        if phys and phys.magnitudes:
            speeds.append(phys.magnitudes[-1])
    if speeds:
        mean_speed = float(np.mean(speeds))

    has_wreckage_proxy = any(d["class_name"] == "boat" for d in detections)
    has_heavy = any(d["class_name"] in {"truck", "bus", "train"} for d in detections)
    overlapping = False
    interacting = False
    for i in range(len(detections)):
        for j in range(i + 1, len(detections)):
            if box_iou(detections[i]["box"], detections[j]["box"]) >= 0.1:
                overlapping = True
                interacting = True
                break
            if tracks_are_interacting(
                detections[i]["box"],
                detections[j]["box"],
                frame_shape,
            ):
                interacting = True
        if overlapping and interacting:
            break

    # Debris plumes allow higher motion; otherwise require relative stillness
    critical = is_critical_plume(debris)
    motion_cap = 1.65 if critical else 0.75
    speed_cap = 14.0 if critical else 8.0
    if scene_motion > motion_cap or mean_speed > speed_cap:
        return None
    if frame_idx < 8:
        return None

    has_train = any(d["class_name"] == "train" for d in detections)
    has_car = any(d["class_name"] == "car" for d in detections)

    if critical and (len(detections) >= 1 or debris >= 0.70):
        pass
    elif has_wreckage_proxy and (area_frac >= 0.06 or max_frac >= 0.10):
        pass
    elif has_train and has_car and (overlapping or interacting) and area_frac >= 0.10:
        pass
    elif (
        has_heavy
        and overlapping
        and area_frac >= 0.20
        and scene_motion < 0.35
        and mean_speed < 4.0
        and not has_train
    ):
        pass
    elif (
        has_heavy
        and max_frac >= 0.16
        and scene_motion < 0.28
        and mean_speed < 5.0
        and frame_idx >= 12
    ):
        pass
    elif (
        has_car
        and len(detections) >= 2
        and interacting
        and mean_speed < 5.0
        and area_frac >= 0.08
        and scene_motion < 0.55
    ):
        # Nearby stopped/tangled cars — moderate property-damage pathway (Level 2–3)
        pass
    else:
        return None

    # Vehicle-damage / wreckage without a critical airborne plume → Level 2–3 only
    damage_delta = 16.0 + area_frac * 45.0 + max_frac * 20.0
    if has_heavy or has_train:
        damage_delta += 4.0
    if has_wreckage_proxy:
        damage_delta += 5.0
    if overlapping or interacting or len(detections) >= 2:
        damage_delta += 3.0

    if critical:
        damage_delta = max(damage_delta, 38.0 + debris * 20.0)
        damage_delta = float(min(MAX_DELTA_V_KMH, damage_delta))
    else:
        # Soft ceiling: crumpled panels / bumper debris → triage 2–3, not trauma dispatch
        damage_delta = float(min(MODERATE_AFTERMATH_MAX_DV, max(20.0, damage_delta)))

    tids = tuple(sorted(d["track_id"] for d in detections)) if detections else (-1,)
    return _event_from_delta(
        frame_idx,
        timestamp_sec,
        damage_delta,
        90.0,
        tids,
        confidence_bias=0.78 if critical else 0.62,
    )

# ---------------------------------------------------------------------------
# Annotation
# ---------------------------------------------------------------------------


_SEVERITY_COLORS = {
    1: (80, 200, 80),
    2: (60, 220, 180),
    3: (0, 200, 255),
    4: (0, 140, 255),
    5: (0, 0, 255),
}


def annotate_frame(
    frame: np.ndarray,
    detections: list[dict],
    physics: dict[int, TrackPhysics],
    warning_level: Optional[int] = None,
    warning_text: Optional[str] = None,
) -> np.ndarray:
    """Overlay boxes, IDs, speed vectors, and optional crash banner (BGR)."""
    out = frame.copy()
    h, w = out.shape[:2]

    for det in detections:
        tid = det["track_id"]
        x1, y1, x2, y2 = det["box"]
        color = (0, 200, 255)
        phys = physics.get(tid)
        mag = phys.magnitudes[-1] if phys and phys.magnitudes else 0.0
        vx, vy = phys.velocities[-1] if phys and phys.velocities else (0.0, 0.0)

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"ID {tid} {det['class_name']} | {mag:.1f} km/h"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        ty = max(0, y1 - th - 6)
        cv2.rectangle(out, (x1, ty), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            out,
            label,
            (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

        # Velocity vector from box center
        cx, cy = box_center(det["box"])
        tip = (int(cx + vx * 1.5), int(cy + vy * 1.5))
        cv2.arrowedLine(out, (int(cx), int(cy)), tip, (255, 180, 50), 2, tipLength=0.35)

    if warning_level and warning_text:
        banner_h = max(36, h // 12)
        color = _SEVERITY_COLORS.get(warning_level, (0, 0, 255))
        overlay = out.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_h), color, -1)
        out = cv2.addWeighted(overlay, 0.72, out, 0.28, 0)
        cv2.putText(
            out,
            warning_text,
            (12, int(banner_h * 0.68)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return out


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------


def open_h264_writer(output_path: Path, fps: float, frame_size: tuple[int, int]):
    """Open a streaming imageio H.264 writer (yuv420p / HTML5-safe)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fps = float(fps) if fps and fps > 0 else 24.0
    w, h = frame_size
    ew, eh = _clamp_even(w), _clamp_even(h)
    writer = imageio.get_writer(
        str(output_path),
        fps=fps,
        codec="libx264",
        format="FFMPEG",
        pixelformat="yuv420p",
        macro_block_size=1,  # avoid silent resize warnings / extra copies
        output_params=["-crf", "28", "-preset", "ultrafast", "-threads", "2"],
    )
    return writer, (ew, eh)


def append_h264_frame(writer, frame_bgr: np.ndarray, size: tuple[int, int]) -> None:
    """Convert BGR→RGB, enforce even size, append to streaming writer."""
    ew, eh = size
    if frame_bgr.shape[1] != ew or frame_bgr.shape[0] != eh:
        frame_bgr = cv2.resize(frame_bgr, (ew, eh), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    writer.append_data(rgb)


def write_h264_mp4(frames_rgb: list[np.ndarray], fps: float, output_path: Path) -> Path:
    """
    Encode RGB frames to browser-native H.264 via imageio/FFMPEG.

    Kept for compatibility; prefer streaming via open_h264_writer in process_video.
    """
    if not frames_rgb:
        raise ValueError("No frames to encode.")

    output_path = Path(output_path)
    h, w = frames_rgb[0].shape[:2]
    writer, size = open_h264_writer(output_path, fps, (w, h))
    try:
        ew, eh = size
        for frame in frames_rgb:
            if frame.shape[1] != ew or frame.shape[0] != eh:
                frame = cv2.resize(frame, (ew, eh), interpolation=cv2.INTER_AREA)
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            writer.append_data(frame)
    except Exception as exc:
        raise RuntimeError(f"H.264 encode failed for {output_path}: {exc}") from exc
    finally:
        try:
            writer.close()
        except Exception as close_exc:  # noqa: BLE001
            logger.warning("Writer close warning: %s", close_exc)

    if not output_path.exists() or output_path.stat().st_size < 500:
        raise RuntimeError(f"Encoded video missing or empty: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Dispatch payload
# ---------------------------------------------------------------------------


def build_dispatch_payload(
    video_path: str,
    collision: Optional[CollisionEvent],
    max_severity: int,
    max_delta_v: float,
    incident_confidence: float,
) -> dict:
    level = max(1, min(5, int(max_severity or 1)))
    ts = collision.timestamp_sec if collision else 0.0
    return {
        "incident_id": f"impact-{int(time.time())}",
        "source_video": str(video_path),
        "timestamp_sec": round(float(ts), 3),
        "collision_frame_idx": int(collision.frame_idx) if collision else None,
        "severity_level": level,
        "severity_label": severity_label(level),
        "kinetic_severity_index": round(float(collision.ksi), 3) if collision else 0.0,
        "max_delta_v_kmh": round(float(max_delta_v), 2),
        "impact_angle_deg": round(float(collision.impact_angle_deg), 1) if collision else None,
        "involved_track_ids": list(collision.track_ids) if collision else [],
        "incident_confidence": round(float(incident_confidence), 3),
        "triage_recommendation": triage_recommendation(level),
        "dispatch_priority": "routine" if level <= 2 else ("elevated" if level == 3 else "emergency"),
        "units_suggested": (
            ["tow"] if level <= 2 else (["traffic", "ems_standby"] if level == 3 else ["ems", "fire", "traffic"])
        ),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def process_video(
    video_path: str,
    progress_callback: Optional[ProgressCallback] = None,
    *,
    model_path: str = DEFAULT_MODEL,
    model: Optional[YOLO] = None,
    output_dir: Optional[str] = None,
    conf_threshold: float = 0.35,
    sudden_decel_kmh: float = SUDDEN_DECEL_KMH,
) -> dict:
    """
    Run Impact-Net triage on a video file.

    Performance notes
    -----------------
    - YOLO + physics run at ~TARGET_ANALYSIS_FPS (strided), not necessarily every
      source frame — kinematics are stride-normalized so Δv/KSI stay honest.
    - Velocity uses bbox-center tracking (primary); dense Farneback is skipped on
      the hot path. Scene stillness for aftermath uses cheap frame-diff motion.
    - Annotated H.264 is streamed frame-by-frame (no full-video RAM buffer).
    """

    def report(frac: float, msg: str) -> None:
        if progress_callback is not None:
            try:
                progress_callback(float(max(0.0, min(1.0, frac))), msg)
            except Exception as cb_exc:  # noqa: BLE001 — UI callback must not kill pipeline
                logger.debug("progress_callback error: %s", cb_exc)

    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    out_root = Path(output_dir) if output_dir else Path.cwd() / OUTPUT_DIR_NAME
    out_root.mkdir(parents=True, exist_ok=True)
    output_video_path = out_root / f"{path.stem}_impactnet_annotated.mp4"

    if model is None:
        report(0.02, "Loading YOLO model…")
        model = load_yolo_model(model_path)
    else:
        report(0.02, "Using cached YOLO model…")
        try:
            model.predictor = None
        except Exception:  # noqa: BLE001
            pass

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video_path}")
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    except Exception:  # noqa: BLE001
        pass

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 24.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    stride = analysis_stride_for_fps(fps)
    vehicle_classes = list(VEHICLE_CLASS_IDS)

    physics: dict[int, TrackPhysics] = {}
    all_events: list[CollisionEvent] = []
    telemetry_rows: list[dict] = []

    prev_small: Optional[np.ndarray] = None
    frame_idx = 0
    analysis_tick = 0
    active_warning_level: Optional[int] = None
    active_warning_until = -1
    peak_event: Optional[CollisionEvent] = None
    scene_motion_history: list[float] = []
    debris_history: list[float] = []
    last_detections: list[dict] = []
    last_anomaly = 0.0
    last_max_accel = 0.0

    writer = None
    writer_size: Optional[tuple[int, int]] = None
    frames_written = 0

    try:
        report(0.05, f"Processing @ ~{TARGET_ANALYSIS_FPS:.0f} Hz analysis (stride={stride})…")
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            proc, _scale = scale_to_max_width(frame_bgr, MAX_PROCESS_WIDTH)
            ph, pw = proc.shape[:2]
            ew, eh = _clamp_even(pw), _clamp_even(ph)
            if (pw, ph) != (ew, eh):
                proc = cv2.resize(proc, (ew, eh), interpolation=cv2.INTER_LINEAR)

            if writer is None:
                writer, writer_size = open_h264_writer(output_video_path, fps, (ew, eh))

            gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
            small = resize_gray(gray, MOTION_WIDTH)
            timestamp = frame_idx / fps
            is_analysis = (frame_idx % stride == 0) or frame_idx == 0

            detections = last_detections
            scene_motion = scene_motion_history[-1] if scene_motion_history else 0.0
            scene_delta_v = 0.0
            max_accel_frame = last_max_accel
            anomaly = last_anomaly
            events: list[CollisionEvent] = []

            if is_analysis:
                analysis_tick += 1
                # --- YOLO track ---
                try:
                    results = model.track(
                        proc,
                        persist=True,
                        conf=conf_threshold,
                        classes=vehicle_classes,
                        imgsz=YOLO_IMGSZ,
                        verbose=False,
                    )
                    detections = parse_yolo_tracks(results[0]) if results else []
                    # Low-conf wreckage pass whenever the primary pass is empty
                    if not detections and conf_threshold > 0.2:
                        results = model.track(
                            proc,
                            persist=True,
                            conf=0.18,
                            classes=vehicle_classes,
                            imgsz=max(YOLO_IMGSZ, 480),
                            verbose=False,
                        )
                        detections = parse_yolo_tracks(results[0]) if results else []
                except Exception as det_exc:
                    logger.warning("YOLO track failed at frame %d: %s", frame_idx, det_exc)
                    detections = []

                # --- Cheap scene motion + debris plume (real crash cue) ---
                debris = debris_plume_score(proc)
                if prev_small is not None and prev_small.shape == small.shape:
                    scene_motion = cheap_scene_motion(prev_small, small)
                    scene_delta_v = impact_burst_delta_v(
                        scene_motion,
                        scene_motion_history,
                        debris,
                        vehicle_count=len(detections),
                        debris_history=debris_history,
                    )
                    scene_motion_history.append(scene_motion)
                    if len(scene_motion_history) > SCENE_FLOW_BASELINE_FRAMES * 3:
                        scene_motion_history = scene_motion_history[-SCENE_FLOW_BASELINE_FRAMES * 2 :]
                else:
                    scene_delta_v = impact_burst_delta_v(
                        0.0,
                        scene_motion_history,
                        debris,
                        vehicle_count=len(detections),
                        debris_history=debris_history,
                    )
                debris_history.append(debris)
                if len(debris_history) > 40:
                    debris_history = debris_history[-30:]

                # --- Per-track velocities (bbox centers, stride-normalized) ---
                max_accel_frame = 0.0
                for det in detections:
                    tid = det["track_id"]
                    if tid not in physics:
                        physics[tid] = TrackPhysics(track_id=tid, class_name=det["class_name"])
                    vx, vy, _mag = estimate_track_velocity(
                        None,
                        physics[tid],
                        det["box"],
                        fps,
                        frame_span=stride,
                    )
                    delta_v = physics[tid].push(frame_idx, det["box"], vx, vy)
                    max_accel_frame = max(
                        max_accel_frame,
                        delta_v,
                        abs(physics[tid].accelerations[-1]),
                    )

                events = evaluate_frame_collisions(
                    frame_idx,
                    timestamp,
                    detections,
                    physics,
                    (proc.shape[0], proc.shape[1]),
                    sudden_decel_kmh=sudden_decel_kmh,
                    scene_delta_v=scene_delta_v,
                )

                aftermath = evaluate_aftermath_severity(
                    detections,
                    physics,
                    (proc.shape[0], proc.shape[1]),
                    frame_idx,
                    timestamp,
                    scene_motion=scene_motion,
                    debris=debris,
                )
                if aftermath is not None:
                    events.append(aftermath)

                anomaly = 0.0
                for ev in events:
                    all_events.append(ev)
                    anomaly = max(anomaly, ev.ksi)
                    if peak_event is None or ev.ksi > peak_event.ksi:
                        peak_event = ev
                    if ev.severity_level >= 3 or ev.delta_v >= sudden_decel_kmh:
                        active_warning_level = ev.severity_level
                        active_warning_until = frame_idx + int(fps * 2.5)

                last_detections = detections
                last_anomaly = anomaly
                last_max_accel = max_accel_frame
                prev_small = small

                telemetry_rows.append(
                    {
                        "frame_idx": frame_idx,
                        "timestamp": round(timestamp, 4),
                        "vehicle_count": len(detections),
                        "max_acceleration": round(max_accel_frame, 4),
                        "anomaly_score": round(float(anomaly), 4),
                    }
                )

            warning_level = active_warning_level if frame_idx <= active_warning_until else None
            warning_text = f"CRASH DETECTED - LEVEL {warning_level}" if warning_level else None

            annotated = annotate_frame(
                proc,
                detections,
                physics,
                warning_level=warning_level,
                warning_text=warning_text,
            )
            append_h264_frame(writer, annotated, writer_size)  # type: ignore[arg-type]
            frames_written += 1
            frame_idx += 1

            if frame_idx % PROGRESS_EVERY == 0 or (total_frames > 0 and frame_idx == total_frames):
                if total_frames > 0:
                    report(
                        0.05 + 0.90 * (frame_idx / total_frames),
                        f"Frame {frame_idx}/{total_frames}",
                    )
                else:
                    report(min(0.92, 0.05 + frame_idx / 800.0), f"Frame {frame_idx}")

    finally:
        cap.release()
        if writer is not None:
            try:
                writer.close()
            except Exception as close_exc:  # noqa: BLE001
                logger.warning("Writer close warning: %s", close_exc)

    if frames_written == 0:
        raise RuntimeError(f"No frames decoded from {video_path}")

    if not output_video_path.exists() or output_video_path.stat().st_size < 500:
        raise RuntimeError(f"Encoded video missing or empty: {output_video_path}")

    frame_telemetry = pd.DataFrame(
        telemetry_rows,
        columns=["frame_idx", "timestamp", "vehicle_count", "max_acceleration", "anomaly_score"],
    )
    if frame_telemetry.empty:
        # Extremely short clip — synthesize a minimal telemetry row
        frame_telemetry = pd.DataFrame(
            [
                {
                    "frame_idx": 0,
                    "timestamp": 0.0,
                    "vehicle_count": 0,
                    "max_acceleration": 0.0,
                    "anomaly_score": 0.0,
                }
            ]
        )

    max_delta_v = float(max((e.delta_v for e in all_events), default=0.0))
    if peak_event is None and not frame_telemetry.empty:
        idx = int(frame_telemetry["anomaly_score"].idxmax())
        max_severity = 1
        collision_frame_idx = int(frame_telemetry.loc[idx, "frame_idx"])
        incident_confidence = 0.15
    else:
        max_severity = int(peak_event.severity_level) if peak_event else 1
        collision_frame_idx = int(peak_event.frame_idx) if peak_event else 0
        incident_confidence = float(peak_event.confidence) if peak_event else 0.1

    if all_events:
        max_severity = max(e.severity_level for e in all_events)
        max_delta_v = max(e.delta_v for e in all_events)
        # Sustained high-Δv bursts → escalate L3 → L4 only for violent kinetics,
        # not for moderate property-damage aftermath (capped Δv).
        strong = [
            e
            for e in all_events
            if e.severity_level >= 3 and e.delta_v >= 34.0
        ]
        if max_severity == 3 and len(strong) >= 2:
            max_severity = 4
            if peak_event is not None:
                incident_confidence = max(incident_confidence, 0.8)

    dispatch_payload = build_dispatch_payload(
        video_path=str(path),
        collision=peak_event,
        max_severity=max_severity,
        max_delta_v=max_delta_v,
        incident_confidence=incident_confidence,
    )

    report(1.0, "Complete")
    logger.info(
        "Processed %d frames (stride=%d) → %s | severity=%d Δv=%.1f",
        frame_idx,
        stride,
        output_video_path,
        max_severity,
        max_delta_v,
    )

    return {
        "output_video_path": str(output_video_path.resolve()),
        "max_severity_level": int(max_severity),
        "max_delta_v": float(round(max_delta_v, 3)),
        "collision_frame_idx": int(collision_frame_idx),
        "frame_telemetry": frame_telemetry,
        "dispatch_payload": dispatch_payload,
    }


__all__ = [
    "process_video",
    "kinetic_severity_index",
    "ksi_to_level",
    "severity_label",
    "triage_recommendation",
]
