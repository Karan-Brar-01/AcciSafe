#!/usr/bin/env python3
"""
Download (or synthesize) short sample collision clips for Impact-Net demos.

Prefer royalty-free public MP4 URLs. If a download fails, generate a synthetic
dashcam-style animated clip with OpenCV so ./samples/ is always demo-ready.
"""

from __future__ import annotations

import logging
import math
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("download_samples")

SAMPLES_DIR = Path(__file__).resolve().parent / "samples"
DOWNLOAD_TIMEOUT_SEC = 60
CHUNK_SIZE = 1 << 16  # 64 KiB


class SampleClip(NamedTuple):
    filename: str
    url: str | None
    description: str
    synthetic_seed: int
    scenario: str = "crash"  # "crash" | "safe"


# Direct public MP4s (Mixkit free stock / Internet Archive public domain).
# Includes severe incidents AND clear/normal traffic for severity variety.
CLIPS: tuple[SampleClip, ...] = (
    # --- Severe / crash demos ---
    SampleClip(
        filename="trailer_highway_crash.mp4",
        url="https://assets.mixkit.co/videos/49147/49147-720.mp4",
        description="Trailer crash on a highway (Mixkit free stock)",
        synthetic_seed=42,
        scenario="crash",
    ),
    SampleClip(
        filename="tree_fallen_on_car.mp4",
        url="https://assets.mixkit.co/videos/9269/9269-720.mp4",
        description="Tree fallen on a car (Mixkit free stock)",
        synthetic_seed=137,
        scenario="crash",
    ),
    SampleClip(
        filename="car_train_crash_archive.mp4",
        url="https://archive.org/download/CEP195/CEP195_512kb.mp4",
        description="Historic car–train crash stock footage (Internet Archive, PD)",
        synthetic_seed=911,
        scenario="crash",
    ),
    # --- Safe / minor demos (no severe accident) ---
    SampleClip(
        filename="safe_city_highway_drive.mp4",
        url="https://assets.mixkit.co/videos/42369/42369-720.mp4",
        description="Normal city highway drive — no crash (Mixkit free stock)",
        synthetic_seed=201,
        scenario="safe",
    ),
    SampleClip(
        filename="safe_sunny_highway_pov.mp4",
        url="https://assets.mixkit.co/videos/42367/42367-720.mp4",
        description="Sunny highway POV travel — no crash (Mixkit free stock)",
        synthetic_seed=202,
        scenario="safe",
    ),
    SampleClip(
        filename="safe_clear_lane_traffic.mp4",
        url=None,  # synthetic steady traffic — guaranteed minor/safe demo
        description="Synthetic clear-lane traffic — no crash (local demo)",
        synthetic_seed=203,
        scenario="safe",
    ),
)


def ensure_samples_dir() -> Path:
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    return SAMPLES_DIR


def download_file(url: str, dest: Path) -> None:
    """Stream a remote file to disk with a timeout and basic integrity checks."""
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Impact-Net/1.0 (sample-clip-fetcher)"},
        method="GET",
    )
    partial = dest.with_suffix(dest.suffix + ".part")

    try:
        with urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT_SEC) as response:
            status = getattr(response, "status", None) or response.getcode()
            if status != 200:
                raise urllib.error.HTTPError(
                    url, status, f"Unexpected HTTP status {status}", response.headers, None
                )

            content_type = (response.headers.get("Content-Type") or "").lower()
            if content_type and "html" in content_type and "video" not in content_type:
                raise ValueError(f"Expected video bytes, got Content-Type={content_type!r}")

            bytes_written = 0
            with partial.open("wb") as out:
                while True:
                    chunk = response.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    out.write(chunk)
                    bytes_written += len(chunk)

            if bytes_written < 10_000:
                raise ValueError(f"Downloaded file too small ({bytes_written} bytes)")

        partial.replace(dest)
    except Exception:
        if partial.exists():
            partial.unlink(missing_ok=True)
        raise


def _draw_road(frame: np.ndarray, t: float, seed: int) -> None:
    h, w = frame.shape[:2]
    rng = np.random.default_rng(seed)

    # Asphalt
    frame[:] = (48, 48, 52)
    # Shoulder / verge
    cv2.rectangle(frame, (0, 0), (w // 8, h), (34, 90, 40), -1)
    cv2.rectangle(frame, (7 * w // 8, 0), (w, h), (34, 90, 40), -1)

    # Lane markings scrolling toward camera
    lane_x = w // 2
    cv2.line(frame, (lane_x, 0), (lane_x, h), (220, 220, 220), 2)
    dash_len = 40
    offset = int((t * 280) % (dash_len * 2))
    for y in range(-dash_len + offset, h, dash_len * 2):
        cv2.rectangle(
            frame,
            (lane_x - 4, y),
            (lane_x + 4, y + dash_len),
            (240, 240, 240),
            -1,
        )

    # Distant sky gradient strip
    for i in range(h // 5):
        shade = int(90 + i * 0.8)
        frame[i, :] = (shade + 20, shade + 10, shade)

    # Noise grain for CCTV feel
    noise = rng.integers(0, 18, size=(h, w, 1), dtype=np.uint8)
    frame[:] = cv2.add(frame, cv2.merge([noise[:, :, 0]] * 3))


def _vehicle_rect(
    center_x: float,
    center_y: float,
    scale: float,
    color: tuple[int, int, int],
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int, int]]:
    half_w = int(55 * scale)
    half_h = int(30 * scale)
    x1, y1 = int(center_x - half_w), int(center_y - half_h)
    x2, y2 = int(center_x + half_w), int(center_y + half_h)
    return (x1, y1), (x2, y2), color


def create_synthetic_clip(
    dest: Path,
    seed: int,
    duration_sec: float = 4.0,
    fps: int = 24,
    *,
    scenario: str = "crash",
) -> None:
    """
    Write a short synthetic dashcam-style clip.

    scenario="crash": two vehicles collide mid-clip.
    scenario="safe": steady lane traffic with no impact (for minor/safe demos).
    """
    width, height = 640, 360
    total_frames = max(1, int(duration_sec * fps))
    impact_frame = int(total_frames * 0.55)
    is_safe = scenario == "safe"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(dest), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for {dest}")

    try:
        for i in range(total_frames):
            t = i / fps
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            _draw_road(frame, t, seed + i)

            if is_safe:
                # Steady lead vehicle + distant traffic — no collision
                ego_y = height * 0.62
                ego_x = width * 0.48 + 6 * math.sin(t * 1.2)
                p1, p2, color = _vehicle_rect(ego_x, ego_y, 1.05, (40, 90, 220))
                cv2.rectangle(frame, p1, p2, color, -1)
                cv2.rectangle(frame, p1, p2, (20, 40, 120), 2)

                other_x = width * 0.28
                other_y = height * 0.38 + (i % 20) * 0.4
                o1, o2, ocolor = _vehicle_rect(other_x, other_y, 0.75, (30, 180, 90))
                cv2.rectangle(frame, o1, o2, ocolor, -1)
                cv2.rectangle(frame, o1, o2, (10, 90, 40), 2)
                banner = "IMPACT-NET SAMPLE (SAFE / NO CRASH)"
                banner_color = (80, 220, 120)
            else:
                # Ego lane vehicle (bottom → center)
                progress = min(1.0, i / impact_frame) if impact_frame else 1.0
                ego_y = height - 40 - progress * (height * 0.35)
                ego_x = width * 0.42 + (i % 3)
                p1, p2, color = _vehicle_rect(ego_x, ego_y, 1.0 + progress * 0.4, (40, 90, 220))
                cv2.rectangle(frame, p1, p2, color, -1)
                cv2.rectangle(frame, p1, p2, (20, 40, 120), 2)

                if i < impact_frame:
                    other_x = width * 0.85 - progress * width * 0.38
                    other_y = height * 0.35
                else:
                    post = (i - impact_frame) / max(1, total_frames - impact_frame)
                    other_x = width * 0.47 + post * 120
                    other_y = height * 0.35 + post * 80

                scale = 0.7 + progress * 0.5
                o1, o2, ocolor = _vehicle_rect(other_x, other_y, scale, (30, 180, 90))
                cv2.rectangle(frame, o1, o2, ocolor, -1)
                cv2.rectangle(frame, o1, o2, (10, 90, 40), 2)

                if abs(i - impact_frame) <= 2:
                    overlay = frame.copy()
                    cv2.circle(
                        overlay,
                        (int(width * 0.48), int(height * 0.42)),
                        70,
                        (255, 255, 255),
                        -1,
                    )
                    frame = cv2.addWeighted(frame, 0.45, overlay, 0.55, 0)

                if i >= impact_frame:
                    rng = np.random.default_rng(seed + i)
                    for _ in range(12):
                        dx = int(rng.integers(-90, 90))
                        dy = int(rng.integers(-40, 60))
                        cx = int(width * 0.48 + dx)
                        cy = int(height * 0.42 + dy)
                        cv2.circle(frame, (cx, cy), int(rng.integers(2, 6)), (180, 180, 200), -1)

                banner = "IMPACT-NET SAMPLE (SYNTHETIC CRASH)"
                banner_color = (0, 220, 255)

            stamp = f"CAM-SYNTH-{seed:03d}  t={t:0.2f}s"
            cv2.putText(
                frame,
                stamp,
                (12, height - 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                banner,
                (12, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                banner_color,
                1,
                cv2.LINE_AA,
            )

            writer.write(frame)
    finally:
        writer.release()

    if not dest.exists() or dest.stat().st_size < 1000:
        raise RuntimeError(f"Synthetic clip was not written correctly: {dest}")


def prepare_clip(clip: SampleClip) -> Path:
    ensure_samples_dir()
    dest = SAMPLES_DIR / clip.filename

    if dest.exists() and dest.stat().st_size > 10_000:
        logger.info("Already present: %s (%s)", dest.name, clip.description)
        return dest

    if clip.url:
        logger.info("Downloading %s …", clip.filename)
        try:
            download_file(clip.url, dest)
            logger.info("Saved download → %s (%.1f KB)", dest, dest.stat().st_size / 1024)
            return dest
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError, OSError) as exc:
            logger.warning("Download failed for %s: %s", clip.filename, exc)
            if dest.exists():
                dest.unlink(missing_ok=True)
    else:
        logger.info("No URL configured for %s; synthesizing.", clip.filename)

    logger.info("Generating synthetic %s clip → %s", clip.scenario, clip.filename)
    create_synthetic_clip(dest, seed=clip.synthetic_seed, scenario=clip.scenario)
    logger.info("Saved synthetic → %s (%.1f KB)", dest, dest.stat().st_size / 1024)
    return dest


def main() -> int:
    logger.info("Impact-Net sample prep → %s", SAMPLES_DIR)
    ensure_samples_dir()

    failures: list[str] = []
    for clip in CLIPS:
        try:
            prepare_clip(clip)
        except Exception as exc:  # noqa: BLE001 — surface per-clip failure, continue others
            logger.exception("Failed to prepare %s: %s", clip.filename, exc)
            failures.append(clip.filename)

    ready = sorted(SAMPLES_DIR.glob("*.mp4"))
    logger.info("Samples ready: %d file(s) in %s", len(ready), SAMPLES_DIR)
    for path in ready:
        logger.info("  • %s (%.1f KB)", path.name, path.stat().st_size / 1024)

    if failures and not ready:
        logger.error("No sample clips available.")
        return 1
    if failures:
        logger.warning("Partial success; failed: %s", ", ".join(failures))
        return 0
    logger.info("All sample clips ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
