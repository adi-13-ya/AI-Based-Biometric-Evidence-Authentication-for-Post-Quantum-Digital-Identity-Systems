"""Frame encoding/decoding helpers for the Streamlit demo.

Keeps image/video handling out of streamlit_app.py so it's independently
testable without a Streamlit runtime.
"""
from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
VIDEO_SUFFIXES = {".mp4", ".mov"}
_MAX_VIDEO_FRAMES = 900  # read at most 30 s of a 30 fps clip


def frame_to_png_bytes(frame: np.ndarray) -> bytes:
    buf = BytesIO()
    Image.fromarray(frame).convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def frame_to_b64(frame: np.ndarray) -> str:
    return base64.b64encode(frame_to_png_bytes(frame)).decode("ascii")


def load_image_frames(path: Path) -> tuple[list[np.ndarray], None]:
    img = Image.open(path).convert("RGB")
    return [np.array(img)], None


_BURST_START = 3  # skip the first frames (camera exposure settling)
BURST_FRAMES = 8  # consecutive frames appended for the injection classifier


def load_video_frames(path: Path, max_frames: int = 8, burst: int = BURST_FRAMES) -> tuple[list[np.ndarray], list[float], int]:
    """Uniformly samples up to `max_frames` frames from the video, appends
    `burst` *consecutive* frames from the start of the clip, and returns the
    timestamp (seconds, from the container's frame rate) of *every* frame in
    the file - the same shape as a live recording (recorder.select_frames).
    Returns (frames, timestamps, number of trailing burst frames)."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or max_frames
    step = max(1, total // max_frames)

    frames: list[np.ndarray] = []
    burst_frames: list[np.ndarray] = []
    timestamps: list[float] = []
    idx = 0
    try:
        while idx < _MAX_VIDEO_FRAMES:
            ok, bgr = cap.read()
            if not ok:
                break
            timestamps.append(idx / fps)
            rgb = None
            if idx % step == 0 and len(frames) < max_frames:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                frames.append(rgb)
            if _BURST_START <= idx < _BURST_START + burst:
                burst_frames.append(rgb if rgb is not None else cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            idx += 1
    finally:
        cap.release()

    if not frames:
        raise ValueError(f"No frames could be read from video file: {path}")
    if len(burst_frames) < 2:
        burst_frames = []
    return frames + burst_frames, timestamps, len(burst_frames)


def quality_check(frame: np.ndarray) -> tuple[bool, str]:
    """Local capture pre-check (spec Phase 2): reject frames too dark,
    over-exposed or blurry to be worth sending. Sharpness is whole-frame
    Laplacian variance at a 640 px long side; webcam frames typically
    measure 15-50 on this scale, so only a severely blurred frame (< 10) is
    flagged. The server's PAD check measures the face itself."""
    img = Image.fromarray(frame).convert("L")
    scale = 640 / max(img.size)
    if scale < 1:
        img = img.resize((round(img.width * scale), round(img.height * scale)))
    gray = np.asarray(img, dtype=np.float64)
    brightness = float(gray.mean())
    if brightness < 40:
        return False, f"too dark (mean brightness {brightness:.0f}/255) - add light in front of the face"
    if brightness > 220:
        return False, f"over-exposed (mean brightness {brightness:.0f}/255)"
    lap = -4 * gray[1:-1, 1:-1] + gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:]
    sharpness = float(lap.var())
    if sharpness < 10:
        return False, f"too blurry (sharpness {sharpness:.0f}) - hold still and clean the lens"
    return True, "ok"


def load_attack_file(path: Path, max_frames: int = 8) -> tuple[list[np.ndarray], list[float] | None, int]:
    """(frames, timestamps, burst_frames) - see load_video_frames."""
    suffix = path.suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return (*load_image_frames(path), 0)
    if suffix in VIDEO_SUFFIXES:
        return load_video_frames(path, max_frames=max_frames)
    raise ValueError(f"Unsupported attack sample file type {suffix!r} (expected one of {IMAGE_SUFFIXES | VIDEO_SUFFIXES})")
