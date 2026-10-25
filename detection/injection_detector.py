"""Injection Attack Detection (virtual camera / pre-recorded stream) - spec Section 6.5.

Threat: instead of pointing a camera at a face, the attacker feeds a video
file (a recording, or a deepfake render) into a virtual camera driver, so
the capture app receives synthetic frames that never passed through a
sensor. The client reports frame timestamps itself, so a malicious client
can forge them; the classifier therefore relies on what the pixels show.
Three artefacts follow from how a virtual camera is fed:

1. Frame-rate conversion. A 24/25 fps file played into a 30 fps virtual
   camera repeats frames. A sensor never delivers two identical frames
   because read noise is fresh every exposure, so near-zero differences
   between consecutive frames are a strong cue (`dup_rate`, `min_diff`).
2. Missing sensor noise. Inter-frame differences of a live capture contain
   fresh, spatially white noise; a decoded file's differences are smoothed
   by the codec's motion-compensated prediction (`temporal_noise`,
   `diff_hf_ratio`).
3. A second compression generation. The file was encoded once when it was
   made and the stream is compressed again on the way in, which raises
   gradient energy on the 8x8 (and 16x16 macroblock) grid relative to
   off-grid positions, both in the frames and in the frame differences
   (`block_ratio_8`, `block_ratio_16`, `diff_block_ratio`). Re-encoding also
   copies more blocks unchanged from the previous frame ("skip" blocks), so
   the share of 16x16 and 4x4 blocks whose inter-frame difference never
   exceeds one grey level rises (`static_16`, `static_4`).

`sequence_features` computes these from the decoded frames; a standardised
logistic regression maps them to P(bona fide). The weights are trained by
evaluation/train_injection.py and stored as JSON (feature means/scales,
coefficients, intercept) in detection/weights/injection_lr.json - no
pickled objects are loaded.

Limitation (measured, see evaluation/results): a lossless source file at
the camera's own frame rate leaves neither duplicates nor a second
compression generation, and is not reliably detected.

Fallback (no weights file): the previous heuristic - robust frame-timing
jitter (0.65) plus 8x8 block energy of one frame (0.35). It is kept as the
baseline the classifier is compared against; it trusts client-reported
timestamps and is defeated by any attacker who adds plausible jitter.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from PIL import Image

from detection.base import DetectionResult, clip01

logger = logging.getLogger(__name__)

WEIGHTS_PATH = Path(__file__).parent / "weights" / "injection_lr.json"

FEATURE_NAMES = [
    "dup_rate",
    "log_min_diff",
    "log_temporal_noise",
    "diff_hf_ratio",
    "block_ratio_8",
    "block_ratio_16",
    "diff_block_ratio",
    "static_16",
    "static_4",
]
_DUP_THRESHOLD = 0.35  # mean |difference| in grey levels below which two frames count as the same picture
_MAX_SIDE = 720  # larger frames are centre-cropped (never resized: resizing destroys the block grid)

# Frame-interval jitter bands for a live camera feed. Jitter is a robust
# coefficient of variation (1.4826 * MAD / median interval), so a few late
# or dropped frames in an otherwise steady live stream don't dominate it.
_MIN_LIVE_JITTER = 0.002  # at or below: synthetic, exactly regular timestamps
_FULL_CREDIT_JITTER = 0.02
_MAX_LIVE_JITTER = 0.35  # above: increasingly erratic, reaching 0 at 1.0
_MIN_TIMING_INTERVALS = 8


def _grey(frame: np.ndarray) -> np.ndarray:
    g = np.asarray(Image.fromarray(np.asarray(frame, dtype=np.uint8)).convert("L"), dtype=np.float32)
    h, w = g.shape
    if max(h, w) > _MAX_SIDE:
        y0, x0 = max(0, (h - _MAX_SIDE) // 2) // 16 * 16, max(0, (w - _MAX_SIDE) // 2) // 16 * 16
        g = g[y0: y0 + _MAX_SIDE, x0: x0 + _MAX_SIDE]
    return g


def _grid_ratio(img: np.ndarray, period: int) -> float:
    """Mean |gradient| on block boundaries (every `period` px) over the mean
    elsewhere, averaged over both axes. ~1 for unblocked content."""
    ratios = []
    for axis in (0, 1):
        d = np.abs(np.diff(img, axis=axis))
        n = d.shape[axis]
        on = np.zeros(n, dtype=bool)
        on[period - 1:: period] = True
        on_e = d.take(np.flatnonzero(on), axis=axis).mean() if on.any() else 0.0
        off_e = d.take(np.flatnonzero(~on), axis=axis).mean() if (~on).any() else 1e-6
        ratios.append(on_e / (off_e + 1e-6))
    return float(np.mean(ratios))


def _highpass(img: np.ndarray) -> np.ndarray:
    return 4 * img[1:-1, 1:-1] - img[:-2, 1:-1] - img[2:, 1:-1] - img[1:-1, :-2] - img[1:-1, 2:]


def sequence_features(frames: list[np.ndarray]) -> dict[str, float]:
    """Feature vector of one submitted clip (>= 2 frames, temporal order)."""
    grey = [_grey(f) for f in frames]
    h = min(g.shape[0] for g in grey)
    w = min(g.shape[1] for g in grey)
    grey = [g[:h, :w] for g in grey]
    diffs = [b - a for a, b in zip(grey, grey[1:])]
    mean_abs = np.array([np.abs(d).mean() for d in diffs]) if diffs else np.array([0.0])

    moving = [d for d, m in zip(diffs, mean_abs) if m >= _DUP_THRESHOLD] or diffs
    h16, w16 = h // 16 * 16, w // 16 * 16
    static = {16: [], 4: []}
    for d in moving:
        for size in static:
            blocks = np.abs(d[:h16, :w16]).reshape(h16 // size, size, w16 // size, size).max(axis=(1, 3))
            static[size].append(float((blocks <= 1).mean()))
    if moving:
        hp = [_highpass(d) for d in moving]
        temporal_noise = float(np.median([np.median(np.abs(x)) for x in hp]))
        diff_hf = float(np.median([np.abs(x).mean() / (np.abs(d).mean() + 1e-6) for x, d in zip(hp, moving)]))
        diff_block = float(np.median([_grid_ratio(d, 8) for d in moving]))
    else:
        temporal_noise, diff_hf, diff_block = 0.0, 0.0, 1.0

    return {
        "dup_rate": float(np.mean(mean_abs < _DUP_THRESHOLD)) if diffs else 0.0,
        "log_min_diff": float(np.log10(mean_abs.min() + 1e-3)),
        "log_temporal_noise": float(np.log10(temporal_noise + 1e-3)),
        "diff_hf_ratio": diff_hf,
        "block_ratio_8": float(np.median([_grid_ratio(g, 8) for g in grey])),
        "block_ratio_16": float(np.median([_grid_ratio(g, 16) for g in grey])),
        "diff_block_ratio": diff_block,
        "static_16": float(np.mean(static[16])) if static[16] else 0.0,
        "static_4": float(np.mean(static[4])) if static[4] else 0.0,
    }


class LogisticModel:
    """Standardised logistic regression loaded from JSON."""

    def __init__(self, params: dict):
        if params["features"] != FEATURE_NAMES:
            raise ValueError(f"weights were trained on {params['features']}, expected {FEATURE_NAMES}")
        self.mean = np.asarray(params["mean"], dtype=np.float64)
        self.scale = np.asarray(params["scale"], dtype=np.float64)
        self.coef = np.asarray(params["coef"], dtype=np.float64)
        self.intercept = float(params["intercept"])

    def p_bona_fide(self, features: dict[str, float]) -> float:
        x = (np.array([features[k] for k in FEATURE_NAMES]) - self.mean) / self.scale
        return float(1.0 / (1.0 + np.exp(-(x @ self.coef + self.intercept))))


class InjectionDetector:
    module_name = "injection"

    def __init__(self, mode: str = "pretrained", device: str = "cpu"):
        self.device = device
        self.mode = mode
        self._model = None
        if self.mode == "pretrained":
            self._model = self._try_load_model()
            if self._model is None:
                self.mode = "mock"

    def _try_load_model(self):
        try:
            if not WEIGHTS_PATH.exists():
                raise FileNotFoundError(f"{WEIGHTS_PATH} not found - run evaluation/train_injection.py")
            return LogisticModel(json.loads(WEIGHTS_PATH.read_text()))
        except Exception as exc:
            logger.warning("Injection: trained classifier unavailable (%s); using heuristic-only scoring.", exc)
            return None

    def predict_sequence(self, frames: list[np.ndarray], frame_timestamps: list[float] | None = None) -> DetectionResult:
        """Score a whole submitted clip. Uses the trained classifier when
        loaded and the clip has >= 2 frames; otherwise the legacy heuristic
        on the first frame (+ timestamps)."""
        if self._model is None or len(frames) < 2:
            return self.predict(frames[0], frame_timestamps=frame_timestamps)
        features = sequence_features(frames)
        return DetectionResult(
            module=self.module_name,
            score=self._model.p_bona_fide(features),
            is_mock=False,
            details={**features, "num_frames": len(frames)},
        )

    def predict(self, frame: np.ndarray, frame_timestamps: list[float] | None = None) -> DetectionResult:
        """Legacy single-frame heuristic (fallback and baseline)."""
        compression_score, compression_details = self._compression_artifact_heuristic(frame)
        timing_score, timing_details = self._frame_timing_heuristic(frame_timestamps)

        # Timing is the stronger, better-grounded signal when available;
        # compression-artifact proxy is weak on its own (see module docstring).
        if frame_timestamps:
            score = clip01(0.65 * timing_score + 0.35 * compression_score)
        else:
            score = compression_score

        return DetectionResult(
            module=self.module_name,
            score=score,
            is_mock=True,
            details={**compression_details, **timing_details},
        )

    def _compression_artifact_heuristic(self, frame: np.ndarray) -> tuple[float, dict]:
        """8x8 block-boundary energy: re-encoding a screen-recorded or
        virtual-camera stream through a second lossy codec tends to leave
        elevated gradient energy exactly at 8-pixel block boundaries
        (classic JPEG/MPEG block-artifact signature)."""
        gray = np.asarray(Image.fromarray(frame).convert("L"), dtype=np.float64)
        h, w = gray.shape

        col_diffs = np.abs(np.diff(gray, axis=1))
        row_diffs = np.abs(np.diff(gray, axis=0))

        block_cols = [c for c in range(7, w - 1, 8)]
        block_rows = [r for r in range(7, h - 1, 8)]

        block_col_energy = float(col_diffs[:, block_cols].mean()) if block_cols else 0.0
        block_row_energy = float(row_diffs[block_rows, :].mean()) if block_rows else 0.0
        overall_col_energy = float(col_diffs.mean()) if col_diffs.size else 1e-6
        overall_row_energy = float(row_diffs.mean()) if row_diffs.size else 1e-6

        block_ratio = ((block_col_energy / (overall_col_energy or 1e-6)) + (block_row_energy / (overall_row_energy or 1e-6))) / 2

        # block_ratio near 1.0 = no unusual block structure (genuine);
        # elevated ratio = re-encoding artifact (injected/replayed stream).
        score = clip01(1.0 - (block_ratio - 1.1) / 1.5)
        return score, {"block_boundary_energy_ratio": block_ratio}

    def _frame_timing_heuristic(self, frame_timestamps: list[float] | None) -> tuple[float, dict]:
        # With only a handful of intervals, live jitter can read as "machine-
        # perfect" by chance (~2% of 4-interval live sequences did), so
        # timing is only judged on longer sequences. Live recordings send
        # every frame's timestamp (~100+); video files send all of theirs.
        if not frame_timestamps or len(frame_timestamps) < _MIN_TIMING_INTERVALS + 1:
            return 1.0, {"timing_checked": False}

        intervals = np.diff(frame_timestamps)
        median = float(np.median(intervals))
        if median <= 0:
            return 1.0, {"timing_checked": False}

        jitter_ratio = float(1.4826 * np.median(np.abs(intervals - median)) / median)

        # Real camera capture has some scheduling jitter (typically a few
        # percent of the frame interval). Both ends are penalized:
        #  - near-zero jitter: timestamps synthesized from a container's
        #    nominal fps (a pre-recorded file or virtual camera) are exact.
        #    The previous symmetric curve still gave jitter 0 a score of
        #    0.75, so injected streams passed; this ramp gives it 0.
        #  - very high jitter: dropped/duplicated frames from a virtual camera.
        low_side = (jitter_ratio - _MIN_LIVE_JITTER) / (_FULL_CREDIT_JITTER - _MIN_LIVE_JITTER)
        high_side = 1.0 - (jitter_ratio - _MAX_LIVE_JITTER) / (1.0 - _MAX_LIVE_JITTER)
        score = clip01(min(low_side, high_side))
        return score, {"timing_checked": True, "frame_interval_jitter_ratio": jitter_ratio}
