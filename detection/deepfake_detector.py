"""Deepfake Detection - spec Section 6.3.

Model: EfficientNet-B0 (torchvision, ImageNet-1k pretrained) with a single
logit head, fine-tuned on DeepFakeFace face-swap data by
evaluation/train_deepfake.py (identity-disjoint split; see that file for the
full protocol). The logit is the log-odds that the face is real, so
sigmoid(logit) is directly the module score (0 = fake, 1 = real).

Preprocessing: RetinaFace box (detection/face_detector.py) enlarged 1.3x,
square crop, resized to 160x160, ImageNet mean/std normalisation. A clip is
scored as the mean of its frames' scores: the per-frame threshold is
calibrated on single images, and the minimum of several frames would bias
bona fide clips downward as the number of frames grows.

Fallback (`mode="mock"`, or no weights / torchvision): the earlier 2-D FFT
high-frequency-energy heuristic (Frank et al., ICML 2020 motivate the cue).
It is kept so the server starts without the checkpoint; its results are
flagged is_mock=True, and evaluation shows it is not a usable detector on
real faces.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image

from detection.base import DetectionResult, clip01

logger = logging.getLogger(__name__)

WEIGHTS_PATH = Path(__file__).parent / "weights" / "deepfake_effb0_dff.pth"
INPUT_SIZE = 160
FACE_SCALE = 1.3
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def build_model(pretrained: bool = False):
    """EfficientNet-B0 with a 1-logit head (log-odds of bona fide)."""
    import torch.nn as nn
    import torchvision

    model = torchvision.models.efficientnet_b0(weights="IMAGENET1K_V1" if pretrained else None)
    model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(model.classifier[1].in_features, 1))
    return model


def face_crop(detector, rgb: np.ndarray) -> tuple[np.ndarray, float]:
    """Square face crop (RGB uint8, INPUT_SIZE x INPUT_SIZE) and the face
    detector's confidence."""
    from detection.face_detector import crop_scaled

    bgr = np.ascontiguousarray(rgb[..., ::-1])
    (x, y, w, h), conf = detector.get_bbox(bgr)
    side = max(w, h)
    square = [int(x + w / 2 - side / 2), int(y + h / 2 - side / 2), side, side]
    return crop_scaled(rgb, square, FACE_SCALE, INPUT_SIZE, INPUT_SIZE), conf


def to_tensor(crops: np.ndarray):
    """uint8 (N, H, W, 3) RGB -> normalised float tensor (N, 3, H, W)."""
    import torch

    x = (np.asarray(crops, dtype=np.float32) / 255.0 - _MEAN) / _STD
    return torch.from_numpy(np.ascontiguousarray(x.transpose(0, 3, 1, 2)))


class DeepfakeDetector:
    module_name = "deepfake"

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
            # Check for weights first: importing torch is slow, and pointless with no checkpoint.
            if not WEIGHTS_PATH.exists():
                raise FileNotFoundError(f"{WEIGHTS_PATH} not found - run evaluation/train_deepfake.py")
            import torch

            from detection.face_detector import shared_detector

            self._face_detector = shared_detector()
            model = build_model(pretrained=False)
            model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=self.device, weights_only=True))
            model.to(self.device).eval()
            return model
        except Exception as exc:
            logger.warning("Deepfake: pretrained EfficientNet-B0 unavailable (%s); using mock FFT heuristic.", exc)
            return None

    def predict(self, frame: np.ndarray) -> DetectionResult:
        if self.mode == "mock" or self._model is None:
            return self._predict_mock(frame)
        return self._predict_pretrained([frame])

    def predict_video(self, frames: list[np.ndarray]) -> DetectionResult:
        """Score a clip: mean per-frame score for the model, minimum for the
        mock heuristic (its historical behaviour)."""
        if not frames:
            raise ValueError("predict_video requires at least one frame")
        if self.mode != "mock" and self._model is not None:
            return self._predict_pretrained(frames)
        results = [self._predict_mock(f) for f in frames]
        worst = min(results, key=lambda r: r.score)
        return DetectionResult(
            module=self.module_name,
            score=worst.score,
            is_mock=True,
            details={**worst.details, "num_frames": len(results), "per_frame_scores": [r.score for r in results]},
        )

    def _predict_pretrained(self, frames: list[np.ndarray]) -> DetectionResult:
        import torch

        crops, confs = zip(*(face_crop(self._face_detector, np.asarray(f)[..., :3]) for f in frames))
        with torch.no_grad():
            probs = torch.sigmoid(self._model(to_tensor(np.stack(crops)).to(self.device))).squeeze(1).cpu().numpy()
        return DetectionResult(
            module=self.module_name,
            score=float(probs.mean()),
            is_mock=False,
            details={
                "p_real": float(probs.mean()),
                "num_frames": len(frames),
                "per_frame_scores": [float(p) for p in probs],
                "face_confidence": float(min(confs)),
            },
        )

    def _predict_mock(self, frame: np.ndarray) -> DetectionResult:
        gray = np.asarray(Image.fromarray(frame).convert("L").resize((256, 256)), dtype=np.float64)
        spectrum = np.fft.fftshift(np.fft.fft2(gray))
        magnitude = np.abs(spectrum)

        h, w = magnitude.shape
        cy, cx = h // 2, w // 2
        radius = min(h, w) // 4
        yy, xx = np.ogrid[:h, :w]
        dist = np.hypot(yy - cy, xx - cx)

        low_freq_energy = float(magnitude[dist <= radius].sum())
        high_freq_energy = float(magnitude[dist > radius].sum())
        total = low_freq_energy + high_freq_energy or 1.0
        high_freq_ratio = high_freq_energy / total

        score = clip01(1.0 - (high_freq_ratio - 0.35) / 0.25)
        return DetectionResult(
            module=self.module_name,
            score=score,
            is_mock=True,
            details={"high_freq_ratio": high_freq_ratio},
        )
