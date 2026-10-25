"""Presentation Attack Detection (PAD) - spec Section 6.1.

Pretrained model: MiniFASNet from Silent-Face-Anti-Spoofing (Minivision,
Apache-2.0), the two released checkpoints used as an ensemble exactly as
the reference test.py does:

    2.7_80x80_MiniFASNetV2.pth      face box enlarged 2.7x, resized to 80x80
    4_0_0_80x80_MiniFASNetV1SE.pth  face box enlarged 4.0x, resized to 80x80

Each network outputs softmax over {0: print attack, 1: real, 2: replay
attack}; the module score is the mean P(real) of the two, so 0 = attack and
1 = bona fide as everywhere else in detection/. Faces are located with the
RetinaFace detector shipped in the same release (detection/face_detector.py).
Input is BGR, unnormalised 0-255, matching the reference transform (its
to_tensor deliberately does not divide by 255).

The operating threshold is not 0.5 by assumption: it is the EER point on
CASIA-FASD development subjects (0.57, evaluation/benchmark_pad.py), read
from THRESHOLD_PRESENTATION_ATTACK. The released checkpoints are the
default: fine-tuning on CASIA-FASD (evaluation/finetune_pad.py) raised
intra-dataset AUC but lowered cross-dataset AUC on CelebA-Spoof and webcam
video, so PAD_WEIGHTS=finetuned is kept only for comparison.

Fallback (`mode="mock"`, or the weights missing): the earlier three-cue
recapture heuristic (black level, face sharpness, saturation). It is kept
only so the server still starts without the checkpoint; every result it
produces is flagged is_mock=True and a warning is logged.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image

from detection.base import DetectionResult, clip01
from detection.face_region import FaceRegionFinder

logger = logging.getLogger(__name__)

WEIGHTS_DIR = Path(__file__).parent / "weights" / "minifasnet"
# Same two networks after fine-tuning on CelebA-Spoof train (evaluation/finetune_pad.py).
FINETUNED_DIR = Path(__file__).parent / "weights" / "minifasnet_finetuned"
# (checkpoint file, architecture name, crop scale)
CHECKPOINTS = [
    ("2.7_80x80_MiniFASNetV2.pth", "MiniFASNetV2", 2.7),
    ("4_0_0_80x80_MiniFASNetV1SE.pth", "MiniFASNetV1SE", 4.0),
]
_INPUT_SIZE = 80
_REAL_CLASS = 1

_FACE_WIDTH = 160  # face region is resized to this width before measuring sharpness
# (attack-like, live-like) ramps for each cue
_BLACK_LEVEL_RAMP = (40.0, 15.0)  # 1st-percentile face luminance, 0-255
_SHARPNESS_RAMP = (np.log10(8.0), np.log10(25.0))  # log10 Laplacian variance
_SATURATION_RAMP = (45.0, 75.0)  # mean HSV saturation, 0-255
_WEIGHTS = {"black_level": 0.5, "sharpness": 0.3, "saturation": 0.2}


def _ramp(x: float, attack_live: tuple[float, float]) -> float:
    attack, live = attack_live
    return clip01((x - attack) / (live - attack))


class PresentationAttackDetector:
    module_name = "presentation_attack"

    def __init__(self, mode: str = "pretrained", device: str = "cpu", weights: str = "released"):
        """`weights`: "released" (default) or "finetuned" (falls back to "released" if not present)."""
        self.device = device
        self.mode = mode
        self.weights = weights
        self._model = None
        # `mode="mock"` also skips MediaPipe face cropping (central crop only).
        self._faces = FaceRegionFinder(use_mediapipe=mode != "mock", owner="PAD")
        if self.mode == "pretrained":
            self._model = self._try_load_model()
            if self._model is None:
                self.mode = "mock"

    def _try_load_model(self):
        try:
            # Check for weights first: importing torch is slow, and pointless with no checkpoint.
            weights_dir = WEIGHTS_DIR
            if self.weights == "finetuned":
                if all((FINETUNED_DIR / f).exists() for f, _, _ in CHECKPOINTS):
                    weights_dir = FINETUNED_DIR
                else:
                    logger.warning("PAD: fine-tuned MiniFASNet not found in %s; using the released checkpoints.", FINETUNED_DIR)
                    self.weights = "released"
            missing = [f for f, _, _ in CHECKPOINTS if not (weights_dir / f).exists()]
            if missing:
                raise FileNotFoundError(f"{missing} not found in {weights_dir} - see detection/README-models.md")
            import torch

            from detection.architectures import minifasnet
            from detection.face_detector import shared_detector

            self._face_detector = shared_detector()
            models = []
            for filename, arch, scale in CHECKPOINTS:
                net = getattr(minifasnet, arch)(conv6_kernel=((_INPUT_SIZE + 15) // 16, (_INPUT_SIZE + 15) // 16))
                state = torch.load(weights_dir / filename, map_location=self.device, weights_only=True)
                net.load_state_dict({k.removeprefix("module."): v for k, v in state.items()})
                net.to(self.device).eval()
                models.append((net, scale))
            return models
        except Exception as exc:
            logger.warning("PAD: pretrained MiniFASNet unavailable (%s); using mock heuristic.", exc)
            return None

    def predict(self, frame: np.ndarray) -> DetectionResult:
        if self.mode == "mock" or self._model is None:
            return self._predict_mock(frame)
        return self._predict_pretrained(frame)

    def _predict_mock(self, frame: np.ndarray) -> DetectionResult:
        region, crop_details = self._faces.crop(np.asarray(Image.fromarray(frame).convert("RGB")))
        face = Image.fromarray(region)

        luminance = np.asarray(face.convert("L"), dtype=np.float64)
        black_level = float(np.percentile(luminance, 1))

        height = max(8, round(face.height * _FACE_WIDTH / face.width))
        gray = np.asarray(face.convert("L").resize((_FACE_WIDTH, height), Image.BILINEAR), dtype=np.float64)
        # Manual 3x3 Laplacian (no OpenCV/scipy dependency in the mock path).
        lap = -4 * gray[1:-1, 1:-1] + gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:]
        sharpness = float(lap.var()) if lap.size else 0.0

        saturation = float(np.asarray(face.convert("HSV"), dtype=np.float64)[..., 1].mean())

        cues = {
            "black_level": _ramp(black_level, _BLACK_LEVEL_RAMP),
            "sharpness": _ramp(np.log10(max(sharpness, 1e-6)), _SHARPNESS_RAMP),
            "saturation": _ramp(saturation, _SATURATION_RAMP),
        }
        score = sum(_WEIGHTS[k] * v for k, v in cues.items())

        return DetectionResult(
            module=self.module_name,
            score=score,
            is_mock=True,
            details={
                "face_black_level": black_level,
                "face_sharpness": sharpness,
                "face_saturation": saturation,
                **crop_details,
            },
        )

    def _predict_pretrained(self, frame: np.ndarray) -> DetectionResult:
        import torch

        from detection.face_detector import crop_scaled

        bgr = np.ascontiguousarray(np.asarray(frame)[..., 2::-1])  # RGB -> BGR, as the checkpoints expect
        bbox, face_conf = self._face_detector.get_bbox(bgr)
        batch = np.stack([crop_scaled(bgr, bbox, scale, _INPUT_SIZE, _INPUT_SIZE) for _, scale in self._model])
        tensor = torch.from_numpy(batch.transpose(0, 3, 1, 2).astype(np.float32)).to(self.device)
        probs = []
        with torch.no_grad():
            for i, (net, _) in enumerate(self._model):
                probs.append(torch.softmax(net(tensor[i: i + 1]), dim=1)[0].cpu().numpy())
        mean = np.mean(probs, axis=0)
        return DetectionResult(
            module=self.module_name,
            score=float(mean[_REAL_CLASS]),
            is_mock=False,
            details={
                "p_real": float(mean[_REAL_CLASS]),
                "p_print": float(mean[0]),
                "p_replay": float(mean[2]),
                "face_confidence": face_conf,
                "face_bbox": bbox,
                "checkpoint": self.weights,
            },
        )
