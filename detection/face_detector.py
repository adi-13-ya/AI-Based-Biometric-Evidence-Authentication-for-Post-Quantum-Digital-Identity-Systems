"""Face localisation for the learned detectors (PAD, deepfake).

Uses the same RetinaFace (Caffe, WIDER FACE) detector that the
Silent-Face-Anti-Spoofing release ships with MiniFASNet, run through
OpenCV's DNN module, so PAD crops are produced exactly the way the
released checkpoints were trained on. Files (see detection/README-models.md):

    detection/weights/minifasnet/deploy.prototxt
    detection/weights/minifasnet/Widerface-RetinaFace.caffemodel

`get_bbox` follows the reference implementation: downscale large frames to
about 192x192 pixels of area, take the highest-confidence box. Unlike the
reference it also returns that confidence, so callers can tell "no face"
apart from "a face that looks spoofed".
"""
from __future__ import annotations

import math
import threading
from pathlib import Path

import numpy as np

WEIGHTS_DIR = Path(__file__).parent / "weights" / "minifasnet"
PROTOTXT = WEIGHTS_DIR / "deploy.prototxt"
CAFFEMODEL = WEIGHTS_DIR / "Widerface-RetinaFace.caffemodel"


class FaceDetector:
    def __init__(self):
        import cv2

        if not (PROTOTXT.exists() and CAFFEMODEL.exists()):
            raise FileNotFoundError(f"RetinaFace files missing in {WEIGHTS_DIR} - see detection/README-models.md")
        self._cv2 = cv2
        self._net = cv2.dnn.readNetFromCaffe(str(PROTOTXT), str(CAFFEMODEL))
        # cv2.dnn.Net is not safe to call from several threads at once.
        self._lock = threading.Lock()

    def get_bbox(self, bgr: np.ndarray) -> tuple[list[int], float]:
        """Return ([x, y, w, h], confidence) of the most confident face."""
        cv2 = self._cv2
        height, width = bgr.shape[:2]
        img = bgr
        if width * height >= 192 * 192:
            aspect = width / height
            img = cv2.resize(bgr, (int(192 * math.sqrt(aspect)), int(192 / math.sqrt(aspect))), interpolation=cv2.INTER_LINEAR)
        blob = cv2.dnn.blobFromImage(img, 1, mean=(104, 117, 123))
        with self._lock:
            self._net.setInput(blob, "data")
            out = self._net.forward("detection_out").squeeze()
        out = out.reshape(-1, 7)
        best = int(np.argmax(out[:, 2]))
        conf = float(out[best, 2])
        left, top = out[best, 3] * width, out[best, 4] * height
        right, bottom = out[best, 5] * width, out[best, 6] * height
        return [int(left), int(top), int(right - left + 1), int(bottom - top + 1)], conf


def crop_scaled(img: np.ndarray, bbox: list[int], scale: float, out_w: int, out_h: int) -> np.ndarray:
    """Crop `scale` times the face box around its centre, shifted to stay
    inside the image, and resize. Same geometry as the reference
    CropImage._get_new_box, so MiniFASNet sees the context it was trained on
    (2.7x and 4.0x the face box)."""
    import cv2

    x0, y0, x1, y1 = crop_box(img.shape, bbox, scale)
    return cv2.resize(img[y0: y1 + 1, x0: x1 + 1], (out_w, out_h))


def crop_box(shape: tuple, bbox: list[int], scale: float) -> tuple[int, int, int, int]:
    """Inclusive pixel box (x0, y0, x1, y1) that crop_scaled cuts out."""
    src_h, src_w = shape[:2]
    x, y, box_w, box_h = bbox
    box_w, box_h = max(box_w, 1), max(box_h, 1)
    scale = min((src_h - 1) / box_h, (src_w - 1) / box_w, scale)
    new_w, new_h = box_w * scale, box_h * scale
    cx, cy = x + box_w / 2, y + box_h / 2
    x0, y0, x1, y1 = cx - new_w / 2, cy - new_h / 2, cx + new_w / 2, cy + new_h / 2
    if x0 < 0:
        x1 -= x0
        x0 = 0
    if y0 < 0:
        y1 -= y0
        y0 = 0
    if x1 > src_w - 1:
        x0 -= x1 - src_w + 1
        x1 = src_w - 1
    if y1 > src_h - 1:
        y0 -= y1 - src_h + 1
        y1 = src_h - 1
    return int(max(0, x0)), int(max(0, y0)), int(x1), int(y1)


_shared: FaceDetector | None = None
_shared_lock = threading.Lock()


def shared_detector() -> FaceDetector:
    """One process-wide detector (loading the Caffe model takes ~50 ms)."""
    global _shared
    with _shared_lock:
        if _shared is None:
            _shared = FaceDetector()
        return _shared
