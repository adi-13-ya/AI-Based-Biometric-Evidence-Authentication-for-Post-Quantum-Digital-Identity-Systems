"""Public PAD datasets -> cached MiniFASNet input crops.

Every image is run once through the deployed preprocessing
(detection/face_detector.py: RetinaFace box, then 2.7x and 4.0x crops at
80x80, BGR) and the result is cached as .npz under datasets/cache/, so the
benchmark and the fine-tuning script score exactly what the server would
see. The legacy heuristic needs the full frame, so its score is computed in
the same pass and cached too.

Sources (all public, see datasets/download_samples.md):
    casia_train    CASIA-FASD train frames, subjects 1-16      (fine-tuning)
    casia_dev      CASIA-FASD train frames, subjects 17-20     (threshold / model selection)
    casia_test     CASIA-FASD test frames, 30 further subjects (intra-dataset test)
    celeba_test    CelebA-Spoof test split, 2 parquet shards   (cross-dataset test)
    webcam         UniqueData web-camera liveness sample, 5 frames per video (cross-dataset test)

CASIA-FASD here is the akahana/anti-spoofing-casiafasd copy: frames sampled
from the original videos and cropped to 256x256 around the face, with the
original train (20 subjects) / test (30 subjects) partition. Development
data is carved out of train by subject, so no test subject is ever used to
choose a threshold or an epoch.

Species labels: CelebA-Spoof's public parquet only says live/spoof, so its
attacks form one "spoof" species. CASIA-FASD encodes the attack in the clip
id: 1, 2, HR_1 bona fide; 3, 4, HR_2 warped photo; 5, 6, HR_3 cut photo;
7, 8, HR_4 video replay. The webcam set uses its folder names.
"""
from __future__ import annotations

import io
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

DATASETS = REPO / "datasets"
CACHE = DATASETS / "cache"

CASIA_SPECIES = {
    "1": "bona_fide", "2": "bona_fide", "HR_1": "bona_fide",
    "3": "warped_photo", "4": "warped_photo", "HR_2": "warped_photo",
    "5": "cut_photo", "6": "cut_photo", "HR_3": "cut_photo",
    "7": "video_replay", "8": "video_replay", "HR_4": "video_replay",
}
_CASIA_RE = re.compile(r"^(\d+)_(HR_\d|\d)\.avi_(\d+)_(real|fake)\.jpg$")
WEBCAM_FRAMES_PER_VIDEO = 5


def _celeba(pattern: str):
    import pyarrow.parquet as pq
    from PIL import Image

    for shard in sorted((DATASETS / "celeba_spoof").glob(pattern)):
        table = pq.read_table(shard).to_pandas()
        for i, row in table.iterrows():
            img = np.asarray(Image.open(io.BytesIO(row.Filepath["bytes"])).convert("RGB"))
            live = row.Class == "live"
            yield f"{shard.stem}:{i}", img, live, "bona_fide" if live else "spoof", f"{shard.stem}:{i}"


CASIA_DEV_SUBJECTS = {17, 18, 19, 20}


def _casia(split: str):
    from PIL import Image

    folder = "test_img" if split == "test" else "train_img"
    for path in sorted((DATASETS / "casia_fasd" / folder / folder / "color").glob("*.jpg")):
        m = _CASIA_RE.match(path.name)
        if not m:
            continue
        subject, clip = m.group(1), m.group(2)
        if split != "test" and (int(subject) in CASIA_DEV_SUBJECTS) != (split == "dev"):
            continue
        species = CASIA_SPECIES[clip]
        yield path.name, np.asarray(Image.open(path).convert("RGB")), species == "bona_fide", species, f"{subject}_{clip}"


def _webcam():
    import av

    for path in sorted((DATASETS / "webcam_liveness" / "files").glob("*/*.mp4")):
        with av.open(str(path)) as c:
            frames = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
        idx = np.linspace(0, len(frames) - 1, WEBCAM_FRAMES_PER_VIDEO).round().astype(int)
        species = path.parent.name
        for k in idx:
            yield f"{species}/{path.stem}:{k}", frames[k], species == "real", "bona_fide" if species == "real" else species, f"{species}/{path.stem}"


SOURCES = {
    "casia_train": lambda: _casia("train"),
    "casia_dev": lambda: _casia("dev"),
    "casia_test": lambda: _casia("test"),
    "celeba_test": lambda: _celeba("test-*.parquet"),
    "webcam": _webcam,
}


def build(name: str, with_legacy: bool = True) -> dict:
    """Return (and cache) crops/labels for dataset `name`."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"pad_{name}.npz"
    if path.exists():
        return dict(np.load(path, allow_pickle=False))

    from detection.face_detector import crop_scaled, shared_detector
    from detection.presentation_attack import PresentationAttackDetector

    det = shared_detector()
    legacy = PresentationAttackDetector(mode="mock") if with_legacy else None
    rec = {k: [] for k in ("ids", "groups", "labels", "species", "crop27", "crop40", "face_conf", "legacy")}
    for n, (sid, rgb, live, species, group) in enumerate(SOURCES[name]()):
        bgr = np.ascontiguousarray(rgb[..., ::-1])
        bbox, conf = det.get_bbox(bgr)
        rec["ids"].append(sid)
        rec["groups"].append(group)
        rec["labels"].append(live)
        rec["species"].append(species)
        rec["crop27"].append(crop_scaled(bgr, bbox, 2.7, 80, 80))
        rec["crop40"].append(crop_scaled(bgr, bbox, 4.0, 80, 80))
        rec["face_conf"].append(conf)
        rec["legacy"].append(legacy.predict(rgb).score if legacy else np.nan)
        if (n + 1) % 500 == 0:
            print(f"  {name}: {n + 1} images")
    out = {
        "ids": np.array(rec["ids"]),
        "groups": np.array(rec["groups"]),
        "labels": np.array(rec["labels"], dtype=bool),
        "species": np.array(rec["species"]),
        "crop27": np.stack(rec["crop27"]).astype(np.uint8),
        "crop40": np.stack(rec["crop40"]).astype(np.uint8),
        "face_conf": np.array(rec["face_conf"], dtype=np.float32),
        "legacy": np.array(rec["legacy"], dtype=np.float32),
    }
    np.savez_compressed(path, **out)
    print(f"cached {name}: {len(out['labels'])} images ({out['labels'].sum()} bona fide) -> {path}")
    return out


if __name__ == "__main__":
    for name in sys.argv[1:] or SOURCES:
        build(name)
