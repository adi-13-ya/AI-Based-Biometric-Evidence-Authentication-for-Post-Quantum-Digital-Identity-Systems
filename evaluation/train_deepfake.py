"""Fine-tune an ImageNet-pretrained EfficientNet-B0 as the deepfake detector,
and evaluate it on held-out identities and unseen generators.

Data: DeepFakeFace (DFF; OpenRL/DeepFakeFace, Song et al. 2023). 30,000 real
IMDB-WIKI portraits (wiki.zip) and, for the same file names, 30,000 fakes
from each of three generators:
    insight     InsightFace face swap           (training generator)
    inpainting  Stable Diffusion inpainting     (unseen at training time)
    text2img    Stable Diffusion v1.5 text2img  (unseen at training time)

Split: by IMDB-WIKI person id (the number before the first "_" in the file
name), hashed into train 70% / dev 10% / test 20%, so no person appears in
two splits. Sampled per split: train 800 real + 800 insight; dev 200 + 200;
test 400 real + 400 of each generator (evaluation/fetch_data.py downloads
exactly these members; the sample is small because the archives total 5 GB).

Preprocessing (identical to detection/deepfake_detector.py): RetinaFace box
enlarged 1.3x, square crop, 160x160, ImageNet normalisation. Training
augmentation: horizontal flip, random JPEG re-compression (quality 60-95,
p = 0.5) so the model does not key on one encoder. AdamW lr 3e-4, wd 1e-4,
batch 64, 6 epochs, best dev-AUC epoch kept.

Outputs:
    detection/weights/deepfake_effb0_dff.pth
    datasets/cache/dff_*.npz                    (face crops)
    evaluation/results/deepfake_train_log.json
"""
from __future__ import annotations

import hashlib
import io
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from evaluation.metrics import auc  # noqa: E402

DFF = REPO / "datasets" / "dff"
CACHE = REPO / "datasets" / "cache"
GENERATORS = ["insight", "inpainting", "text2img"]
PER_SPLIT = {"train": (800, {"insight": 800}), "dev": (200, {"insight": 200}),
             "test": (400, {g: 400 for g in GENERATORS})}
EPOCHS = 6
BATCH = 64
SEED = 2026


def person_split(name: str) -> str:
    person = Path(name).name.split("_")[0]
    h = int(hashlib.sha256(person.encode()).hexdigest(), 16) % 100
    return "train" if h < 70 else "dev" if h < 80 else "test"


def _members(zf: zipfile.ZipFile) -> dict[str, str]:
    """relative key (e.g. '00/123_1980-01-01_2005.jpg') -> member name."""
    out = {}
    for m in zf.namelist():
        if m.lower().endswith((".jpg", ".png")):
            parts = Path(m).parts
            out["/".join(parts[-2:])] = m
    return out


def build_split(split: str) -> dict:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"dff_{split}.npz"
    if path.exists():
        return dict(np.load(path, allow_pickle=False))
    from PIL import Image

    from detection.deepfake_detector import face_crop
    from detection.face_detector import shared_detector

    det = shared_detector()
    rng = np.random.default_rng(SEED + {"train": 1, "dev": 2, "test": 3}[split])
    zips = {name: zipfile.ZipFile(DFF / f"{name}.zip") for name in ["wiki", *GENERATORS]}
    members = {name: _members(z) for name, z in zips.items()}
    keys = sorted(k for k in members["wiki"] if person_split(k) == split)
    n_real, n_fake = PER_SPLIT[split]
    picks = [("wiki", k) for k in rng.choice(keys, size=min(n_real, len(keys)), replace=False)]
    for gen, n in n_fake.items():
        avail = sorted(k for k in members[gen] if person_split(k) == split)
        picks += [(gen, k) for k in rng.choice(avail, size=min(n, len(avail)), replace=False)]
    crops, labels, sources, keys_out = [], [], [], []
    for i, (src, key) in enumerate(picks):
        rgb = np.asarray(Image.open(io.BytesIO(zips[src].read(members[src][key]))).convert("RGB"))
        crop, _ = face_crop(det, rgb)
        crops.append(crop)
        labels.append(src == "wiki")
        sources.append(src)
        keys_out.append(key)
        if (i + 1) % 1000 == 0:
            print(f"  dff {split}: {i + 1}/{len(picks)}")
    out = {"crops": np.stack(crops), "labels": np.array(labels), "sources": np.array(sources), "keys": np.array(keys_out)}
    np.savez(path, **out)
    return out


def scores(model, crops: np.ndarray, device, batch: int = 256) -> np.ndarray:
    import torch

    from detection.deepfake_detector import to_tensor

    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(crops), batch):
            out.append(torch.sigmoid(model(to_tensor(crops[i: i + batch]).to(device))).squeeze(1).cpu().numpy())
    return np.concatenate(out)


def jpeg(img: np.ndarray, quality: int) -> np.ndarray:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=int(quality))
    return np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"))


def main() -> None:
    import torch

    from detection.deepfake_detector import WEIGHTS_PATH, build_model, to_tensor

    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    train, dev = build_split("train"), build_split("dev")
    build_split("test")  # cache now so the benchmark only reads it
    print(f"train {len(train['labels'])}, dev {len(dev['labels'])}, device {device}")

    model = build_model(pretrained=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    log = {"epochs": [], "n_train": int(len(train["labels"])), "n_dev": int(len(dev["labels"]))}
    best, best_state = -1.0, None
    y = train["labels"].astype(np.float32)  # 1 = real: the logit is log-odds of bona fide
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()
        order = rng.permutation(len(y))
        total = 0.0
        for i in range(0, len(order), BATCH):
            idx = order[i: i + BATCH]
            batch = train["crops"][idx].copy()
            flip = rng.random(len(idx)) < 0.5
            batch[flip] = batch[flip, :, ::-1]
            for j in np.flatnonzero(rng.random(len(idx)) < 0.5):
                batch[j] = jpeg(batch[j], rng.integers(60, 96))
            logits = model(to_tensor(batch).to(device)).squeeze(1)
            loss = loss_fn(logits, torch.from_numpy(y[idx]).to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss) * len(idx)
        dev_auc = auc(scores(model, dev["crops"], device), dev["labels"])
        log["epochs"].append({"epoch": epoch, "train_loss": total / len(y), "dev_auc": dev_auc, "seconds": time.time() - t0})
        print(f"epoch {epoch}: loss {total / len(y):.4f} dev AUC {dev_auc:.4f} ({time.time() - t0:.0f}s)")
        if dev_auc > best:
            best, log["best_epoch"] = dev_auc, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, WEIGHTS_PATH)
    log["best_dev_auc"] = best
    (REPO / "evaluation" / "results").mkdir(parents=True, exist_ok=True)
    (REPO / "evaluation" / "results" / "deepfake_train_log.json").write_text(json.dumps(log, indent=2))
    print(f"saved epoch {log['best_epoch']} (dev AUC {best:.4f}) to {WEIGHTS_PATH}")


if __name__ == "__main__":
    main()
