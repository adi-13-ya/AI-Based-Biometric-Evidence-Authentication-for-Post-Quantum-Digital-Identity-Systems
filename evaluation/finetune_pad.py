"""Fine-tune the released MiniFASNet checkpoints on CASIA-FASD (train subjects 1-16).

Starts from the two released Silent-Face-Anti-Spoofing checkpoints and
keeps their architecture and 3-way head. The loss is binary over the head:
P(real) = softmax class 1 against the merged attack classes 0 (print) and
2 (replay), so print and replay attacks need no separate labels:

    loss = -w_y * log P(y),   P(spoof) = P(class 0) + P(class 2)

with class weights w inversely proportional to class frequency (CASIA train
is ~2.5:1 attack:bona fide). Augmentation: horizontal flip and +/-20%
contrast / +/-20 grey-level brightness jitter. Adam, lr 1e-4, weight decay
1e-4, batch 128, 6 epochs; the epoch with the best development-set (CASIA
train subjects 17-20) AUC of the two-model ensemble is kept. Test sets are
never touched here.

Writes detection/weights/minifasnet_finetuned/{same filenames as released}
and evaluation/results/pad_finetune_log.json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from evaluation.metrics import auc  # noqa: E402
from evaluation.pad_data import build  # noqa: E402

OUT_DIR = REPO / "detection" / "weights" / "minifasnet_finetuned"
EPOCHS = 6
BATCH = 128
SEED = 2026


def load_released(device):
    import torch

    from detection.architectures import minifasnet
    from detection.presentation_attack import CHECKPOINTS, WEIGHTS_DIR

    nets = []
    for filename, arch, _ in CHECKPOINTS:
        net = getattr(minifasnet, arch)(conv6_kernel=(5, 5))
        state = torch.load(WEIGHTS_DIR / filename, map_location="cpu", weights_only=True)
        net.load_state_dict({k.removeprefix("module."): v for k, v in state.items()})
        nets.append((filename, net.to(device)))
    return nets


def p_real(net, crops: np.ndarray, device, batch: int = 512) -> np.ndarray:
    import torch

    net.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(crops), batch):
            x = torch.from_numpy(crops[i: i + batch].transpose(0, 3, 1, 2).astype(np.float32)).to(device)
            out.append(torch.softmax(net(x), dim=1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def train_one(net, crops, labels, device, rng, class_w):
    import torch

    net.train()
    opt = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-4)
    order = rng.permutation(len(crops))
    total = 0.0
    for i in range(0, len(order), BATCH):
        idx = order[i: i + BATCH]
        x = crops[idx].astype(np.float32)
        flip = rng.random(len(idx)) < 0.5
        x[flip] = x[flip, :, ::-1]
        contrast = rng.uniform(0.8, 1.2, (len(idx), 1, 1, 1))
        bright = rng.uniform(-20, 20, (len(idx), 1, 1, 1))
        x = np.clip((x - 128) * contrast + 128 + bright, 0, 255).astype(np.float32)
        xt = torch.from_numpy(np.ascontiguousarray(x.transpose(0, 3, 1, 2))).to(device)
        y = torch.from_numpy(labels[idx].astype(np.int64)).to(device)
        logp = torch.log_softmax(net(xt), dim=1)
        log_real = logp[:, 1]
        log_spoof = torch.logsumexp(logp[:, [0, 2]], dim=1)
        w = torch.tensor(class_w, dtype=torch.float32, device=device)[y]
        loss = -(w * torch.where(y == 1, log_real, log_spoof)).sum() / w.sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += float(loss) * len(idx)
    return total / len(order)


def main() -> None:
    import torch

    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    train = build("casia_train", with_legacy=False)
    dev = build("casia_dev")  # with legacy scores: the benchmark reuses this cache
    y_tr = train["labels"].astype(int)
    class_w = [len(y_tr) / (2 * (y_tr == 0).sum()), len(y_tr) / (2 * (y_tr == 1).sum())]
    print(f"train {len(y_tr)} ({y_tr.sum()} live), dev {len(dev['labels'])} ({dev['labels'].sum()} live), device {device}")

    nets = load_released(device)
    crop_key = ["crop27", "crop40"]
    log = {"epochs": [], "class_weights": class_w, "n_train": int(len(y_tr)), "n_dev": int(len(dev["labels"]))}
    dev_scores = np.mean([p_real(n, dev[k], device) for (_, n), k in zip(nets, crop_key)], axis=0)
    best_auc = auc(dev_scores, dev["labels"])
    log["epochs"].append({"epoch": 0, "dev_auc": best_auc, "note": "released checkpoints"})
    print(f"epoch 0 (released): dev AUC {best_auc:.4f}")
    best_states = None
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        losses = [train_one(n, train[k], y_tr, device, rng, class_w) for (_, n), k in zip(nets, crop_key)]
        dev_scores = np.mean([p_real(n, dev[k], device) for (_, n), k in zip(nets, crop_key)], axis=0)
        dev_auc = auc(dev_scores, dev["labels"])
        log["epochs"].append({"epoch": epoch, "train_loss": losses, "dev_auc": dev_auc, "seconds": time.time() - t0})
        print(f"epoch {epoch}: loss {losses[0]:.4f}/{losses[1]:.4f} dev AUC {dev_auc:.4f} ({time.time() - t0:.0f}s)")
        if dev_auc > best_auc:
            best_auc = dev_auc
            best_states = [(f, {k: v.detach().cpu().clone() for k, v in n.state_dict().items()}) for f, n in nets]
            log["best_epoch"] = epoch
    log["best_dev_auc"] = best_auc
    if best_states is None:
        print("fine-tuning never beat the released checkpoints on dev; nothing saved")
        log["best_epoch"] = 0
    else:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        for filename, state in best_states:
            torch.save(state, OUT_DIR / filename)
        print(f"saved epoch {log['best_epoch']} to {OUT_DIR}")
    (REPO / "evaluation" / "results").mkdir(parents=True, exist_ok=True)
    (REPO / "evaluation" / "results" / "pad_finetune_log.json").write_text(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
