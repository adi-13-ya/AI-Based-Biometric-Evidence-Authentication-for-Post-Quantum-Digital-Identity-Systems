"""ISO/IEC 30107-3 evaluation of the PAD module on public datasets.

Systems compared (same cached face crops, evaluation/pad_data.py):
    legacy      a three-cue heuristic baseline (black level, sharpness, saturation)
    released    MiniFASNetV2 + MiniFASNetV1SE, released checkpoints (zero-shot)
    finetuned   the same two networks after evaluation/finetune_pad.py

Protocol. Each system's operating threshold is fixed once, on the CASIA-FASD
development subjects (train subjects 17-20), at that split's EER point, and
then applied unchanged to every test set: CASIA-FASD test (intra-dataset;
30 unseen subjects), CelebA-Spoof test and the webcam sample
(cross-dataset). The legacy heuristic is also
reported at the 0.5 threshold it was deployed with. Reported per test set:
APCER for each PAI species and the worst species (ISO/IEC 30107-3 clause
13), BPCER, 95% Wilson intervals, and threshold-free AUC, EER and BPCER at
APCER = 5% / 10% (the latter two use the test scores, so they describe the
curve, not a deployable operating point). The webcam sample is scored per
video (mean over 5 frames), matching how a clip is submitted.

Run: python evaluation/benchmark_pad.py
Writes evaluation/results/pad_benchmark.json and pad_scores_<dataset>.csv
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from evaluation.metrics import apcer_bpcer, apcer_per_species, auc, bpcer_at_apcer, eer, wilson_interval  # noqa: E402
from evaluation.pad_data import build  # noqa: E402

RESULTS = REPO / "evaluation" / "results"
TESTS = ["casia_test", "celeba_test", "webcam"]


def minifasnet_scores(weights_dir: Path, data: dict) -> tuple[np.ndarray, float]:
    """Mean P(real) of the two networks; also returns ms per image (CPU, batch 1)."""
    import torch

    from detection.architectures import minifasnet
    from detection.presentation_attack import CHECKPOINTS

    torch.set_num_threads(4)
    probs, nets = [], []
    for (filename, arch, _), key in zip(CHECKPOINTS, ["crop27", "crop40"]):
        net = getattr(minifasnet, arch)(conv6_kernel=(5, 5))
        state = torch.load(weights_dir / filename, map_location="cpu", weights_only=True)
        net.load_state_dict({k.removeprefix("module."): v for k, v in state.items()})
        net.eval()
        nets.append((net, key))
        x = torch.from_numpy(data[key].transpose(0, 3, 1, 2).astype(np.float32))
        with torch.no_grad():
            probs.append(np.concatenate([torch.softmax(net(x[i: i + 512]), 1)[:, 1].numpy() for i in range(0, len(x), 512)]))
    # single-image latency of the two networks (face detection excluded; see D4 for the full module)
    xs = [torch.from_numpy(data[k][:1].transpose(0, 3, 1, 2).astype(np.float32)) for _, k in nets]
    with torch.no_grad():
        for _ in range(5):
            [n(x) for (n, _), x in zip(nets, xs)]
        t0 = time.perf_counter()
        for _ in range(50):
            [n(x) for (n, _), x in zip(nets, xs)]
    return np.mean(probs, axis=0), (time.perf_counter() - t0) / 50 * 1000


def per_video(data: dict, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    groups = data["groups"]
    uniq = sorted(set(groups))
    idx = {g: np.flatnonzero(groups == g) for g in uniq}
    return (np.array([scores[idx[g]].mean() for g in uniq]),
            np.array([data["labels"][idx[g][0]] for g in uniq]),
            np.array([data["species"][idx[g][0]] for g in uniq]))


def report(scores, labels, species, threshold) -> dict:
    a, b = apcer_bpcer(scores, labels, threshold)
    n_att, n_bf = int((~labels).sum()), int(labels.sum())
    per = apcer_per_species(scores, labels, species, threshold)
    worst = max(per, key=per.get) if per else None
    return {
        "n_bona_fide": n_bf, "n_attack": n_att, "threshold": threshold,
        "apcer_per_species": per,
        "apcer_worst": per[worst] if worst else a, "apcer_worst_species": worst,
        "apcer_all": a, "apcer_all_ci": wilson_interval(round(a * n_att), n_att),
        "bpcer": b, "bpcer_ci": wilson_interval(round(b * n_bf), n_bf),
        "auc": auc(scores, labels), "eer": eer(scores, labels)[0],
        "bpcer_at_apcer5": bpcer_at_apcer(scores, labels, 0.05), "bpcer_at_apcer10": bpcer_at_apcer(scores, labels, 0.10),
    }


def main() -> None:
    from detection.presentation_attack import CHECKPOINTS, FINETUNED_DIR, WEIGHTS_DIR

    dev = build("casia_dev")
    tests = {name: build(name) for name in TESTS}
    systems = {"legacy": None, "released": WEIGHTS_DIR}
    if all((FINETUNED_DIR / f).exists() for f, _, _ in CHECKPOINTS):
        systems["finetuned"] = FINETUNED_DIR

    out = {"protocol": __doc__.split("Protocol.")[1].split("Run:")[0].strip(), "datasets": {}, "systems": {}}
    for name, d in {"casia_dev": dev, **tests}.items():
        out["datasets"][name] = {"images": int(len(d["labels"])), "bona_fide": int(d["labels"].sum()),
                                 "species": {s: int((d["species"] == s).sum()) for s in sorted(set(d["species"]))},
                                 "face_conf_below_0.6": float((d["face_conf"] < 0.6).mean())}

    score_cache = {}
    for sys_name, wdir in systems.items():
        res = {}
        if wdir is None:
            dev_scores, ms = dev["legacy"], None
            test_scores = {n: d["legacy"] for n, d in tests.items()}
        else:
            dev_scores, ms = minifasnet_scores(wdir, dev)
            test_scores = {n: minifasnet_scores(wdir, d)[0] for n, d in tests.items()}
        res["inference_ms_two_networks"] = ms
        dev_eer, t_dev = eer(dev_scores, dev["labels"])
        res["dev"] = {"eer": dev_eer, "threshold": t_dev, "auc": auc(dev_scores, dev["labels"])}
        thresholds = {"dev_eer": t_dev} if wdir is not None else {"dev_eer": t_dev, "deployed_0.5": 0.5}
        for tname, t in thresholds.items():
            for name, d in tests.items():
                s, y, sp = (per_video(d, test_scores[name]) if name == "webcam" else (test_scores[name], d["labels"], d["species"]))
                res.setdefault(tname, {})[name] = report(s, y, sp, t)
        out["systems"][sys_name] = res
        score_cache[sys_name] = {"dev": dev_scores, **test_scores}
        print(sys_name, json.dumps({k: {n: (round(v["apcer_worst"], 3), round(v["bpcer"], 3), round(v["auc"], 3)) for n, v in r.items()}
                                    for k, r in res.items() if isinstance(r, dict) and k not in ("dev",)}, indent=None))

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "pad_benchmark.json").write_text(json.dumps(out, indent=2))
    for name, d in {"dev": dev, **tests}.items():
        with open(RESULTS / f"pad_scores_{name}.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["id", "label_bona_fide", "species", *score_cache])
            for i in range(len(d["labels"])):
                w.writerow([d["ids"][i], int(d["labels"][i]), d["species"][i], *[f"{score_cache[s][name][i]:.5f}" for s in score_cache]])


if __name__ == "__main__":
    main()
