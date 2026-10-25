"""Evaluate the deepfake module on DeepFakeFace (held-out identities).

Systems: the v0.2 FFT high-frequency heuristic ("legacy", applied to the
same face crops) and the fine-tuned EfficientNet-B0. The operating
threshold of each is fixed at the EER point of the development split and
applied unchanged to the test split. Each generator is reported as its own
attack species: insight (the training generator, new identities) and the
two diffusion generators never seen in training (inpainting, text2img).

Run after evaluation/train_deepfake.py.
Writes evaluation/results/deepfake_benchmark.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from evaluation.benchmark_pad import report  # noqa: E402
from evaluation.metrics import auc, eer  # noqa: E402
from evaluation.train_deepfake import build_split, scores  # noqa: E402


def main() -> None:
    import torch

    from detection.deepfake_detector import WEIGHTS_PATH, DeepfakeDetector, build_model

    dev, test = build_split("dev"), build_split("test")
    model = build_model()
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True))
    legacy = DeepfakeDetector(mode="mock")

    out = {"datasets": {s: {"images": int(len(d["labels"])), "sources": {k: int((d["sources"] == k).sum()) for k in sorted(set(d["sources"]))}}
                        for s, d in [("dev", dev), ("test", test)]}, "systems": {}}
    for name in ["legacy", "efficientnet_b0"]:
        if name == "legacy":
            s_dev = np.array([legacy.predict(c).score for c in dev["crops"]])
            s_test = np.array([legacy.predict(c).score for c in test["crops"]])
        else:
            s_dev, s_test = scores(model, dev["crops"], "cpu"), scores(model, test["crops"], "cpu")
        _, t = eer(s_dev, dev["labels"])
        species = np.where(test["labels"], "bona_fide", test["sources"])
        res = {"dev_auc": auc(s_dev, dev["labels"]), "dev_eer": eer(s_dev, dev["labels"])[0], "threshold": t,
               "test": report(s_test, test["labels"], species, t)}
        res["test_auc_per_generator"] = {
            g: auc(np.r_[s_test[test["labels"]], s_test[test["sources"] == g]],
                   np.r_[np.ones(test["labels"].sum(), bool), np.zeros((test["sources"] == g).sum(), bool)])
            for g in sorted(set(test["sources"][~test["labels"]]))}
        out["systems"][name] = res
        print(name, json.dumps({"apcer": res["test"]["apcer_per_species"], "bpcer": res["test"]["bpcer"], "auc": res["test_auc_per_generator"]}))
    (REPO / "evaluation" / "results" / "deepfake_benchmark.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
