"""95% confidence intervals at the level of the independent unit.

Frames of one subject, clips of one source video and images of one person
are not independent samples, so per-image Wilson intervals are too narrow.
Here every metric is recomputed on B cluster-bootstrap resamples: whole
clusters are drawn with replacement, and the 2.5th and 97.5th percentiles
are reported. Thresholds stay frozen at the development values used in
the report; nothing is re-fitted.

  PAD, CASIA-FASD test   cluster = subject (30), deployed released MiniFASNet, tau from pad_benchmark.json
  PAD, webcam sample     cluster = video (70; the sample has no subject ids), scored per video as in the report
  Deepfake, DFF test     cluster = IMDB-WIKI person id (from the file name), EfficientNet-B0
  Injection              cluster = source video (70), out-of-fold scores and decisions of the classifier

CelebA-Spoof is not included: the subset has no subject ids, so only
per-image intervals are possible (the report keeps the Wilson intervals
there and says so).

Run after the benchmarks: python evaluation/cluster_ci.py
Writes evaluation/results/cluster_ci.json
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from evaluation.metrics import auc, eer  # noqa: E402

RES = REPO / "evaluation" / "results"
B = 2000
SEED = 20261107


def _rates(scores, labels, species, thr):
    acc = scores >= thr
    out = {"bpcer": float(np.mean(~acc[labels])) if labels.any() else np.nan}
    att = ~labels
    out["apcer_all"] = float(np.mean(acc[att])) if att.any() else np.nan
    for sp in sorted(set(species[att])):
        m = att & (species == sp)
        out[f"apcer_{sp}"] = float(np.mean(acc[m]))
    if labels.any() and att.any():
        out["eer"] = float(eer(scores, labels)[0])
        out["auc"] = float(auc(scores, labels))
    return out


def cluster_bootstrap(scores, labels, species, clusters, thr, b=B, seed=SEED) -> dict:
    """Point estimates plus cluster-bootstrap percentile intervals."""
    scores, labels, species, clusters = map(np.asarray, (scores, labels, species, clusters))
    point = _rates(scores, labels, species, thr)
    ids = np.unique(clusters)
    index = defaultdict(list)
    for i, c in enumerate(clusters):
        index[c].append(i)
    index = {c: np.array(v) for c, v in index.items()}
    rng = np.random.default_rng(seed)
    draws = defaultdict(list)
    for _ in range(b):
        pick = np.concatenate([index[c] for c in rng.choice(ids, size=len(ids), replace=True)])
        r = _rates(scores[pick], labels[pick], species[pick], thr)
        for k in point:
            if k in r and not np.isnan(r[k]):
                draws[k].append(r[k])
    return {
        "n_clusters": int(len(ids)), "n_items": int(len(scores)), "threshold": float(thr),
        "metrics": {k: {"value": v, "ci95": [float(np.percentile(draws[k], 2.5)), float(np.percentile(draws[k], 97.5))]}
                    for k, v in point.items() if draws.get(k)},
    }


def _read(name):
    with open(RES / name) as fh:
        return list(csv.DictReader(fh))


def pad() -> dict:
    bench = json.loads((RES / "pad_benchmark.json").read_text())
    thr = bench["systems"]["released"]["dev"]["threshold"]
    out = {}
    rows = _read("pad_scores_casia_test.csv")
    out["casia_test_released"] = cluster_bootstrap(
        [float(r["released"]) for r in rows], [r["label_bona_fide"] == "1" for r in rows],
        [r["species"] for r in rows], [r["id"].split("_")[0] for r in rows], thr)
    # Webcam: one score per video (mean over its frames), as in benchmark_pad.per_video.
    per = defaultdict(list)
    meta = {}
    for r in _read("pad_scores_webcam.csv"):
        vid = r["id"].split(":")[0]
        per[vid].append(float(r["released"]))
        meta[vid] = (r["label_bona_fide"] == "1", r["species"])
    vids = sorted(per)
    out["webcam_released_per_video"] = cluster_bootstrap(
        [np.mean(per[v]) for v in vids], [meta[v][0] for v in vids], [meta[v][1] for v in vids], vids, thr)
    return out


def deepfake() -> dict:
    import torch

    from detection.deepfake_detector import WEIGHTS_PATH, build_model
    from evaluation.train_deepfake import build_split, scores

    bench = json.loads((RES / "deepfake_benchmark.json").read_text())
    thr = bench["systems"]["efficientnet_b0"]["threshold"]
    test = build_split("test")
    model = build_model()
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True))
    s = scores(model, test["crops"], "cpu")
    persons = [Path(k).name.split("_")[0] for k in test["keys"]]
    species = np.where(test["labels"], "bona_fide", test["sources"])
    out = {"dff_test_efficientnet_b0": cluster_bootstrap(s, test["labels"], species, persons, thr)}
    # AUC per generator on the same resamples is threshold-free; report it per generator too.
    for g in sorted(set(test["sources"][~test["labels"]])):
        m = test["labels"] | (test["sources"] == g)
        out[f"dff_test_auc_{g}"] = cluster_bootstrap(s[m], test["labels"][m], species[m], np.array(persons)[m], thr)
    return out


def injection() -> dict:
    rows = _read("injection_features.csv")
    # Out-of-fold decisions already use each fold's own frozen threshold; feed
    # them as scores with threshold 0.5 so _rates reproduces exactly those decisions.
    acc = np.array([1.0 if r["oof_accept"] == "True" else 0.0 for r in rows])
    res = cluster_bootstrap(acc, [r["bona_fide"] == "True" for r in rows], [r["condition"] for r in rows],
                            [r["group"] for r in rows], 0.5)
    for k in ("eer", "auc"):  # meaningless on binary decisions
        res["metrics"].pop(k, None)
    s = np.array([float(r["oof_score"]) for r in rows])
    y = np.array([r["bona_fide"] == "True" for r in rows])
    auc_res = cluster_bootstrap(s, y, np.array([r["condition"] for r in rows]), [r["group"] for r in rows], 0.5)
    res["metrics"]["auc"] = auc_res["metrics"]["auc"]
    return {"injection_oof": res}


def main() -> None:
    out = {"bootstrap_resamples": B, "seed": SEED, **pad(), **injection()}
    if "--no-deepfake" not in sys.argv:
        out.update(deepfake())
    (RES / "cluster_ci.json").write_text(json.dumps(out, indent=2))
    for name, r in out.items():
        if isinstance(r, dict) and "metrics" in r:
            print(name, r["n_clusters"], "clusters:", {k: f"{100 * v['value']:.1f}% [{100 * v['ci95'][0]:.1f}, {100 * v['ci95'][1]:.1f}]"
                                                       if k not in ("auc",) else f"{v['value']:.3f} [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}]"
                                                       for k, v in r["metrics"].items()})


if __name__ == "__main__":
    main()
