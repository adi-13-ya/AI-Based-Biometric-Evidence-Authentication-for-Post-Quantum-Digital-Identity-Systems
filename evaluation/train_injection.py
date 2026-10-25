"""Train and evaluate the injection (virtual-camera) classifier.

There is no public benchmark of virtual-camera injection, so the attack
class is produced by emulating the injection pipeline on real webcam
footage, and the protocol is stated here so it can be checked:

Source footage: the 70 webcam recordings of the UniqueData "Web Camera Face
Liveness Detection" sample (CC BY-NC-ND 4.0; 10 each of real, print,
print_cut, monitor, mask, outline, silicone). All 70 are camera captures, so
all 70 supply *camera-stream* (bona fide for this module) clips: 8
consecutive decoded frames, exactly as the client would send them.

Injected clips: the same footage, treated as the attacker's video file, is
(1) encoded to H.264 at the file's frame rate and quality, (2) decoded, and
(3) resampled to the 30 fps the virtual camera advertises (a source frame is
repeated whenever the file is slower). Conditions:

    inj_25fps_crf23   typical downloaded / rendered clip
    inj_24fps_crf28   lower-quality film-rate clip
    inj_30fps_crf23   matched frame rate: no repeats, compression cue only
    inj_30fps_crf18   matched frame rate, high quality
    inj_lossless_30   lossless file at matched rate (expected to be missed:
                      it is pixel-identical to the camera clip)

The lossless condition is evaluated but excluded from training.

Evaluation: 5-fold GroupKFold over source videos (no video contributes to
both training and test of a fold). The decision threshold of each fold is
fixed on that fold's training scores at BPCER = 5% and then applied to its
test scores. Reported: APCER per condition, BPCER, AUC, EER (out-of-fold).

Baseline: the previous heuristic (InjectionDetector.predict on one frame
with client-reported timestamps). Camera timestamps are modelled as 30 fps
with N(0, 1.7 ms) jitter; the naive attacker sends exact 1/30 s intervals,
the adaptive attacker sends the same jittered distribution as a camera
(timestamps are client-controlled, so this costs the attacker nothing).

Outputs:
    detection/weights/injection_lr.json            final model (all data)
    evaluation/results/injection_features.csv      every clip's features
    evaluation/results/injection_eval.json         metrics

Run: python evaluation/train_injection.py [--clips-per-video 6]
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from detection.injection_detector import FEATURE_NAMES, InjectionDetector, LogisticModel, WEIGHTS_PATH, sequence_features  # noqa: E402
from evaluation.metrics import apcer_bpcer, auc, eer, threshold_at_bpcer  # noqa: E402

DATA = REPO / "datasets" / "webcam_liveness" / "files"
RESULTS = REPO / "evaluation" / "results"
CLIP = 8
VCAM_FPS = 30
CONDITIONS = {
    "inj_25fps_crf23": (25, 23),
    "inj_24fps_crf28": (24, 28),
    "inj_30fps_crf23": (30, 23),
    "inj_30fps_crf18": (30, 18),
    "inj_lossless_30": (30, None),
}
TRAIN_CONDITIONS = [c for c in CONDITIONS if c != "inj_lossless_30"]


def decode(path: Path, max_frames: int = 240) -> list[np.ndarray]:
    import av

    frames = []
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
            if len(frames) >= max_frames:
                break
    return frames


def h264_roundtrip(frames: list[np.ndarray], fps: int, crf: int) -> list[np.ndarray]:
    import av

    h, w = (frames[0].shape[0] // 2) * 2, (frames[0].shape[1] // 2) * 2
    buf = io.BytesIO()
    with av.open(buf, mode="w", format="mp4") as out:
        stream = out.add_stream("libx264", rate=fps)
        stream.width, stream.height, stream.pix_fmt = w, h, "yuv420p"
        stream.options = {"crf": str(crf), "preset": "veryfast"}
        for f in frames:
            for packet in stream.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(f[:h, :w]), format="rgb24")):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
    buf.seek(0)
    with av.open(buf) as container:
        return [fr.to_ndarray(format="rgb24") for fr in container.decode(video=0)]


def virtual_camera(source: list[np.ndarray], file_fps: int, crf: int | None, n_out: int) -> list[np.ndarray]:
    """Emulate playing `source` (a file at file_fps) into a 30 fps virtual camera."""
    n_src = int(np.ceil(n_out * file_fps / VCAM_FPS)) + 1
    src = source[:n_src]
    played = src if crf is None else h264_roundtrip(src, file_fps, crf)
    idx = [min(len(played) - 1, int(k * file_fps / VCAM_FPS)) for k in range(n_out)]
    return [played[i] for i in idx]


def camera_timestamps(rng: np.random.Generator, n: int) -> list[float]:
    return list(np.cumsum(np.r_[0.0, rng.normal(1 / 30, 0.0017, n - 1)]))


def build_clips(clips_per_video: int, seed: int = 2026) -> list[dict]:
    rng = np.random.default_rng(seed)
    legacy = InjectionDetector(mode="mock")
    rows = []
    videos = sorted(DATA.glob("*/*.mp4"))
    for vi, path in enumerate(videos):
        frames = decode(path)
        need = CLIP * 2 + 2
        if len(frames) < need:
            print(f"skip {path} ({len(frames)} frames)")
            continue
        starts = rng.choice(len(frames) - need, size=min(clips_per_video, len(frames) - need), replace=False)
        for start in starts:
            source = frames[start: start + need]
            clips = {"camera": source[:CLIP]}
            for name, (fps, crf) in CONDITIONS.items():
                clips[name] = virtual_camera(source, fps, crf, CLIP)
            for name, clip in clips.items():
                feats = sequence_features(clip)
                bona = name == "camera"
                regular = [i / 30 for i in range(30)]
                jittered = camera_timestamps(rng, 30)
                rows.append({
                    "video": f"{path.parent.name}/{path.stem}",
                    "group": vi,
                    "condition": name,
                    "bona_fide": bona,
                    **feats,
                    # legacy heuristic: camera clips get camera-like timestamps;
                    # injected clips get regular ones (naive) or forged jitter (adaptive)
                    "legacy_naive": legacy.predict(clip[0], frame_timestamps=jittered if bona else regular).score,
                    "legacy_adaptive": legacy.predict(clip[0], frame_timestamps=jittered).score,
                })
        print(f"[{vi + 1}/{len(videos)}] {path.parent.name}/{path.name}: {len(frames)} frames")
    return rows


def fit(X: np.ndarray, y: np.ndarray) -> dict:
    from sklearn.linear_model import LogisticRegression

    mean, scale = X.mean(axis=0), X.std(axis=0) + 1e-9
    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit((X - mean) / scale, y)
    return {
        "features": FEATURE_NAMES,
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "coef": clf.coef_[0].tolist(),
        "intercept": float(clf.intercept_[0]),
    }


def main() -> None:
    import pandas as pd
    from sklearn.model_selection import GroupKFold

    ap = argparse.ArgumentParser()
    ap.add_argument("--clips-per-video", type=int, default=6)
    args = ap.parse_args()

    rows = build_clips(args.clips_per_video)
    df = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS / "injection_features.csv", index=False)

    train_mask = df.condition.isin(["camera", *TRAIN_CONDITIONS]).to_numpy()
    X = df[FEATURE_NAMES].to_numpy(dtype=np.float64)
    y = df.bona_fide.to_numpy()
    groups = df.group.to_numpy()

    oof = np.full(len(df), np.nan)
    oof_accept = np.zeros(len(df), dtype=bool)
    fold_thresholds = []
    for train_idx, test_idx in GroupKFold(n_splits=5).split(X, y, groups):
        tr = train_idx[train_mask[train_idx]]
        model = LogisticModel(fit(X[tr], y[tr]))
        train_scores = np.array([model.p_bona_fide(dict(zip(FEATURE_NAMES, X[i]))) for i in tr])
        t = threshold_at_bpcer(train_scores, y[tr], 0.05)
        fold_thresholds.append(t)
        for i in test_idx:
            oof[i] = model.p_bona_fide(dict(zip(FEATURE_NAMES, X[i])))
            oof_accept[i] = oof[i] >= t
    df["oof_score"] = oof
    df["oof_accept"] = oof_accept

    def summarize(score_col: str, accept: np.ndarray | None, threshold: float | None) -> dict:
        res = {}
        bona = df.bona_fide.to_numpy()
        acc = accept if accept is not None else (df[score_col].to_numpy() >= threshold)
        res["bpcer"] = float((~acc[bona]).mean())
        res["apcer_per_condition"] = {c: float(acc[(df.condition == c).to_numpy()].mean()) for c in CONDITIONS}
        trained = df.condition.isin(["camera", *TRAIN_CONDITIONS]).to_numpy()
        res["apcer_max_trained_conditions"] = max(res["apcer_per_condition"][c] for c in TRAIN_CONDITIONS)
        res["auc_trained_conditions"] = auc(df[score_col][trained], bona[trained])
        res["eer_trained_conditions"] = eer(df[score_col][trained], bona[trained])[0]
        return res

    results = {
        "protocol": "5-fold GroupKFold by source video; per-fold threshold at BPCER=5% on training scores",
        "n_videos": int(df.group.nunique()),
        "n_clips": int(len(df)),
        "clips_by_condition": df.condition.value_counts().to_dict(),
        "classifier": summarize("oof_score", oof_accept, None),
        "fold_thresholds": fold_thresholds,
        # legacy heuristic at its deployed veto threshold 0.5
        "legacy_naive_attacker": summarize("legacy_naive", None, 0.5),
        "legacy_adaptive_attacker": summarize("legacy_adaptive", None, 0.5),
    }

    final = fit(X[train_mask], y[train_mask])
    final_model = LogisticModel(final)
    scores = np.array([final_model.p_bona_fide(dict(zip(FEATURE_NAMES, X[i]))) for i in np.flatnonzero(train_mask)])
    final["threshold_bpcer5"] = threshold_at_bpcer(scores, y[train_mask], 0.05)
    final["training"] = {"clips": int(train_mask.sum()), "source": "evaluation/train_injection.py"}
    results["final_model"] = {k: final[k] for k in ("coef", "intercept", "threshold_bpcer5")}
    WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    WEIGHTS_PATH.write_text(json.dumps(final, indent=2))
    df.to_csv(RESULTS / "injection_features.csv", index=False)
    (RESULTS / "injection_eval.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
