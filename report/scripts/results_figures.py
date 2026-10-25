"""Figures for the revised report, drawn only from saved result files:
evaluation/results/*.json|csv and report/data/v2/*.

Colour use: grey = the v0.2 heuristic baseline, blue = the deployed model,
orange = the alternative it is compared with (fine-tuned PAD, layered
pipeline). Every figure has a legend; no colour carries meaning on its own.

Run: python report/scripts/results_figures.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
RES = REPO / "evaluation" / "results"
V2 = REPO / "report" / "data" / "v2"
FIG = REPO / "report" / "figures"

BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8a85"
INK, MUTED = "#0b0b0b", "#52514e"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e4e3df",
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
})


def label_small(ax, bars, limit: float = 0.03) -> None:
    """Print the value on bars too short to see, so 0% never reads as missing."""
    for bar in bars:
        h = bar.get_height()
        if h < limit:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.0%}" if h == 0 else f"{h:.1%}",
                    ha="center", va="bottom", fontsize=6.5, color=MUTED)


def bar_axes(ax) -> None:
    ax.grid(axis="x", visible=False)


def roc(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(APCER, 1 - BPCER) over all thresholds."""
    order = np.argsort(-scores)
    y = labels[order]
    tp = np.r_[0, np.cumsum(y)] / max(1, y.sum())
    fp = np.r_[0, np.cumsum(~y)] / max(1, (~y).sum())
    return fp, tp


def load_scores(name: str) -> dict:
    with open(RES / f"pad_scores_{name}.csv") as fh:
        rows = list(csv.DictReader(fh))
    out = {"labels": np.array([r["label_bona_fide"] == "1" for r in rows])}
    for key in ("legacy", "released", "finetuned"):
        if key in rows[0]:
            out[key] = np.array([float(r[key]) for r in rows])
    return out


def fig_pad_roc() -> None:
    bench = json.loads((RES / "pad_benchmark.json").read_text())
    titles = {"casia_test": "CASIA-FASD test (30 unseen subjects)", "celeba_test": "CelebA-Spoof test (cross-dataset)"}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
    for ax, name in zip(axes, titles):
        d = load_scores(name)
        for key, color, label in [("legacy", GREY, "v0.2 heuristic"), ("released", BLUE, "MiniFASNet (released)"), ("finetuned", ORANGE, "MiniFASNet (CASIA fine-tuned)")]:
            if key not in d:
                continue
            fp, tp = roc(d[key], d["labels"])
            a = bench["systems"][key]["dev_eer"][name]["auc"]
            ax.plot(fp, tp, color=color, lw=2, label=f"{label}, AUC {a:.3f}")
        ax.plot([0, 1], [0, 1], color="#c9c8c3", lw=1, ls="--")
        ax.set_title(titles[name], fontsize=9, color=INK)
        ax.set_xlabel("APCER (attacks accepted)")
        ax.legend(loc="lower right", fontsize=7.5)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.01)
    axes[0].set_ylabel("1 − BPCER (bona fide accepted)")
    fig.tight_layout()
    fig.savefig(FIG / "fig_v2_pad_roc.png", dpi=300)
    plt.close(fig)


def fig_pad_species() -> None:
    bench = json.loads((RES / "pad_benchmark.json").read_text())
    panels = [("casia_test", "CASIA-FASD test"), ("webcam", "Webcam sample (per video)")]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [4, 7]})
    for ax, (name, title) in zip(axes, panels):
        species = list(bench["systems"]["released"]["dev_eer"][name]["apcer_per_species"]) + ["BPCER"]
        x = np.arange(len(species))
        for i, (key, color, label) in enumerate([("legacy", GREY, "v0.2 heuristic"), ("released", BLUE, "MiniFASNet (released)")]):
            r = bench["systems"][key]["dev_eer"][name]
            vals = [r["apcer_per_species"][s] for s in species[:-1]] + [r["bpcer"]]
            label_small(ax, ax.bar(x + (i - 0.5) * 0.38, vals, width=0.36, color=color, label=label, edgecolor="white", linewidth=1))
        bar_axes(ax)
        ax.set_xticks(x, [s.replace("_", " ") for s in species], rotation=30, ha="right", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_title(title, fontsize=9, color=INK)
    axes[0].set_ylabel("Error rate at the dev-EER threshold")
    axes[1].legend(loc="upper right", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(FIG / "fig_v2_pad_species.png", dpi=300)
    plt.close(fig)


def fig_injection() -> None:
    r = json.loads((RES / "injection_eval.json").read_text())
    conds = ["inj_25fps_crf23", "inj_24fps_crf28", "inj_30fps_crf23", "inj_30fps_crf18", "inj_lossless_30"]
    labels = ["25→30 fps\nCRF 23", "24→30 fps\nCRF 28", "30 fps\nCRF 23", "30 fps\nCRF 18", "30 fps\nlossless"]
    systems = [("legacy_naive_attacker", GREY, "timing heuristic, naive attacker"),
               ("legacy_adaptive_attacker", "#4d4d4a", "timing heuristic, forged jitter"),
               ("classifier", BLUE, "content classifier (this work)")]
    x = np.arange(len(conds))
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    for i, (key, color, label) in enumerate(systems):
        vals = [r[key]["apcer_per_condition"][c] for c in conds]
        label_small(ax, ax.bar(x + (i - 1) * 0.27, vals, width=0.25, color=color, label=f"{label} (BPCER {r[key]['bpcer']:.1%})", edgecolor="white", linewidth=1))
    bar_axes(ax)
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_ylabel("APCER (injected clips accepted)")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(FIG / "fig_v2_injection.png", dpi=300)
    plt.close(fig)


def fig_deepfake() -> None:
    path = RES / "deepfake_benchmark.json"
    if not path.exists():
        return
    r = json.loads(path.read_text())
    gens = ["insight", "inpainting", "text2img"]
    names = ["InsightFace swap\n(training generator)", "SD inpainting\n(unseen)", "SD text2img\n(unseen)"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for i, (key, color, label) in enumerate([("legacy", GREY, "v0.2 FFT heuristic"), ("efficientnet_b0", BLUE, "EfficientNet-B0 (fine-tuned)")]):
        s = r["systems"][key]
        axes[0].bar(np.arange(3) + (i - 0.5) * 0.38, [s["test_auc_per_generator"][g] for g in gens], width=0.36, color=color, label=label, edgecolor="white")
        label_small(axes[1], axes[1].bar(np.arange(3) + (i - 0.5) * 0.38, [s["test"]["apcer_per_species"][g] for g in gens], width=0.36, color=color, label=label, edgecolor="white"))
    axes[0].set_ylabel("AUC (real vs this generator)")
    axes[0].set_ylim(0, 1.05)
    axes[0].axhline(0.5, color="#c9c8c3", ls="--", lw=1)
    axes[1].set_ylabel("APCER at dev-EER threshold")
    axes[1].set_ylim(0, 1.05)
    for ax in axes:
        bar_axes(ax)
        ax.set_xticks(np.arange(3), names, fontsize=7.5)
    axes[0].legend(loc="lower left", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(FIG / "fig_v2_deepfake.png", dpi=300)
    plt.close(fig)


def _read(name: str) -> list[dict]:
    with open(V2 / name) as fh:
        return list(csv.DictReader(fh))


def fig_latency() -> None:
    if not (V2 / "d_summary.json").exists():
        return
    g = _read("d1_genuine_integrated.csv")
    med = lambda k: float(np.median([float(r[k]) for r in g if r.get(k) not in (None, "")]))  # noqa: E731
    client = [("encode frames (PNG)", med("encode_frames_ms")), ("SHA-256", med("hash_ms")), ("ML-DSA sign", med("sign_ms")), ("ML-KEM encaps + AES-GCM seal", med("encaps_seal_ms"))]
    server = [("ML-KEM decaps", med("server_kem_decapsulate_ms")), ("AES-GCM open", med("server_aead_decrypt_ms")),
              ("binding gate (incl. ML-DSA verify)", med("server_binding_check_ms")), ("decode + liveness", med("server_liveness_check_ms")),
              ("five detectors", med("server_detection_ms")), ("fusion", med("server_fusion_ms"))]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [1, 1.25]})
    for ax, items, color, title in [(axes[0], client, AQUA, "Capture device (client process)"), (axes[1], server, BLUE, "Server process")]:
        names, vals = zip(*items)
        y = np.arange(len(vals))[::-1]
        ax.barh(y, vals, color=color, height=0.6)
        ax.set_xscale("log")
        ax.set_yticks(y, names, fontsize=7.5)
        ax.set_xlabel("median latency (ms, log scale)")
        ax.set_title(title, fontsize=9, color=INK)
        for yi, v in zip(y, vals):
            ax.text(v * 1.15, yi, f"{v:.3g}", va="center", fontsize=7, color=MUTED)
        ax.set_xlim(min(vals) / 2, max(vals) * 6)
    fig.tight_layout()
    fig.savefig(FIG / "fig_v2_latency.png", dpi=300)
    plt.close(fig)


def fig_dos() -> None:
    path, flood_path = V2 / "d_summary.json", V2 / "d3_flood.json"
    if not (path.exists() and flood_path.exists()):
        return
    s = json.loads(path.read_text())
    flood = {(r["mode"], r["flood"]): r for r in json.loads(flood_path.read_text())}
    kinds = ["forged_signature", "reused_nonce", "stale_timestamp", "substituted_frames"]
    labels = ["forged\nsignature", "reused\nnonce", "stale\ntimestamp", "substituted\nframes"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [1.5, 1]})
    x = np.arange(len(kinds))
    for i, (mode, color, label) in enumerate([("integrated", BLUE, "integrated (this work)"), ("layered", ORANGE, "layered baseline")]):
        vals = [s[mode]["reject"][k]["server_total_ms"]["median"] for k in kinds]
        bars = axes[0].bar(x + (i - 0.5) * 0.38, vals, width=0.36, color=color, label=label, edgecolor="white")
        for bar, v in zip(bars, vals):
            axes[0].text(bar.get_x() + bar.get_width() / 2, v * 1.15, f"{v:.3g}", ha="center", va="bottom", fontsize=6.5, color=MUTED)
    axes[0].set_yscale("log")
    axes[0].set_xticks(x, labels, fontsize=8)
    axes[0].set_ylabel("server time per rejected request (ms)")
    axes[0].legend(loc="upper left", fontsize=7.5)
    axes[0].set_ylim(top=axes[0].get_ylim()[1] * 4)
    # genuine latency without / with a forged-request flood
    xs = np.arange(2)
    for i, (mode, color) in enumerate([("integrated", BLUE), ("layered", ORANGE)]):
        vals = [flood[(mode, False)]["victim_ms_median"] / 1000, flood[(mode, True)]["victim_ms_median"] / 1000]
        bars = axes[1].bar(xs + (i - 0.5) * 0.38, vals, width=0.36, color=color, edgecolor="white")
        for bar, v in zip(bars, vals):
            axes[1].text(bar.get_x() + bar.get_width() / 2, v, f"{v:.2f} s", ha="center", va="bottom", fontsize=7, color=MUTED)
    axes[1].set_xticks(xs, ["no attack", f"{flood[('integrated', True)]['attackers']} flooding\nprocesses"], fontsize=8)
    axes[1].set_ylabel("genuine verification latency (s)")
    for ax in axes:
        bar_axes(ax)
    fig.tight_layout()
    fig.savefig(FIG / "fig_v2_dos.png", dpi=300)
    plt.close(fig)


def fig_primitives() -> None:
    """Re-draw fig_r1/fig_r2 from this run's primitive benchmark (v2/s1_primitives.csv)."""
    import sys

    sys.path.insert(0, str(REPO / "report" / "scripts"))
    from system_experiments import plot_primitives

    rows = []
    for r in _read("s1_primitives.csv"):
        rows.append({k: (float(v) if k.endswith(("_ms", "_B")) and v not in ("", None) else v) for k, v in r.items()})
    plot_primitives(rows)


if __name__ == "__main__":
    fig_primitives()
    fig_pad_roc()
    fig_pad_species()
    fig_injection()
    fig_deepfake()
    fig_latency()
    fig_dos()
