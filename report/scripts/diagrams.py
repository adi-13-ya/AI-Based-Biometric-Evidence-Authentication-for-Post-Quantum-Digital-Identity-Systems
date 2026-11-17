"""Architecture and protocol diagrams for the final report (matplotlib only,
so no Graphviz/Mermaid toolchain is needed)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

FIG = Path(__file__).resolve().parents[1] / "figures"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 9})

BLUE, BLUE_L = "#2a6fbb", "#e3eefa"
ORANGE, ORANGE_L = "#c46f12", "#fdf0e1"
GREEN, GREEN_L = "#23864a", "#e4f4ea"
RED, RED_L = "#b3261e", "#fbe7e5"
GREY, GREY_L = "#5f6b7a", "#f2f4f7"


def box(ax, x, y, w, h, text, edge, face, fs=8.5, bold=False, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                lw=1.2, ec=edge, fc=face, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", wrap=True)


def arrow(ax, x1, y1, x2, y2, color="black", ls="-", lw=1.2, text=None, fs=7.5, rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=10,
                                 color=color, lw=lw, ls=ls, connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.12, text, ha="center", va="bottom", fontsize=fs, color=color)


def architecture() -> None:
    fig, ax = plt.subplots(figsize=(12, 8.2))
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 16.4)
    ax.axis("off")

    # --- Quantum adversary layer -------------------------------------------
    ax.add_patch(FancyBboxPatch((0.3, 12.6), 23.4, 3.5, boxstyle="round,pad=0.02,rounding_size=0.2",
                                lw=1.4, ec=RED, fc="#fff8f7", ls="--"))
    ax.text(0.6, 15.75, "QUANTUM ADVERSARY MODEL (illustrative circuits in Appendix A)", fontsize=9.5, fontweight="bold", color=RED)
    box(ax, 0.8, 13.0, 5.4, 2.3, "Shor's algorithm\nQFT order finding\n⇒ breaks RSA-3072, ECDSA-P256,\nX25519 in poly-time",
        RED, RED_L)
    box(ax, 6.7, 13.0, 5.4, 2.3, "Grover's algorithm\namplitude amplification\n⇒ AES-256 → 128-bit,\nSHA-256 preimage → 128-bit", RED, RED_L)
    box(ax, 12.6, 13.0, 5.4, 2.3, "Harvest-now, decrypt-later\nrecord biometric sessions\ntoday, decrypt with a future\nCRQC", RED, RED_L)
    box(ax, 18.5, 13.0, 4.8, 2.3, "Quantum entropy (QRNG)\nH⊗n |0⟩ → uniform bits\nsimulated in Appendix A;\nnot used: N ← os.urandom", GREEN, GREEN_L)

    # --- Client ---------------------------------------------------------------
    ax.add_patch(FancyBboxPatch((0.3, 0.4), 7.4, 11.7, boxstyle="round,pad=0.02,rounding_size=0.2",
                                lw=1.2, ec=BLUE, fc="#fbfdff"))
    ax.text(4.0, 11.65, "CAPTURE DEVICE (holds sk_dev)", ha="center", fontsize=9.5, fontweight="bold", color=BLUE)
    cy = [9.9, 8.0, 6.1, 4.2, 2.3]
    labels = [
        "0. Enrol once: send pk_dev only\n1. Signed session request → offer\n(N, C, pk_KEM) signed by server; verify",
        "2. Capture 8 spread frames +\n8-frame burst while\nperforming challenge C",
        "3. h = SHA-256(frames)\n(Grover-resistant: 128-bit PQ)",
        "4. σ = ML-DSA-65.Sign(sk_dev,\n⟨h, N, ts, r, id, H(ct), μ⟩)\n(on the device, never on the server)",
        "5. (ct, K) = ML-KEM-768.Encaps(pk)\nk = HKDF(K, salt=N); seal payload\nwith AES-256-GCM (AAD = session id)",
    ]
    for y, t in zip(cy, labels):
        box(ax, 0.8, y, 6.4, 1.45, t, BLUE, BLUE_L, fs=8)
    for a, b in zip(cy[:-1], cy[1:]):
        arrow(ax, 4.0, a, 4.0, b + 1.45, BLUE)

    # --- Channel --------------------------------------------------------------
    box(ax, 8.4, 5.2, 3.6, 3.0, "Quantum-safe\nchannel\n\nML-KEM-768 (FIPS 203)\nMLWE hardness\nIND-CCA2", GREY, GREY_L, fs=8, bold=False)
    arrow(ax, 7.2, 3.0, 10.2, 5.2, BLUE, rad=0.15)
    ax.text(10.0, 4.3, "ct, Enc_k(payload)", ha="left", fontsize=7.5, color=BLUE)
    arrow(ax, 12.0, 7.4, 13.2, 10.2, GREY, rad=0.25)

    # --- Server ---------------------------------------------------------------
    ax.add_patch(FancyBboxPatch((12.7, 0.4), 11.0, 11.7, boxstyle="round,pad=0.02,rounding_size=0.2",
                                lw=1.2, ec=ORANGE, fc="#fffcf8"))
    ax.text(18.2, 11.65, "SERVER (FastAPI + liboqs + PyTorch/MediaPipe)", ha="center", fontsize=9.5, fontweight="bold", color=ORANGE)
    box(ax, 13.2, 9.9, 4.9, 1.45, "6. K = ML-KEM.Decaps(sk, ct)\nk = HKDF(K); AES-GCM open (tag)", ORANGE, ORANGE_L, fs=8)
    box(ax, 13.2, 6.4, 4.9, 3.0,
        "7. Binding gate, cheapest first,\nstop at first failure:\n(1) N unused (read)  (2) |now − ts| ≤ 30 s\n(3) ML-DSA.Verify(pk_dev, σ) over stored N\n(4) H(frames) = h, 8+8 frames; consume N\n(5) decode + motion challenge (fail closed)",
        ORANGE, ORANGE_L, fs=7.6, bold=False)
    arrow(ax, 15.65, 9.9, 15.65, 9.4, ORANGE)
    box(ax, 13.2, 4.3, 4.9, 1.5, "REJECT early: (1)-(4) need no\nimage decoding; N stays unused", RED, RED_L, fs=8)
    arrow(ax, 15.65, 6.4, 15.65, 5.8, RED, text="any check fails", fs=7)

    box(ax, 18.6, 3.0, 4.7, 6.4, "", ORANGE, ORANGE_L)
    ax.text(20.95, 9.0, "8. AI attack-detection\npipeline", ha="center", va="top", fontsize=8.5, fontweight="bold")
    dets = ["PAD: 2× MiniFASNet (released),\n3 frames, worst score", "Morph: ghosting, mirror\nsymmetry of fine detail", "Deepfake: EfficientNet-B0\n(DeepFakeFace fine-tuned)",
            "Replay: pHash + residual\ncorrelation (binary)", "Injection: frame repeats,\ncodec features + LR"]
    for i, d in enumerate(dets):
        box(ax, 18.9, 7.25 - i * 0.92, 4.1, 0.78, d, GREY, "white", fs=7.2)
    arrow(ax, 18.1, 7.9, 18.6, 7.9, GREEN, text="all pass", fs=7)

    box(ax, 18.6, 0.8, 4.7, 1.7, "9. Per-detector veto: ACCEPT\niff ∀i: sᵢ ≥ τᵢ (τ from dev data;\nmean reported only)", GREEN, GREEN_L, fs=8)
    arrow(ax, 20.95, 3.0, 20.95, 2.5, GREEN)
    box(ax, 13.2, 0.8, 4.9, 1.7, "10. Decision + per-stage\nlatency → client\n(session consumed at (4))", GREY, GREY_L, fs=8)
    arrow(ax, 18.6, 1.65, 18.1, 1.65, GREY)

    # Threat-to-countermeasure links
    arrow(ax, 3.5, 13.0, 10.2, 8.25, RED, ls=":", lw=1.0, rad=-0.15)
    WB = dict(boxstyle="round,pad=0.2", fc="white", ec="none")
    ax.text(10.2, 10.2, "Shor + HNDL: assumed\nresisted by\nML-KEM / ML-DSA", fontsize=7, color=RED, ha="center", bbox=WB)
    arrow(ax, 9.4, 13.0, 4.0, 7.45, RED, ls=":", lw=1.0, rad=0.1)
    ax.text(9.6, 12.15, "Grover absorbed by 256-bit\nsymmetric / hash sizes", fontsize=7, color=RED, ha="center", bbox=WB)
    arrow(ax, 15.3, 13.0, 10.2, 8.25, RED, ls=":", lw=1.0, rad=0.15)

    fig.tight_layout()
    fig.savefig(FIG / "fig_a1_architecture.png", dpi=300)
    plt.close(fig)


def sequence() -> None:
    fig, ax = plt.subplots(figsize=(11, 8.2))
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 16)
    ax.axis("off")
    lanes = {"Capture device": 2.5, "Server API": 9.0, "Session / device store": 14.0, "Detectors + fusion": 19.2}
    for name, x in lanes.items():
        box(ax, x - 1.8, 14.8, 3.6, 0.9, name, GREY, GREY_L, fs=9, bold=True)
        ax.plot([x, x], [0.4, 14.8], color=GREY, lw=0.8, ls="--")

    C, S, ST, D = lanes.values()
    # Exactly docs/PROTOCOL.md, Sections 2-7.
    steps = [
        (C, S, "POST /device/enroll (alg, pk_dev) → device_id   [once; sk_dev never leaves; token if ENROLLMENT_TOKEN set]"),
        (C, S, "POST /session/start (device_id, ts, rn, Sign(sk_dev, REQ))   [fresh, rn unseen, ≤ 3 open, ≤ 12/min]"),
        (S, ST, "id, N←{0,1}²⁵⁶, C, (pk_KEM, sk_KEM)←ML-KEM-768.KeyGen; look up pk_dev"),
        (S, C, "OFFER = (id, N, C, pk_KEM, device_id, t, 30 s),  σ_srv = ML-DSA.Sign(sk_srv, OFFER)"),
        (C, C, "Verify(pk_srv pinned, OFFER, σ_srv), else abort  →  (ct, K) ← Encaps(pk_KEM)"),
        (C, C, "h = SHA-256(len‖frame …);  σ_dev = Sign(sk_dev, tag‖h‖N‖ts‖r‖id‖SHA-256(ct)‖meta)"),
        (C, C, "k = HKDF(K, salt N);  PQB1 = ct ‖ GCM_k(header: h, ts, r, σ_dev, meta) ‖ GCM_k(frames)"),
        (C, S, "POST /verify-sealed  PQB1   (≈8 KB header first, then ≈5.7 MB of frames)"),
        (S, S, "read head only; Decaps(ct); open header (bad tag ⇒ REJECT)"),
        (S, ST, "N unused? (read only) → |Δt| ≤ 30 s? → Verify(pk_dev, σ_dev)?   (stop at first failure)"),
        (ST, S, "any failure ⇒ REJECT now; rest of body drained unread (no buffer, decrypt or decode)"),
        (S, S, "all pass ⇒ open frames; check h, then shape; consume N atomically"),
        (S, D, "first pixel decode → liveness challenge → PAD, morph, deepfake, replay, injection"),
        (D, S, "s₁..s₅: ACCEPT iff every sᵢ ≥ τᵢ (separate vetoes; mean reported only)"),
        (S, C, "ACCEPT / REJECT, failed check, reasons, timing_ms"),
    ]
    y = 14.25
    for a, b, t in steps:
        y -= 0.9
        if a == b:
            x0 = a - 2.1 if a == C else a - 2.6
            ax.add_patch(FancyBboxPatch((x0, y - 0.3), 11.4 if a == C else 10.6, 0.6, boxstyle="round,pad=0.02",
                                        fc=BLUE_L if a == C else ORANGE_L, ec=BLUE if a == C else ORANGE, lw=0.8))
            ax.text(x0 + (11.4 if a == C else 10.6) / 2, y, t, ha="center", va="center", fontsize=7.5)
            continue
        color = BLUE if a == C else ORANGE if a == S else GREEN
        arrow(ax, a, y, b, y, color)
        ax.text((a + b) / 2 if abs(a - b) > 8 else max(a, b) - abs(a - b) / 2, y + 0.1, t, ha="center", va="bottom", fontsize=7.5,
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none"))
    fig.tight_layout()
    fig.savefig(FIG / "fig_a2_binding_sequence.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    import sys

    if "sequence" not in sys.argv[1:]:
        architecture()
    sequence()
