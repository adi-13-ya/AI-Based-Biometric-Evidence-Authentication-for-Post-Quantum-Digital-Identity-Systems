"""D5: validly signed attack submissions, and which check rejects each one.

Every submission here is signed by the *enrolled* device key and sealed for
its own fresh session, exactly as the real client does: nonce, freshness,
signature and evidence hash all pass. This is the case the cryptographic
layer cannot help with (a compromised or malicious client, or an attacker
who holds a device key), so it isolates what the liveness check and the
five detectors catch. Sessions are re-opened until a spoken-digit challenge
is issued (as in D1): a head-motion challenge would reject most pre-recorded
material at g5 and hide the detectors.

Submissions (public data only; counts in SAMPLES):
  bona fide      real webcam clips (UniqueData sample, 'real')
  presentation   webcam attack videos: print, print_cut, outline, monitor, mask, silicone
  replay         an accepted bona fide clip, resubmitted in a new session
  injection      real-person webcam footage played through the emulated
                 virtual camera of evaluation/train_injection.py (5 conditions)
  still image    CASIA-FASD test frames (warped photo, cut photo, video
                 replay) and DeepFakeFace test fakes (insight, inpainting,
                 text2img), each injected as a 16-frame stream of the still
  morph          the demo morph sample only (the morph detector is not
                 benchmarked; reported for completeness, not as a rate)

For every submission the server's response gives the first failed binding
check (if any) and the list of detectors below their thresholds; a
submission can be vetoed by several detectors and every veto is counted.

Run (no other server on port 8765): python report/scripts/signed_attacks.py
Writes report/data/v2/d5_signed_attacks.csv and d5_signed_attacks.json
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "report" / "scripts"))

from distributed_experiments import OUT, Server, api, fresh_device, genuine_submission  # noqa: E402
from evaluation.metrics import wilson_interval  # noqa: E402

SAMPLES = {"bona_fide_clips_per_video": 3, "pa_clips_per_video": 1, "injection_sources": 10,
           "casia_per_species": 20, "dff_per_generator": 25}
MODULES = ["presentation_attack", "deepfake", "injection", "replay", "morph"]
WEBCAM = REPO / "datasets" / "webcam_liveness" / "files"


def clip_from(frames: list[np.ndarray], start: int) -> tuple[list[np.ndarray], int]:
    from client_demo.recorder import select_frames

    return select_frames(frames[start: start + 60])


def build_submissions(rng: np.random.Generator) -> list[dict]:
    from evaluation.train_injection import CONDITIONS, decode, virtual_camera

    subs = []
    videos = {sp: sorted((WEBCAM / sp).glob("*.mp4")) for sp in sorted(p.name for p in WEBCAM.iterdir() if p.is_dir())}
    decoded = {sp: [decode(v, 160) for v in vs] for sp, vs in videos.items()}
    for sp, clips in decoded.items():
        per = SAMPLES["bona_fide_clips_per_video"] if sp == "real" else SAMPLES["pa_clips_per_video"]
        for vi, frames in enumerate(clips):
            for _ in range(per):
                start = int(rng.integers(0, max(1, len(frames) - 60)))
                f, burst = clip_from(frames, start)
                subs.append({"cls": "bona fide" if sp == "real" else "presentation", "species": "webcam_real" if sp == "real" else sp,
                             "source": f"{sp}/{vi}", "frames": f, "burst": burst})
    for cond, (fps, crf) in CONDITIONS.items():
        for vi, frames in enumerate(decoded["real"][: SAMPLES["injection_sources"]]):
            played = virtual_camera(frames, fps, crf, 90)
            f, burst = clip_from(played, 0)
            subs.append({"cls": "injection", "species": cond, "source": f"real/{vi}", "frames": f, "burst": burst})

    casia = np.load(REPO / "datasets" / "cache" / "pad_casia_test.npz", allow_pickle=False)
    from PIL import Image

    folder = REPO / "datasets" / "casia_fasd" / "test_img" / "test_img" / "color"
    for sp in ["warped_photo", "cut_photo", "video_replay"]:
        ids = [i for i, s in zip(casia["ids"], casia["species"]) if s == sp]
        for name in rng.choice(ids, size=SAMPLES["casia_per_species"], replace=False):
            img = np.asarray(Image.open(folder / str(name)).convert("RGB"))
            subs.append({"cls": "still image", "species": f"casia_{sp}", "source": str(name).split("_")[0],
                         "frames": [img] * 16, "burst": 8})
    dff = np.load(REPO / "datasets" / "cache" / "dff_test.npz", allow_pickle=False)
    for gen in ["insight", "inpainting", "text2img"]:
        idx = np.flatnonzero(dff["sources"] == gen)
        for i in rng.choice(idx, size=SAMPLES["dff_per_generator"], replace=False):
            crop = dff["crops"][i]
            img = np.asarray(Image.fromarray(crop).resize((320, 320)))
            subs.append({"cls": "still image", "species": f"dff_{gen}", "source": Path(str(dff["keys"][i])).name.split("_")[0],
                         "frames": [img] * 16, "burst": 8})
    morph = REPO / "demo_assets" / "attacks"
    for p in sorted(morph.glob("morph*.*")):
        img = np.asarray(Image.open(p).convert("RGB"))
        subs.append({"cls": "morph (not benchmarked)", "species": "demo_morph", "source": p.name, "frames": [img] * 16, "burst": 8})
    return subs


def outcome(resp: dict) -> dict:
    failed = next((k for k, v in resp["binding_checks"].items() if not v), "")
    vetoes = resp.get("below_threshold_modules") or []
    return {"accepted": resp["accepted"], "binding_failed": failed, "detection_ran": resp["detection_ran"],
            "vetoes": "+".join(vetoes), **{f"veto_{m}": m in vetoes for m in MODULES},
            **{f"score_{m}": (resp.get("detection_scores") or {}).get(m) for m in MODULES}}


def main() -> None:
    rng = np.random.default_rng(1107)
    subs = build_submissions(rng)
    print(len(subs), "submissions", Counter(s["cls"] for s in subs))
    fresh_device("ML-DSA-65")
    server = Server("integrated")
    client = api()
    client._device = None
    rows = []
    try:
        client.enroll_device()
        accepted_genuine = []
        for k, s in enumerate(subs):
            client.reset_replay_store()  # each submission is judged on its own, except the replay tests below
            resp, _ = genuine_submission(client, s["frames"], s["burst"])
            row = {"k": k, "cls": s["cls"], "species": s["species"], "source": s["source"], **outcome(resp)}
            rows.append(row)
            if s["cls"] == "bona fide" and resp["accepted"] and len(accepted_genuine) < 10:
                accepted_genuine.append(s)
            if k % 25 == 0:
                print(k, row["cls"], row["species"], row["accepted"], row["binding_failed"], row["vetoes"], flush=True)
        for s in accepted_genuine:  # replay: same evidence, new session, replay memory intact
            client.reset_replay_store()
            genuine_submission(client, s["frames"], s["burst"])
            resp, _ = genuine_submission(client, s["frames"], s["burst"])
            rows.append({"k": len(rows), "cls": "replay", "species": "resubmitted_genuine", "source": s["source"], **outcome(resp)})
    finally:
        server.stop()

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "d5_signed_attacks.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    groups = defaultdict(list)
    for r in rows:
        groups[(r["cls"], r["species"])].append(r)
    summary = []
    for (cls, sp), rs in sorted(groups.items()):
        n = len(rs)
        acc = sum(r["accepted"] for r in rs)
        lo, hi = wilson_interval(acc, n)
        summary.append({
            "cls": cls, "species": sp, "n": n, "sources": len({r["source"] for r in rs}),
            "accepted": acc, "accept_rate": acc / n, "accept_ci95": [lo, hi],
            "rejected_at_binding": dict(Counter(r["binding_failed"] for r in rs if r["binding_failed"])),
            "veto_rate": {m: sum(r[f"veto_{m}"] for r in rs) / n for m in MODULES},
            "sole_veto": dict(Counter(r["vetoes"] for r in rs if r["vetoes"] and "+" not in r["vetoes"])),
        })
    (OUT / "d5_signed_attacks.json").write_text(json.dumps(summary, indent=2))
    for s in summary:
        print(f"{s['cls']:24s} {s['species']:22s} n={s['n']:3d} accepted={s['accept_rate']:.0%} "
              f"binding={s['rejected_at_binding']} veto={ {m: round(v, 2) for m, v in s['veto_rate'].items() if v} }")


if __name__ == "__main__":
    main()
