"""D6: session safety and authenticated malicious clients, before and after v1.3.

Every part runs against two servers started from two source trees: "before"
(v1.2.1, commit 3586b1e, checked out with `git worktree add`) and "after"
(this tree). Same evidence, same machine, same client code.

A  Session burning (no device key). The attacker knows a live session id
   (it is in the URL) and, in the on-path cases, has seen the device's own
   sealed envelope. After each invalid request the device submits its own,
   correctly signed evidence; we record whether that submission still passes
   the cryptographic gate. Also the server cost of a forged request after
   the fix (the nonce is now read, not consumed, before the signature).
B  Authenticated malicious client (holds an enrolled device key, signs
   everything validly):
   B1 single-frame submission of a still attack image under a motion challenge
   B2 burst_frames rewritten on re-encoded injection clips (8 -> 2, 0)
   B3 one bona fide frame spliced in front of a webcam presentation attack
   B4 oversized submissions (12 MP frames, 64 frames)
   B5 white-box PGD against MiniFASNet + EfficientNet-B0 on webcam attacks
   B6 resource abuse: an enrolled device floods validly signed submissions,
      with and without the per-device session limits (after only)

Run (port 8765 free; before tree at BEFORE):
  python report/scripts/malicious_client.py [--before PATH] [--parts A,B1,...]
Writes report/data/v2/d6_malicious_client.json and d6_*.csv
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import statistics
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "report" / "scripts"))

from distributed_experiments import OUT, PORT, Server, api, fresh_device, webcam_clips, write_csv  # noqa: E402

BEFORE = REPO.parent / "before-fix-v1.2.1"
URL = f"http://127.0.0.1:{PORT}"
WEBCAM = REPO / "datasets" / "webcam_liveness" / "files"
RNG = np.random.default_rng(1116)


# ----------------------------------------------------------------- plumbing
def png(frame: np.ndarray) -> bytes:
    from client_demo.frame_utils import frame_to_png_bytes

    return frame_to_png_bytes(np.ascontiguousarray(frame))


def session_with(client, kind: str) -> dict:
    """Reopen until the server issues a challenge of `kind` ('digits' or
    'motion'), closing the others with a signed close."""
    for _ in range(80):
        s = client.session_start()
        if ("number" in s["challenge"].lower()) == (kind == "digits"):
            return s
        client.end_session(s)
    raise RuntimeError(f"no {kind} challenge")


def envelope(client, session, frame_bytes, burst, response=None, signer=None, ts=None) -> bytes:
    from crypto.channel import evidence_hash_of
    from crypto.signature import build_binding_payload

    response = client.challenge_answer(session["challenge"]) if response is None else response
    ts = ts or datetime.now(timezone.utc).isoformat()
    kem = client.encapsulate(session)
    h = evidence_hash_of(frame_bytes)
    ctx = client.binding_context(session, kem[0], burst, None)
    payload = build_binding_payload(h, base64.b64decode(session["nonce"]), ts, response, **ctx)
    sig = (signer or client.device()._signer).sign(payload)
    header = {"evidence_hash": h.hex(), "timestamp": ts, "challenge_response": response, "signature": sig.hex(),
              "frame_timestamps": None, "burst_frames": burst}
    return client.seal_submission(session, header, frame_bytes, kem)


def post(session_id: str, body: bytes) -> tuple[int, dict, float]:
    import requests

    t0 = time.perf_counter()
    r = requests.post(f"{URL}/session/{session_id}/verify-sealed", data=body,
                      headers={"Content-Type": "application/octet-stream"}, timeout=300)
    ms = (time.perf_counter() - t0) * 1000
    try:
        return r.status_code, r.json(), ms
    except ValueError:
        return r.status_code, {}, ms


def outcome(status: int, body: dict) -> dict:
    checks = body.get("binding_checks") or {}
    failed = next((k for k, v in checks.items() if not v), "")
    return {"http": status, "accepted": bool(body.get("accepted")), "binding_failed": failed if status == 200 else f"HTTP {status}",
            "liveness_passed": checks.get("challenge_fulfilled"), "detection_ran": bool(body.get("detection_ran")),
            "vetoes": "+".join(body.get("below_threshold_modules") or []),
            "server_ms": (body.get("timing_ms") or {}).get("total_ms")}


def submit(client, kind: str, frames: list[np.ndarray] | list[bytes], burst: int, **kw) -> dict:
    client.reset_replay_store()
    s = session_with(client, kind)
    fb = frames if isinstance(frames[0], bytes) else [png(f) for f in frames]
    status, body, ms = post(s["session_id"], envelope(client, s, fb, burst, **kw))
    return {**outcome(status, body), "http_ms": ms}


def start(tree: str, extra: dict | None = None):
    repo = BEFORE if tree == "before" else REPO
    return Server("integrated", extra_env=extra, repo=repo)


# ---------------------------------------------------------------- data
def pa_clips(n: int) -> list[tuple[str, list[np.ndarray], int]]:
    from client_demo.recorder import select_frames
    from evaluation.train_injection import decode

    out = []
    species = sorted(p.name for p in WEBCAM.iterdir() if p.is_dir() and p.name != "real")
    vids = [(sp, v) for sp in species for v in sorted((WEBCAM / sp).glob("*.mp4"))]
    for sp, v in vids[:n]:
        frames = decode(v, 160)
        f, burst = select_frames(frames[:60])
        out.append((sp, f, burst))
    return out


def stills(per: int) -> list[tuple[str, np.ndarray]]:
    from PIL import Image

    out = []
    casia = np.load(REPO / "datasets" / "cache" / "pad_casia_test.npz", allow_pickle=False)
    folder = REPO / "datasets" / "casia_fasd" / "test_img" / "test_img" / "color"
    for sp in ["warped_photo", "cut_photo", "video_replay"]:
        ids = [i for i, s in zip(casia["ids"], casia["species"]) if s == sp]
        for name in RNG.choice(ids, size=per, replace=False):
            out.append((f"casia_{sp}", np.asarray(Image.open(folder / str(name)).convert("RGB"))))
    dff = np.load(REPO / "datasets" / "cache" / "dff_test.npz", allow_pickle=False)
    for gen in ["insight", "inpainting", "text2img"]:
        for i in RNG.choice(np.flatnonzero(dff["sources"] == gen), size=per, replace=False):
            out.append((f"dff_{gen}", np.asarray(Image.fromarray(dff["crops"][i]).resize((320, 320)))))
    return out


def injection_clips(per_condition: int) -> list[tuple[str, list[np.ndarray], int]]:
    from client_demo.recorder import select_frames
    from evaluation.train_injection import CONDITIONS, decode, virtual_camera

    real = sorted((WEBCAM / "real").glob("*.mp4"))[:per_condition]
    decoded = [decode(v, 160) for v in real]
    out = []
    for cond, (fps, crf) in CONDITIONS.items():
        if crf is None:
            continue  # lossless conditions are not detectable by design; B2 asks whether the layout matters
        for frames in decoded:
            f, burst = select_frames(virtual_camera(frames, fps, crf, 90)[:60])
            out.append((cond, f, burst))
    return out


# ---------------------------------------------------------------- part A
def part_a(tree: str, clips) -> list[dict]:
    from crypto.signature import SignerIdentity

    fresh_device("ML-DSA-65")
    server = start(tree)
    client = api()
    client._device = None
    attacker = SignerIdentity(alg="ML-DSA-65")
    rows = []
    try:
        client.enroll_device()
        kinds = ["none (control)", "garbage body", "forged signature", "envelope of another session",
                 "on-path: victim envelope, frames bit-flipped", "on-path: victim envelope, truncated body", "unsigned session close"]
        for kind in kinds:
            for i in range(10):
                import requests

                frames, burst = clips[i % len(clips)]
                fb = [png(f) for f in frames]
                client.reset_replay_store()
                s = session_with(client, "digits")
                own = envelope(client, s, fb, burst)
                if kind == "garbage body":
                    post(s["session_id"], b"PQB1" + os.urandom(64) + os.urandom(20000))
                elif kind == "forged signature":
                    post(s["session_id"], envelope(client, s, fb, burst, signer=attacker))
                elif kind == "envelope of another session":
                    other = session_with(client, "digits")
                    post(s["session_id"], envelope(client, other, fb, burst))
                elif kind.startswith("on-path: victim envelope, frames"):
                    bad = bytearray(own)
                    bad[-100] ^= 1
                    post(s["session_id"], bytes(bad))
                elif kind.endswith("truncated body"):
                    post(s["session_id"], own[: len(own) // 2])
                elif kind == "unsigned session close":
                    requests.delete(f"{URL}/session/{s['session_id']}", timeout=10)
                status, body, _ = post(s["session_id"], own)
                checks = body.get("binding_checks") or {}
                survived = status == 200 and checks.get("signature_valid") is True and checks.get("nonce_unused") is True \
                    and checks.get("nonce_consumed", True) is True
                rows.append({"tree": tree, "attack": kind, "i": i, "http": status, "session_survived": survived,
                             "victim_accepted": bool(body.get("accepted")),
                             "victim_failed_check": next((k for k, v in checks.items() if not v), "")})
        # Cost of a forged request on this tree.
        costs = []
        frames, burst = clips[0]
        fb = [png(f) for f in frames]
        for _ in range(30):
            s = session_with(client, "digits")
            _, body, _ = post(s["session_id"], envelope(client, s, fb, burst, signer=attacker))
            costs.append(body["timing_ms"]["total_ms"])
        rows.append({"tree": tree, "attack": "_forged_cost_ms_median", "i": 0, "http": 200,
                     "session_survived": None, "victim_accepted": None, "victim_failed_check": statistics.median(costs)})
    finally:
        server.stop()
    return rows


# ---------------------------------------------------------------- part B
def part_b(tree: str, parts: set[str], data: dict) -> list[dict]:
    fresh_device("ML-DSA-65")
    server = start(tree)
    client = api()
    client._device = None
    rows = []

    def add(part, variant, label, r):
        rows.append({"tree": tree, "part": part, "variant": variant, "item": label, **r})

    try:
        client.enroll_device()
        if "B1" in parts:
            for label, img in data["stills"]:
                add("B1", "16-frame still stream", label, submit(client, "motion", [img] * 16, 8))
                add("B1", "single frame", label, submit(client, "motion", [img], 0))
        if "B2" in parts:
            for label, f, burst in data["injection"]:
                for b in (burst, 2, 0):
                    add("B2", f"burst_frames={b}", label, submit(client, "digits", f, b))
        if "B3" in parts:
            live = data["bona_fide"]
            for k, (label, f, burst) in enumerate(data["pa"]):
                add("B3", "honest", label, submit(client, "digits", f, burst))
                spliced = [live[k % len(live)][0][0]] + f[1:]
                add("B3", "live frame spliced first", label, submit(client, "digits", spliced, burst))
        if "B4" in parts:
            base = data["bona_fide"][0][0]
            from PIL import Image

            big = [np.asarray(Image.fromarray(x).resize((4000, 3000))) for x in base]
            hd = [np.asarray(Image.fromarray(x).resize((1920, 1080))) for x in base]
            for _ in range(3):
                add("B4", "16 x 1920x1080 (allowed)", "hd", submit(client, "digits", hd, 8))
                add("B4", "16 x 4000x3000", "12MP", submit(client, "digits", big, 8))
                add("B4", "64 frames", "64", submit(client, "digits", base * 4, 8))
        if "B5" in parts:
            for eps in (0, 4, 8):
                for label, f, burst in data["pa_pgd"]:
                    adv = pgd(f, eps) if eps else f
                    add("B5", f"PGD eps={eps}/255", label, submit(client, "digits", adv, burst))
    finally:
        server.stop()
    return rows


# --------------------------------------------------------------- PGD (B5)
_PGD = {}


def pgd(frames: list[np.ndarray], eps: int, steps: int = 20) -> list[np.ndarray]:
    """L-inf PGD on all frames against both PAD networks and the deepfake
    model, through the same face crops the server computes (boxes from the
    clean frames). Output is rounded to uint8 and sent losslessly (PNG)."""
    import torch
    import torch.nn.functional as F

    from detection.deepfake_detector import FACE_SCALE, INPUT_SIZE, DeepfakeDetector, _MEAN, _STD
    from detection.face_detector import crop_box
    from detection.presentation_attack import PresentationAttackDetector, _INPUT_SIZE, _REAL_CLASS

    if not _PGD:
        _PGD["pad"] = PresentationAttackDetector()
        _PGD["df"] = DeepfakeDetector()
    pad, df = _PGD["pad"], _PGD["df"]
    mean = torch.tensor(_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(_STD).view(1, 3, 1, 1)
    boxes = []
    for fr in frames:
        bgr = np.ascontiguousarray(fr[..., ::-1])
        bbox, _ = pad._face_detector.get_bbox(bgr)
        x, y, w, h = bbox
        side = max(w, h)
        square = [int(x + w / 2 - side / 2), int(y + h / 2 - side / 2), side, side]
        boxes.append(([crop_box(fr.shape, bbox, sc) for _, sc in pad._model], crop_box(fr.shape, square, FACE_SCALE)))
    clean = [torch.tensor(fr, dtype=torch.float32) for fr in frames]
    delta = [torch.zeros_like(c, requires_grad=True) for c in clean]
    alpha = max(1.0, eps / 4)

    def crop(img, box, size):  # img (H, W, 3) float -> (1, 3, size, size)
        x0, y0, x1, y1 = box
        patch = img[y0: y1 + 1, x0: x1 + 1].permute(2, 0, 1).unsqueeze(0)
        return F.interpolate(patch, size=(size, size), mode="bilinear", align_corners=False)

    for _ in range(steps):
        loss = 0.0
        df_in = []
        for c, d, (pad_boxes, df_box) in zip(clean, delta, boxes):
            img = (c + d).clamp(0, 255)
            for (net, _), box in zip(pad._model, pad_boxes):
                p = torch.softmax(net(crop(img, box, _INPUT_SIZE).flip(1)), dim=1)[0, _REAL_CLASS]
                loss = loss - torch.log(p + 1e-6)
            df_in.append(crop(img, df_box, INPUT_SIZE))
        logits = df._model(((torch.cat(df_in) / 255.0) - mean) / std).squeeze(1)
        loss = loss - F.logsigmoid(logits).sum()
        grads = torch.autograd.grad(loss, delta)
        with torch.no_grad():
            for d, g in zip(delta, grads):
                d -= alpha * g.sign()
                d.clamp_(-eps, eps)
    return [np.clip((c + d.detach()).round().numpy(), 0, 255).astype(np.uint8) for c, d in zip(clean, delta)]


# ---------------------------------------------------------------- part B6
def part_b6(clips, seconds: int = 60) -> list[dict]:
    """Victim (own device) submits every 6 s; an enrolled malicious device
    runs 4 threads of validly signed submissions. Digit challenges so every
    authenticated submission reaches the detectors."""
    import client_demo.api_client as client_mod
    from client_demo.device import DeviceIdentity

    out = []
    for cond, extra in [("no attack", {}), ("flood, limits off", {"MAX_SESSIONS_PER_MINUTE_PER_DEVICE": "1000000", "MAX_ACTIVE_SESSIONS_PER_DEVICE": "1000000"}),
                        ("flood, default limits (12/min, 3 open)", {"MAX_SESSIONS_PER_MINUTE_PER_DEVICE": "12", "MAX_ACTIVE_SESSIONS_PER_DEVICE": "3"})]:
        import tempfile

        server = start("after", {"CHALLENGE_KINDS": "digits", **extra} if extra else {"CHALLENGE_KINDS": "digits"})
        client_mod.API_BASE_URL = URL
        devs = {}
        for name in ("victim", "attacker"):
            devs[name] = DeviceIdentity(alg="ML-DSA-65", key_path=Path(tempfile.mkdtemp()) / "d.json")
        stop_at = time.time() + seconds
        stats = defaultdict(int)
        lat = []
        lock = threading.Lock()
        fb_all = [[png(f) for f in fr] for fr, _ in clips[:4]]
        burst = clips[0][1]

        def run_as(dev, fn):
            # api_client keeps one device per process; swap it under a lock for the few calls that need it.
            with lock:
                client_mod._device = dev
                return fn()

        def enroll(dev):
            return run_as(dev, client_mod.enroll_device)

        for d in devs.values():
            enroll(d)

        def one(dev, fb):
            try:
                s = run_as(dev, client_mod.session_start)
            except client_mod.ApiError as exc:
                return "429" if "429" in str(exc) else "error", None
            body = run_as(dev, lambda: envelope(client_mod, s, fb, burst))
            status, resp, ms = post(s["session_id"], body)
            return ("inference" if resp.get("detection_ran") else "rejected"), ms

        def attacker(k):
            while time.time() < stop_at:
                kind, _ = one(devs["attacker"], fb_all[k % 4])
                stats[kind] += 1
                if kind == "429":
                    time.sleep(0.05)

        threads = [threading.Thread(target=attacker, args=(k,)) for k in range(4)] if cond != "no attack" else []
        for t in threads:
            t.start()
        i = 0
        while time.time() < stop_at - 6:
            t0 = time.time()
            kind, ms = one(devs["victim"], fb_all[i % 4])
            if ms is not None:
                lat.append(ms)
            i += 1
            time.sleep(max(0.0, 6 - (time.time() - t0)))
        for t in threads:
            t.join()
        cpu = server.cpu_seconds()
        server.stop()
        out.append({"condition": cond, "seconds": seconds, "victim_ms_median": statistics.median(lat) if lat else None,
                    "victim_n": len(lat), "attacker_inferences": stats["inference"], "attacker_429": stats["429"],
                    "attacker_inferences_per_min": stats["inference"] * 60 / seconds, "server_cpu_s": cpu})
        print(out[-1], flush=True)
    return out


# ---------------------------------------------------------------- summary
def summarise(rows: list[dict], keys: list[str]) -> list[dict]:
    g = defaultdict(list)
    for r in rows:
        g[tuple(r[k] for k in keys)].append(r)
    res = []
    for k, rs in sorted(g.items()):
        d = dict(zip(keys, k))
        d["n"] = len(rs)
        for f in ("accepted", "session_survived", "liveness_passed", "detection_ran"):
            vals = [r[f] for r in rs if r.get(f) is not None]
            if vals:
                d[f] = sum(bool(v) for v in vals)
        ms = [r["server_ms"] for r in rs if r.get("server_ms") is not None]
        if ms:
            d["server_ms_median"] = statistics.median(ms)
        if any("http_ms" in r for r in rs):
            d["http_ms_median"] = statistics.median(r["http_ms"] for r in rs)
        d["failed_at"] = dict(sorted({r.get("binding_failed", r.get("victim_failed_check", "")): 0 for r in rs}.items()))
        for r in rs:
            key = r.get("binding_failed", r.get("victim_failed_check", ""))
            d["failed_at"][key] = d["failed_at"].get(key, 0) + 1
        d["vetoes"] = dict(defaultdict(int, {}))
        for r in rs:
            for v in (r.get("vetoes") or "").split("+"):
                if v:
                    d["vetoes"][v] = d["vetoes"].get(v, 0) + 1
        res.append(d)
    return res


def main() -> None:
    global BEFORE
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", default=str(BEFORE))
    ap.add_argument("--parts", default="A,B1,B2,B3,B4,B5,B6")
    args = ap.parse_args()
    BEFORE = Path(args.before)
    parts = set(args.parts.split(","))
    result_path = OUT / "d6_malicious_client.json"
    result = json.loads(result_path.read_text()) if result_path.exists() else {}

    bona = webcam_clips(12)
    if "A" in parts:
        rows = part_a("before", bona) + part_a("after", bona)
        write_csv(rows, "d6a_session_burning.csv")
        result["A"] = summarise([r for r in rows if not r["attack"].startswith("_")], ["attack", "tree"])
        result["A_forged_cost_ms"] = {r["tree"]: r["victim_failed_check"] for r in rows if r["attack"].startswith("_")}
        print(json.dumps(result["A"], indent=1), result["A_forged_cost_ms"], flush=True)
    bparts = parts & {"B1", "B2", "B3", "B4", "B5"}
    if bparts:
        data = {"bona_fide": bona}
        if "B1" in bparts:
            data["stills"] = stills(5)
        if "B2" in bparts:
            data["injection"] = injection_clips(5)
        if bparts & {"B3", "B5"}:
            data["pa"] = pa_clips(60)
            data["pa_pgd"] = data["pa"][::3]
        rows = part_b("before", bparts, data) + part_b("after", bparts, data)
        write_csv(rows, "d6b_authenticated_client.csv")
        for p in sorted(bparts):
            result[p] = summarise([r for r in rows if r["part"] == p], ["variant", "tree"])
            print(p, json.dumps(result[p], indent=1), flush=True)
    if "B6" in parts:
        result["B6"] = part_b6(bona)
    result_path.write_text(json.dumps(result, indent=1, default=str))


if __name__ == "__main__":
    main()
