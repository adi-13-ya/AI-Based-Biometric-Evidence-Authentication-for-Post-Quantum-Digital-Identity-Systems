"""D6: session safety, concurrency and authenticated malicious clients.

Every row runs the deployed server and, where a defence is responsible for
the result, the same server with that one defence switched off (an
ablation flag in server/config.py), on the same evidence and machine.
Recorded clips cannot answer a head-motion challenge, so all parts except B1
use the typed-digit challenge (distributed_experiments.MEASUREMENT_ENV).

Adversaries (report Section 3.1):
  N  network adversary without a device key: sees session ids, offers and
     envelopes on the path, can send, reorder, corrupt and drop requests.
  M  malicious client holding one (or K) enrolled device keys: signs
     anything it likes, controls frame content, count, size, burst length.

Parts
  A   session safety (N): after each invalid request against a live session
      the device sends its own submission; does it still pass the binding
      gate? Ablation ABLATE_EARLY_NONCE_CONSUME. Also the server cost of a
      rejected forged submission.
  A2  forged session starts (N): unsigned / wrongly signed / replayed
      start requests; status and server CPU per request.
  C   concurrency: k concurrent copies of one valid submission (exactly one
      admitted?); one valid submission racing 16 forged ones; latency with
      1-8 devices submitting at once.
  B1  single-frame and still-stream submissions under a motion challenge (M).
      Ablation: ENFORCE_SUBMISSION_SHAPE=false, STRICT_LIVENESS=false.
  B2  burst_frames rewritten on re-encoded injections (M). Ablation: shape.
  B3  bona fide frame spliced in front of webcam attacks (M). Ablation:
      PAD_FRAMES=1.
  B4  oversized submissions (M). Ablation: shape off, body cap 1 GB.
  B5  white-box PGD against MiniFASNet + EfficientNet-B0 (M), final only.
  B6  one enrolled key floods valid submissions, limits on / off (M).
  B7  K = 1, 2, 4, 8 enrolled keys flood together, default limits (M, Sybil).

Run (port 8765 free):  python report/scripts/malicious_client.py [--parts A,A2,C,B1,...]
Writes report/data/v2/d6_malicious_client.json and d6*_*.csv
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import statistics
import sys
import tempfile
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "report" / "scripts"))

from distributed_experiments import OUT, PORT, Server, api, fresh_device, webcam_clips, write_csv  # noqa: E402

URL = f"http://127.0.0.1:{PORT}"
WEBCAM = REPO / "datasets" / "webcam_liveness" / "files"
RNG = np.random.default_rng(1116)
DEFAULT_LIMITS = {"MAX_SESSIONS_PER_MINUTE_PER_DEVICE": "12", "MAX_ACTIVE_SESSIONS_PER_DEVICE": "3"}

ABLATIONS = {
    "final": {},
    "early nonce consumption": {"ABLATE_EARLY_NONCE_CONSUME": "true"},
    "no shape check, lenient liveness": {"ENFORCE_SUBMISSION_SHAPE": "false", "STRICT_LIVENESS": "false"},
    "no shape check": {"ENFORCE_SUBMISSION_SHAPE": "false"},
    "PAD on first frame only": {"PAD_FRAMES": "1"},
    "no shape check, no body cap": {"ENFORCE_SUBMISSION_SHAPE": "false", "MAX_BODY_BYTES": str(2**30)},
}


# ----------------------------------------------------------------- plumbing
def png(frame: np.ndarray) -> bytes:
    from client_demo.frame_utils import frame_to_png_bytes

    return frame_to_png_bytes(np.ascontiguousarray(frame))


def session_with(client, kind: str) -> dict:
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


def submit(client, kind: str, frames, burst: int, **kw) -> dict:
    client.reset_replay_store()
    s = session_with(client, kind)
    fb = frames if isinstance(frames[0], bytes) else [png(f) for f in frames]
    status, body, ms = post(s["session_id"], envelope(client, s, fb, burst, **kw))
    return {**outcome(status, body), "http_ms": ms}


def start(config: str, extra: dict | None = None) -> Server:
    return Server("integrated", extra_env={**ABLATIONS[config], **(extra or {})})


def new_client():
    fresh_device("ML-DSA-65")
    client = api()
    client._device = None
    client.enroll_device()
    return client


# ---------------------------------------------------------------- data
def pa_clips(n: int):
    from client_demo.recorder import select_frames
    from evaluation.train_injection import decode

    species = sorted(p.name for p in WEBCAM.iterdir() if p.is_dir() and p.name != "real")
    vids = [(sp, v) for sp in species for v in sorted((WEBCAM / sp).glob("*.mp4"))]
    out = []
    for sp, v in vids[:n]:
        f, burst = select_frames(decode(v, 160)[:60])
        out.append((sp, f, burst))
    return out


def stills(per: int):
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


def injection_clips(per_condition: int):
    from client_demo.recorder import select_frames
    from evaluation.train_injection import CONDITIONS, decode, virtual_camera

    decoded = [decode(v, 160) for v in sorted((WEBCAM / "real").glob("*.mp4"))[:per_condition]]
    out = []
    for cond, (fps, crf) in CONDITIONS.items():
        if crf is None:
            continue
        for frames in decoded:
            f, burst = select_frames(virtual_camera(frames, fps, crf, 90)[:60])
            out.append((cond, f, burst))
    return out


# ---------------------------------------------------------------- A: session safety
A_KINDS = ["none (control)", "body that cannot be parsed", "envelope signed with the attacker's key",
           "envelope sealed for another session", "on path: device envelope, frames bit-flipped, sent first",
           "on path: device envelope, truncated, sent first", "unsigned session close"]


def part_a(config: str, clips) -> list[dict]:
    import requests

    from crypto.signature import SignerIdentity

    server = start(config)
    rows = []
    try:
        client = new_client()
        attacker = SignerIdentity(alg="ML-DSA-65")
        for kind in A_KINDS:
            for i in range(10):
                frames, burst = clips[i % len(clips)]
                fb = [png(f) for f in frames]
                client.reset_replay_store()
                s = session_with(client, "digits")
                own = envelope(client, s, fb, burst)
                if kind.startswith("body"):
                    post(s["session_id"], b"PQB1" + os.urandom(64) + os.urandom(20000))
                elif kind.startswith("envelope signed"):
                    post(s["session_id"], envelope(client, s, fb, burst, signer=attacker))
                elif kind.startswith("envelope sealed"):
                    post(s["session_id"], envelope(client, session_with(client, "digits"), fb, burst))
                elif "bit-flipped" in kind:
                    bad = bytearray(own)
                    bad[-100] ^= 1
                    post(s["session_id"], bytes(bad))
                elif "truncated" in kind:
                    post(s["session_id"], own[: len(own) // 2])
                elif kind.startswith("unsigned"):
                    requests.delete(f"{URL}/session/{s['session_id']}", timeout=10)
                status, body, _ = post(s["session_id"], own)
                checks = body.get("binding_checks") or {}
                survived = status == 200 and checks.get("signature_valid") is True and checks.get("nonce_unused") is True \
                    and checks.get("nonce_consumed", True) is True
                rows.append({"config": config, "attack": kind, "i": i, "http": status, "session_survived": survived,
                             "device_accepted": bool(body.get("accepted")),
                             "device_failed_check": next((k for k, v in checks.items() if not v), "") if status == 200 else f"HTTP {status}"})
        costs = []
        frames, burst = clips[0]
        fb = [png(f) for f in frames]
        s = session_with(client, "digits")
        for _ in range(100):  # one observed session, reused: forged requests do not consume it
            _, body, _ = post(s["session_id"], envelope(client, s, fb, burst, signer=attacker))
            costs.append(body["timing_ms"]["total_ms"])
        rows.append({"config": config, "attack": "_forged_cost_ms", "i": 0, "http": 200, "session_survived": None,
                     "device_accepted": None, "device_failed_check": json.dumps({"median": statistics.median(costs),
                                                                                  "p95": sorted(costs)[94], "n": len(costs)})})
    finally:
        server.stop()
    return rows


def part_a2() -> dict:
    """Forged session starts: what they cost and that none succeeds."""
    import requests

    from crypto.signature import SignerIdentity, build_session_request

    server = start("final", DEFAULT_LIMITS)
    out = {}
    try:
        client = new_client()
        device_id = client.enroll_device()
        attacker = SignerIdentity(alg="ML-DSA-65")
        good = client.session_request(device_id)
        requests.post(f"{URL}/session/start", json=good, timeout=10)
        kinds = {
            "unsigned": lambda: {"device_id": device_id},
            "signed with the attacker's key": lambda: (lambda ts, rn: {"device_id": device_id, "timestamp": ts, "request_nonce": rn.hex(),
                "signature": attacker.sign(build_session_request(device_id=device_id, timestamp=ts, request_nonce=rn)).hex()})(
                client.now_iso(), os.urandom(16)),
            "replay of a valid start request": lambda: good,
        }
        for name, body in kinds.items():
            statuses, ms = defaultdict(int), []
            cpu0 = server.cpu_seconds()
            for _ in range(300):
                b = body()
                t0 = time.perf_counter()
                r = requests.post(f"{URL}/session/start", json=b, timeout=10)
                ms.append((time.perf_counter() - t0) * 1000)
                statuses[r.status_code] += 1
            out[name] = {"n": 300, "status": dict(statuses), "http_ms_median": statistics.median(ms),
                         "server_cpu_ms_per_request": (server.cpu_seconds() - cpu0) * 1000 / 300}
            print("A2", name, out[name], flush=True)
    finally:
        server.stop()
    return out


# ---------------------------------------------------------------- C: concurrency
def part_c(clips) -> dict:
    from crypto.signature import SignerIdentity
    from client_demo.device import DeviceIdentity

    import client_demo.api_client as client_mod

    server = start("final")
    out = {"copies": [], "race": [], "load": []}
    try:
        client = new_client()
        attacker = SignerIdentity(alg="ML-DSA-65")
        fb_all = [[png(f) for f in fr] for fr, _ in clips[:8]]
        burst = clips[0][1]

        def concurrently(bodies_and_ids):
            res = [None] * len(bodies_and_ids)
            barrier = threading.Barrier(len(bodies_and_ids))

            def go(j, sid, body):
                barrier.wait()
                res[j] = post(sid, body)

            ts = [threading.Thread(target=go, args=(j, sid, b)) for j, (sid, b) in enumerate(bodies_and_ids)]
            for t in ts:
                t.start()
            for t in ts:
                t.join()
            return res

        for k in (2, 8, 32):
            for trial in range(10):
                client.reset_replay_store()
                s = session_with(client, "digits")
                body = envelope(client, s, fb_all[trial % 8], burst)
                res = concurrently([(s["session_id"], body)] * k)
                admitted = sum(1 for st, b, _ in res if st == 200 and (b.get("binding_checks") or {}).get("nonce_consumed") is True)
                out["copies"].append({"k": k, "trial": trial, "admitted": admitted,
                                      "rejected_as": sorted({next((c for c, v in (b.get("binding_checks") or {}).items() if not v), "")
                                                             for st, b, _ in res})})
            print("C copies", k, [r["admitted"] for r in out["copies"] if r["k"] == k], flush=True)
        for trial in range(10):
            client.reset_replay_store()
            s = session_with(client, "digits")
            valid = envelope(client, s, fb_all[trial % 8], burst)
            forged = [envelope(client, s, fb_all[(trial + j) % 8], burst, signer=attacker) for j in range(16)]
            res = concurrently([(s["session_id"], valid)] + [(s["session_id"], f) for f in forged])
            st, b, _ = res[0]
            out["race"].append({"trial": trial, "valid_admitted": st == 200 and (b.get("binding_checks") or {}).get("nonce_consumed") is True,
                                "valid_accepted": bool(b.get("accepted")),
                                "forged_admitted": sum(1 for st2, b2, _ in res[1:] if (b2.get("binding_checks") or {}).get("signature_valid"))})
        print("C race", out["race"], flush=True)
        # Load: D devices submit genuine clips at the same moment, 3 rounds.
        devices = []
        for _ in range(8):
            d = DeviceIdentity(alg="ML-DSA-65", key_path=Path(tempfile.mkdtemp()) / "d.json")
            client_mod._device = d
            client_mod.enroll_device()
            devices.append(d)
        lock = threading.Lock()

        def prepared(dev, fb):
            with lock:
                client_mod._device = dev
                s = client_mod.session_start()
                return s["session_id"], envelope(client_mod, s, fb, burst)

        for n_dev in (1, 2, 4, 8):
            lat, acc = [], 0
            t_all = time.perf_counter()
            for rnd in range(3):
                client_mod.reset_replay_store()
                jobs = [prepared(devices[j], fb_all[(j + rnd) % 8]) for j in range(n_dev)]
                for st, b, ms in concurrently(jobs):
                    lat.append(ms)
                    acc += bool(b.get("accepted"))
            out["load"].append({"devices": n_dev, "submissions": len(lat), "accepted": acc, "median_ms": statistics.median(lat),
                                "max_ms": max(lat)})
            print("C load", out["load"][-1], flush=True)
    finally:
        server.stop()
    return out


# ---------------------------------------------------------------- B: authenticated client
def part_b(config: str, parts: set[str], data: dict, extra: dict | None = None) -> list[dict]:
    server = start(config, extra)
    rows = []

    def add(part, variant, label, r):
        rows.append({"config": config, "part": part, "variant": variant, "item": label, **r})

    try:
        client = new_client()
        if "B1" in parts:
            for label, img in data["stills"]:
                add("B1", "still image as 16 frames", label, submit(client, "motion", [img] * 16, 8))
                add("B1", "still image as one frame", label, submit(client, "motion", [img], 0))
        if "B2" in parts:
            for label, f, burst in data["injection"]:
                for b in (burst, 2, 0):
                    add("B2", f"burst_frames={b}", label, submit(client, "digits", f, b))
        if "B3" in parts:
            live = data["bona_fide"]
            for k, (label, f, burst) in enumerate(data["pa"]):
                add("B3", "unaltered", label, submit(client, "digits", f, burst))
                add("B3", "bona fide frame spliced first", label, submit(client, "digits", [live[k % len(live)][0][0]] + f[1:], burst))
        if "B4" in parts:
            from PIL import Image

            base = data["bona_fide"][0][0]
            big = [np.asarray(Image.fromarray(x).resize((4000, 3000))) for x in base]
            hd = [np.asarray(Image.fromarray(x).resize((1920, 1080))) for x in base]
            for _ in range(3):
                add("B4", "16 frames of 1920x1080", "hd", submit(client, "digits", hd, 8))
                add("B4", "16 frames of 4000x3000", "12MP", submit(client, "digits", big, 8))
                add("B4", "64 frames", "64", submit(client, "digits", base * 4, 8))
        if "B5" in parts:
            for eps in (0, 4, 8):
                for label, f, burst in data["pa_pgd"]:
                    add("B5", f"PGD eps={eps}/255", label, submit(client, "digits", pgd(f, eps) if eps else f, burst))
    finally:
        server.stop()
    return rows


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
        _PGD["pad"], _PGD["df"] = PresentationAttackDetector(), DeepfakeDetector()
    pad, df = _PGD["pad"], _PGD["df"]
    mean, std = torch.tensor(_MEAN).view(1, 3, 1, 1), torch.tensor(_STD).view(1, 3, 1, 1)
    boxes = []
    for fr in frames:
        bbox, _ = pad._face_detector.get_bbox(np.ascontiguousarray(fr[..., ::-1]))
        x, y, w, h = bbox
        side = max(w, h)
        square = [int(x + w / 2 - side / 2), int(y + h / 2 - side / 2), side, side]
        boxes.append(([crop_box(fr.shape, bbox, sc) for _, sc in pad._model], crop_box(fr.shape, square, FACE_SCALE)))
    clean = [torch.tensor(fr, dtype=torch.float32) for fr in frames]
    delta = [torch.zeros_like(c, requires_grad=True) for c in clean]
    alpha = max(1.0, eps / 4)

    def crop(img, box, size):
        x0, y0, x1, y1 = box
        return F.interpolate(img[y0: y1 + 1, x0: x1 + 1].permute(2, 0, 1).unsqueeze(0), size=(size, size), mode="bilinear", align_corners=False)

    for _ in range(steps):
        loss, df_in = 0.0, []
        for c, d, (pad_boxes, df_box) in zip(clean, delta, boxes):
            img = (c + d).clamp(0, 255)
            for (net, _), box in zip(pad._model, pad_boxes):
                loss = loss - torch.log(torch.softmax(net(crop(img, box, _INPUT_SIZE).flip(1)), dim=1)[0, _REAL_CLASS] + 1e-6)
            df_in.append(crop(img, df_box, INPUT_SIZE))
        loss = loss - F.logsigmoid(df._model(((torch.cat(df_in) / 255.0) - mean) / std).squeeze(1)).sum()
        grads = torch.autograd.grad(loss, delta)
        with torch.no_grad():
            for d, g in zip(delta, grads):
                d -= alpha * g.sign()
                d.clamp_(-eps, eps)
    return [np.clip((c + d.detach()).round().numpy(), 0, 255).astype(np.uint8) for c, d in zip(clean, delta)]


# ---------------------------------------------------------------- B6 / B7: floods by enrolled keys
def flood(clips, attackers: int, limits: dict, seconds: int = 60, threads_per_key: int = 2) -> dict:
    """Genuine traffic from 6 devices in turn (one verification every 6 s,
    within any device's limit); `attackers` enrolled keys, each with
    `threads_per_key` threads of validly signed submissions."""
    import client_demo.api_client as client_mod
    from client_demo.device import DeviceIdentity

    server = start("final", {**limits})
    client_mod.API_BASE_URL = URL
    mk = lambda: DeviceIdentity(alg="ML-DSA-65", key_path=Path(tempfile.mkdtemp()) / "d.json")  # noqa: E731
    victims = [mk() for _ in range(6)]
    keys = [mk() for _ in range(attackers)]
    lock = threading.Lock()

    def as_dev(dev, fn):
        with lock:
            client_mod._device = dev
            return fn()

    for d in victims + keys:
        as_dev(d, client_mod.enroll_device)
    fb_all = [[png(f) for f in fr] for fr, _ in clips[:4]]
    burst = clips[0][1]
    stats, lat = defaultdict(int), []
    stop_at = time.time() + seconds
    cpu0 = server.cpu_seconds()

    def one(dev, fb):
        try:
            s = as_dev(dev, client_mod.session_start)
        except client_mod.ApiError as exc:
            return ("429" if "429" in str(exc) else "error"), None
        body = as_dev(dev, lambda: envelope(client_mod, s, fb, burst))
        _, resp, ms = post(s["session_id"], body)
        return ("inference" if resp.get("detection_ran") else "rejected"), ms

    def attack(dev, k):
        while time.time() < stop_at:
            kind, _ = one(dev, fb_all[k % 4])
            stats[kind] += 1
            if kind == "429":
                time.sleep(0.05)

    ts = [threading.Thread(target=attack, args=(d, k)) for d in keys for k in range(threads_per_key)]
    for t in ts:
        t.start()
    i = 0
    while time.time() < stop_at - 6:
        t0 = time.time()
        kind, ms = one(victims[i % len(victims)], fb_all[i % 4])
        if ms is not None:
            lat.append(ms)
        i += 1
        time.sleep(max(0.0, 6 - (time.time() - t0)))
    for t in ts:
        t.join()
    cpu = server.cpu_seconds() - cpu0
    server.stop()
    return {"attacker_keys": attackers, "seconds": seconds, "genuine_ms_median": statistics.median(lat) if lat else None,
            "genuine_n": len(lat), "attacker_inferences": stats["inference"], "attacker_inferences_per_min": stats["inference"] * 60 / seconds,
            "attacker_starts_refused_429": stats["429"], "server_cpu_s_per_s": cpu / seconds}


# ---------------------------------------------------------------- summary
def summarise(rows: list[dict], keys: list[str]) -> list[dict]:
    g = defaultdict(list)
    for r in rows:
        g[tuple(r[k] for k in keys)].append(r)
    res = []
    for k, rs in sorted(g.items()):
        d = dict(zip(keys, k))
        d["n"] = len(rs)
        for f in ("accepted", "session_survived", "liveness_passed", "detection_ran", "device_accepted"):
            vals = [r[f] for r in rs if r.get(f) is not None]
            if vals:
                d[f] = sum(bool(v) for v in vals)
        ms = [r["server_ms"] for r in rs if r.get("server_ms") is not None]
        if ms:
            d["server_ms_median"] = statistics.median(ms)
        if any("http_ms" in r for r in rs):
            d["http_ms_median"] = statistics.median(r["http_ms"] for r in rs)
        d["failed_at"] = defaultdict(int)
        d["vetoes"] = defaultdict(int)
        for r in rs:
            d["failed_at"][r.get("binding_failed", r.get("device_failed_check", ""))] += 1
            for v in (r.get("vetoes") or "").split("+"):
                if v:
                    d["vetoes"][v] += 1
        res.append(d)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="A,A2,C,B1,B2,B3,B4,B5,B6,B7")
    parts = set(ap.parse_args().parts.split(","))
    path = OUT / "d6_malicious_client.json"
    result = json.loads(path.read_text()) if path.exists() else {}
    result = {k: v for k, v in result.items() if k in {"A", "A_forged_cost_ms", "A2", "C", "B1", "B2", "B3", "B4", "B5", "B6", "B7"}}

    def save():
        path.write_text(json.dumps(result, indent=1, default=str))

    bona = webcam_clips(12)
    if "A" in parts:
        rows = part_a("final", bona) + part_a("early nonce consumption", bona)
        write_csv(rows, "d6a_session_safety.csv")
        result["A"] = summarise([r for r in rows if not r["attack"].startswith("_")], ["attack", "config"])
        result["A_forged_cost_ms"] = {r["config"]: json.loads(r["device_failed_check"]) for r in rows if r["attack"].startswith("_")}
        save()
    if "A2" in parts:
        result["A2"] = part_a2()
        save()
    if "C" in parts:
        result["C"] = part_c(bona)
        save()
    data = {"bona_fide": bona}
    if "B1" in parts:
        data["stills"] = stills(5)
    if "B2" in parts:
        data["injection"] = injection_clips(5)
    if parts & {"B3", "B5"}:
        data["pa"] = pa_clips(60)
        data["pa_pgd"] = data["pa"][::3]
    plan = [("B1", ["final", "no shape check, lenient liveness"]), ("B2", ["final", "no shape check"]),
            ("B3", ["final", "PAD on first frame only"]), ("B4", ["final", "no shape check, no body cap"]), ("B5", ["final"])]
    all_rows = []
    for part, configs in plan:
        if part not in parts:
            continue
        rows = []
        for config in configs:
            # B1 is the only part about the motion challenge itself.
            rows += part_b(config, {part}, data, {"CHALLENGE_KINDS": "motion"} if part == "B1" else None)
        all_rows += rows
        result[part] = summarise(rows, ["variant", "config"])
        print(part, json.dumps(result[part], indent=1, default=str), flush=True)
        save()
    if all_rows:
        write_csv(all_rows, "d6b_authenticated_client.csv")
    if "B6" in parts:
        result["B6"] = [dict(flood(bona, 1, {"MAX_SESSIONS_PER_MINUTE_PER_DEVICE": "1000000", "MAX_ACTIVE_SESSIONS_PER_DEVICE": "1000000"}), limits="off"),
                        dict(flood(bona, 1, DEFAULT_LIMITS), limits="default")]
        print("B6", result["B6"], flush=True)
        save()
    if "B7" in parts:
        result["B7"] = [dict(flood(bona, 0, DEFAULT_LIMITS), limits="default")]
        for k in (1, 2, 4, 8):
            result["B7"].append(dict(flood(bona, k, DEFAULT_LIMITS), limits="default"))
            print("B7", result["B7"][-1], flush=True)
        save()


if __name__ == "__main__":
    main()
