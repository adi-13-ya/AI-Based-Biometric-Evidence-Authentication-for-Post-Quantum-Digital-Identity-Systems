"""System experiments for the revised report: a real two-process deployment.

The server runs as its own `uvicorn` process; this script is the client
process, holding the device ML-DSA key (client_demo/device.py) and talking
to the server over HTTP on localhost. Nothing is routed in-process.

  D1. Genuine sessions on real webcam footage (UniqueData sample, 'real'
      folder): 8 evenly spaced frames + an 8-frame consecutive burst per
      clip, as the live client sends them. Client-side and server-side stage
      latencies. Sessions are re-opened until a spoken-digit challenge is
      issued: a public pre-recorded video cannot turn its head on cue, so a
      head-motion challenge would (correctly) reject it at binding and the
      detectors would never run.
  D2. Cost of rejected requests under each check order (PIPELINE_MODE) and
      device signature algorithm: forged signature (attacker's own key;
      anyone can encapsulate to the session's ML-KEM key, so the channel
      alone proves nothing about the sender), reused nonce, stale
      timestamp, substituted frames. Full-size payloads (16 frames).
      Configurations (CONFIGS):
        integrated / ML-DSA-65   header-first gate (the protocol)
        buffered   / ML-DSA-65   receive + decrypt everything, then the
                                 same checks, then inference (baseline)
        layered    / ML-DSA-65   inference before the checks (ablation)
        integrated / ECDSA-P256  same protocol, classical signature
        buffered   / ECDSA-P256  same baseline, classical signature
  D4. Per-module inference latency, in-process, single thread, and model
      sizes.
  (D3, the flood at matched degradation, is report/scripts/flood_experiment.py.)

Run from the repo root:  python report/scripts/distributed_experiments.py
Outputs: report/data/v2/*.csv|json
"""
from __future__ import annotations

import base64
import csv
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
OUT = REPO / "report" / "data" / "v2"
PORT = 8765
N_GENUINE = 30
N_REJECT = 30

CONFIGS = [("integrated", "ML-DSA-65"), ("buffered", "ML-DSA-65"), ("layered", "ML-DSA-65"),
           ("integrated", "ECDSA-P256"), ("buffered", "ECDSA-P256")]

# One server identity for every run (so device files can pin it once per
# config); a fresh device key per configuration.
SERVER_KEY_PATH = str(Path(tempfile.mkdtemp()) / "server_identity.json")


def fresh_device(sig_alg: str) -> None:
    os.environ["PQBIO_DEVICE_KEY"] = str(Path(tempfile.mkdtemp()) / "device.json")
    os.environ["SIG_ALG"] = sig_alg


def write_csv(rows: list[dict], name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
    with open(OUT / name, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def stats(xs: list[float]) -> dict:
    xs = sorted(xs)
    return {"median": statistics.median(xs), "p95": xs[min(len(xs) - 1, int(round(0.95 * (len(xs) - 1))))], "mean": statistics.fmean(xs), "n": len(xs)}


# --------------------------------------------------------------------------
# Server process
# --------------------------------------------------------------------------
# The v1.2 measurements (D1-D5) open many sessions per device per minute and
# reopen sessions until a digit challenge is issued (so pre-recorded clips are
# judged by the detectors, not the motion check). These settings keep them
# reproducible; the per-device limits and the motion-only default are
# evaluated separately in report/scripts/malicious_client.py.
MEASUREMENT_ENV = {"CHALLENGE_KINDS": "motion,digits", "ALLOW_SAMPLE_UPLOADS": "false", "DEMO_MODE": "true", "MAX_SESSIONS_PER_MINUTE_PER_DEVICE": "1000000",
                   "MAX_ACTIVE_SESSIONS_PER_DEVICE": "1000000"}


class Server:
    def __init__(self, mode: str, extra_env: dict | None = None, repo: Path = REPO):
        from server.config import Settings

        # Every tunable from the code defaults (the calibrated values), so a
        # stale local .env cannot change what is measured.
        defaults = {name.upper(): str(f.default) for name, f in Settings.model_fields.items()
                    if name.startswith(("threshold_", "weight_", "fusion_", "pad_", "kem_", "sig_"))}
        env = {**os.environ, **defaults, "PIPELINE_MODE": mode, "DEMO_MODE": "true", "DETECTION_MODE": "pretrained",
               "SERVER_KEY_PATH": SERVER_KEY_PATH, **MEASUREMENT_ENV, **(extra_env or {})}
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "server.main:app", "--port", str(PORT), "--log-level", "warning"],
            cwd=repo, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        import requests

        for _ in range(240):
            try:
                if requests.get(f"http://127.0.0.1:{PORT}/health", timeout=1).ok:
                    return
            except requests.RequestException:
                time.sleep(0.5)
        raise RuntimeError("server did not start")

    def rss_mb(self) -> float:
        import psutil

        return psutil.Process(self.proc.pid).memory_info().rss / 2**20

    def cpu_seconds(self) -> float:
        import psutil

        t = psutil.Process(self.proc.pid).cpu_times()
        return t.user + t.system

    def stop(self) -> None:
        self.proc.terminate()
        self.proc.wait(timeout=30)


# --------------------------------------------------------------------------
# Real webcam clips
# --------------------------------------------------------------------------
def webcam_clips(n: int) -> list[list[np.ndarray]]:
    """n clips of 8 spread + 8 consecutive frames from the bona fide videos."""
    from client_demo.recorder import select_frames
    from evaluation.train_injection import decode

    videos = sorted((REPO / "datasets" / "webcam_liveness" / "files" / "real").glob("*.mp4"))
    decoded = [decode(v, 160) for v in videos]
    clips = []
    rng = np.random.default_rng(7)
    while len(clips) < n:
        frames = decoded[len(clips) % len(decoded)]
        start = int(rng.integers(0, max(1, len(frames) - 60)))
        clips.append(select_frames(frames[start: start + 60]))
    return clips


# --------------------------------------------------------------------------
# Client operations, timed stage by stage
# --------------------------------------------------------------------------
def api():
    import client_demo.api_client as client

    client.API_BASE_URL = f"http://127.0.0.1:{PORT}"
    return client


def digit_session(client) -> dict:
    for _ in range(50):
        session = client.session_start()
        if "number" in session["challenge"].lower():
            return session
        client.end_session(session)
    raise RuntimeError("no digit challenge issued in 50 tries")


def genuine_submission(client, frames, burst, session=None, signer=None, tamper=None) -> tuple[dict, dict]:
    """Sealed submission with per-stage client timings."""
    from client_demo.frame_utils import frame_to_png_bytes
    from crypto.channel import evidence_hash_of

    t = {}
    t0 = time.perf_counter()
    session = session or digit_session(client)
    t["session_start_ms"] = (time.perf_counter() - t0) * 1000
    response = client.challenge_answer(session["challenge"])

    t0 = time.perf_counter()
    frame_bytes = [frame_to_png_bytes(f) for f in frames]
    t["encode_frames_ms"] = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    evidence_hash = evidence_hash_of(frame_bytes)
    t["hash_ms"] = (time.perf_counter() - t0) * 1000

    ts = (datetime.now(timezone.utc) - (timedelta(seconds=120) if tamper == "stale" else timedelta())).isoformat()
    nonce = base64.b64decode(session["nonce"])
    t0 = time.perf_counter()
    kem = client.encapsulate(session)
    t["encaps_ms"] = (time.perf_counter() - t0) * 1000
    context = client.binding_context(session, kem[0], burst, None)
    t0 = time.perf_counter()
    if signer is None:
        signature = client.device().sign_binding(evidence_hash, nonce, ts, response, **context)
    else:
        from crypto.signature import build_binding_payload

        signature = signer.sign(build_binding_payload(evidence_hash, nonce, ts, response, **context))
    t["sign_ms"] = (time.perf_counter() - t0) * 1000

    if tamper == "swap":
        frame_bytes = [frame_to_png_bytes(np.ascontiguousarray(f[:, ::-1])) for f in frames]
    header = {"evidence_hash": evidence_hash.hex(), "timestamp": ts, "challenge_response": response,
              "signature": signature.hex(), "frame_timestamps": None, "burst_frames": burst}
    t0 = time.perf_counter()
    envelope = client.seal_submission(session, header, frame_bytes, kem)
    t["seal_ms"] = (time.perf_counter() - t0) * 1000
    t["payload_bytes"] = len(envelope)

    t0 = time.perf_counter()
    resp = client.post_sealed(session, envelope)
    t["verify_http_ms"] = (time.perf_counter() - t0) * 1000
    if tamper == "reuse":
        t0 = time.perf_counter()
        resp = client.post_sealed(session, envelope)
        t["verify_http_ms"] = (time.perf_counter() - t0) * 1000
    return resp, t


# --------------------------------------------------------------------------
# D1 / D2
# --------------------------------------------------------------------------
def run_config(mode: str, sig_alg: str, clips) -> dict:
    from crypto.signature import SignerIdentity

    fresh_device(sig_alg)
    server = Server(mode)
    client = api()
    client._device = None
    out = {"mode": mode, "sig_alg": sig_alg, "genuine": [], "reject": []}
    try:
        client.enroll_device()
        rss0 = server.rss_mb()
        for i, (frames, burst) in enumerate(clips[:N_GENUINE]):
            client.reset_replay_store()
            resp, t = genuine_submission(client, frames, burst)
            row = {"i": i, **t, **{f"server_{k}": v for k, v in resp["timing_ms"].items()},
                   "binding_accepted": resp["binding_accepted"], "accepted": resp["accepted"],
                   **{f"score_{k}": v for k, v in (resp.get("detection_scores") or {}).items()}}
            out["genuine"].append(row)
        out["rss_mb_after_genuine"] = server.rss_mb()
        out["rss_mb_start"] = rss0

        attacker = SignerIdentity(alg=sig_alg)
        for kind in ["forged_signature", "reused_nonce", "stale_timestamp", "substituted_frames"]:
            for i in range(N_REJECT):
                frames, burst = clips[i % len(clips)]
                cpu0 = server.cpu_seconds()
                resp, t = genuine_submission(
                    client, frames, burst,
                    signer=attacker if kind == "forged_signature" else None,
                    tamper={"reused_nonce": "reuse", "stale_timestamp": "stale", "substituted_frames": "swap"}.get(kind),
                )
                out["reject"].append({"kind": kind, "i": i, "accepted": resp["accepted"], "detection_ran": resp["detection_ran"],
                                      "verify_http_ms": t["verify_http_ms"], "server_cpu_ms": (server.cpu_seconds() - cpu0) * 1000,
                                      "payload_bytes": t["payload_bytes"],
                                      **{f"server_{k}": v for k, v in resp["timing_ms"].items()},
                                      "failed_check": next((k for k, v in resp["binding_checks"].items() if not v), "")})
        attacker.close()
    finally:
        server.stop()
    return out


# --------------------------------------------------------------------------
# D4: per-module latency
# --------------------------------------------------------------------------
def module_latency(clips) -> list[dict]:
    from detection.deepfake_detector import WEIGHTS_PATH as DF_W, DeepfakeDetector
    from detection.injection_detector import InjectionDetector
    from detection.liveness_challenge import evaluate_challenge
    from detection.morph_detector import MorphDetector
    from detection.presentation_attack import FINETUNED_DIR, WEIGHTS_DIR, PresentationAttackDetector
    from detection.replay_detector import ReplayDetector

    pad, morph, df, inj, rep = PresentationAttackDetector(), MorphDetector(), DeepfakeDetector(), InjectionDetector(), ReplayDetector()
    rows = []

    def timeit(name, fn, n=30, model="", size_mb=None):
        fn(0)
        ts = []
        for i in range(n):
            t0 = time.perf_counter()
            fn(i)
            ts.append((time.perf_counter() - t0) * 1000)
        rows.append({"module": name, "model": model, "size_mb": size_mb, **{f"{k}_ms": v for k, v in stats(ts).items() if k != "n"}, "n": n})

    def mb(paths):
        return round(sum(Path(p).stat().st_size for p in paths if Path(p).exists()) / 2**20, 2)

    pad_dir = FINETUNED_DIR if pad.weights == "finetuned" else WEIGHTS_DIR
    timeit("presentation_attack", lambda i: pad.predict(clips[i % len(clips)][0][0]), model=f"RetinaFace + 2x MiniFASNet ({pad.weights})",
           size_mb=mb(list(pad_dir.glob("*.pth")) + list(WEIGHTS_DIR.glob("*.caffemodel"))))
    timeit("deepfake (8 frames)", lambda i: df.predict_video(clips[i % len(clips)][0][:8]), model="RetinaFace + EfficientNet-B0", size_mb=mb([DF_W]))
    timeit("injection (8-frame burst)", lambda i: inj.predict_sequence(clips[i % len(clips)][0][-8:]), model="features + logistic regression")
    timeit("morph", lambda i: morph.predict(clips[i % len(clips)][0][0]), model="signal processing (MediaPipe crop)")
    timeit("replay", lambda i: rep.check_and_record(clips[i % len(clips)][0][0], session_id=str(i)), model="pHash + residual correlation")
    timeit("liveness (MediaPipe)", lambda i: evaluate_challenge("Turn your head slightly left", "x", clips[i % len(clips)][0][:8]), n=10, model="MediaPipe Face Mesh")
    return rows


def tag(mode: str, sig_alg: str) -> str:
    return f"{mode}_{sig_alg.replace('-', '').lower()}"


def summarise(res: dict) -> dict:
    rej = res["reject"]
    return {
        "mode": res["mode"], "sig_alg": res["sig_alg"],
        "rss_mb_start": res["rss_mb_start"], "rss_mb_after_genuine": res["rss_mb_after_genuine"],
        "genuine_accept_rate": float(np.mean([r["accepted"] for r in res["genuine"]])),
        "genuine_verify_http_ms": stats([r["verify_http_ms"] for r in res["genuine"]]),
        "genuine_sign_ms": stats([r["sign_ms"] for r in res["genuine"]]),
        "genuine_server_total_ms": stats([r["server_total_ms"] for r in res["genuine"]]),
        "reject": {k: {"verify_http_ms": stats([r["verify_http_ms"] for r in rej if r["kind"] == k]),
                       "server_total_ms": stats([r["server_total_ms"] for r in rej if r["kind"] == k]),
                       "server_cpu_ms": stats([r["server_cpu_ms"] for r in rej if r["kind"] == k]),
                       "detection_ran": any(r["detection_ran"] for r in rej if r["kind"] == k),
                       "all_rejected": not any(r["accepted"] for r in rej if r["kind"] == k),
                       "failed_check": sorted({r["failed_check"] for r in rej if r["kind"] == k})}
                   for k in sorted({r["kind"] for r in rej})},
    }


def main() -> None:
    sys.path.insert(0, str(REPO / "report" / "scripts"))
    from system_experiments import bench_primitives  # S1, unchanged from the first report

    OUT.mkdir(parents=True, exist_ok=True)
    clips = webcam_clips(max(N_GENUINE, 12))
    print(f"{len(clips)} clips, {len(clips[0][0])} frames each ({clips[0][1]} burst), {clips[0][0][0].shape}")

    only = sys.argv[1:]
    if not only or "d4" in only:
        print("D4 module latency")
        write_csv(module_latency(clips), "d4_module_latency.csv")
        print("S1 primitives")
        write_csv(bench_primitives(), "s1_primitives.csv")

    summary_path = OUT / "d_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() and only else {}
    summary = {k: v for k, v in summary.items() if "_" in k}  # drop pre-v1.1 keys ("integrated", "layered")
    for mode, sig_alg in CONFIGS:
        name = tag(mode, sig_alg)
        if only and name not in only and "d12" not in only:
            continue
        print(f"config {name}")
        res = run_config(mode, sig_alg, clips)
        write_csv(res["genuine"], f"d1_genuine_{name}.csv")
        write_csv(res["reject"], f"d2_reject_{name}.csv")
        summary[name] = summarise(res)
        print(json.dumps(summary[name]["reject"]["forged_signature"]["server_total_ms"]))
        summary_path.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
