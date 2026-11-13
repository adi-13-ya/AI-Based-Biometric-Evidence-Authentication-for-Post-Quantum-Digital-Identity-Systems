"""D3: attacker effort at matched degradation of genuine-user service.

The question is not "how many forged requests did the server absorb in 20 s"
(that depends on how hard the attackers happened to push, and the genuine
user ended up slowed by different amounts in each configuration), but:

    What sustained rate of forged submissions does an attacker need to
    double the median verification latency of a genuine user?

For each configuration (check order x device signature algorithm, the same
CONFIGS as distributed_experiments.py) a fresh server process is started.
Then, REPEATS times:
  1. the victim runs alone for SECONDS -> baseline median latency L0;
  2. for each target rate in RATES[mode], ATTACK_PROCS attacker processes
     send forged-signature submissions paced to that combined rate (or as
     fast as they can, if they cannot keep up), while the victim keeps
     verifying -> achieved forged rate r and victim median L(r);
  3. the rate at which L(r) / L0 = DEGRADATION is interpolated (linear in
     log r between the two bracketing points). If the highest rate tried
     never reaches it, the result is a lower bound and is flagged.

Attackers and victim are separate OS processes (no shared GIL). An attacker
prepares one realistic payload once (16 PNG frames from a real webcam clip,
about 5.7 MB), then loops: open a session for the victim's enrolled device
(device ids are not secret), encapsulate to the session's ML-KEM key, sign
with its own key (same algorithm as the victim's device), seal, submit.
Only the signature check can tell the request is not from the victim.

Run: python report/scripts/flood_experiment.py [config ...]
Writes report/data/v2/d3_flood_matched.json
"""
from __future__ import annotations

import base64
import json
import math
import multiprocessing as mp
import os
import statistics
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "report" / "scripts"))

PORT = 8765
SECONDS = 15
ATTACK_PROCS = 4
REPEATS = 3
DEGRADATION = 2.0
# Target combined forged-request rates (per second). Chosen from a pilot so
# each sweep brackets the 2x point; 0 = as fast as the attackers can go.
RATES = {
    "integrated": [10, 25, 50, 100, 0],
    "buffered": [5, 10, 20, 40, 0],
    "layered": [0.5, 1, 2, 4],
}
OUT = REPO / "report" / "data" / "v2" / "d3_flood_matched.json"


def _api(key_path: str, sig_alg: str):
    os.environ["PQBIO_DEVICE_KEY"] = key_path
    os.environ["SIG_ALG"] = sig_alg
    import client_demo.api_client as client

    client.API_BASE_URL = f"http://127.0.0.1:{PORT}"
    return client


PAYLOAD_CACHE = Path(tempfile.gettempdir()) / "pqbio_flood_payloads.json"


def _prepare_payloads(n: int) -> None:
    """Encode the first n webcam clips once (every spawned process would
    otherwise decode all the videos again)."""
    from client_demo.frame_utils import frame_to_png_bytes
    from distributed_experiments import webcam_clips

    clips = webcam_clips(n)
    data = [{"burst": burst, "frames": [frame_to_png_bytes(f).hex() for f in frames]} for frames, burst in clips[:n]]
    PAYLOAD_CACHE.write_text(json.dumps(data))


def _payload_parts(clip_index: int):
    from crypto.channel import evidence_hash_of

    item = json.loads(PAYLOAD_CACHE.read_text())[clip_index]
    frame_bytes = [bytes.fromhex(f) for f in item["frames"]]
    return evidence_hash_of(frame_bytes), frame_bytes, item["burst"]


def _submit(client, session, evidence_hash, frame_bytes, burst, sign) -> tuple[float, dict]:
    nonce = base64.b64decode(session["nonce"])
    ts = datetime.now(timezone.utc).isoformat()
    resp = client.challenge_answer(session["challenge"])
    kem = client.encapsulate(session)
    context = client.binding_context(session, kem[0], burst, None)
    header = {"evidence_hash": evidence_hash.hex(), "timestamp": ts, "challenge_response": resp,
              "signature": sign(evidence_hash, nonce, ts, resp, context).hex(), "frame_timestamps": None, "burst_frames": burst}
    envelope = client.seal_submission(session, header, frame_bytes, kem)
    t0 = time.perf_counter()
    result = client.post_sealed(session, envelope)
    return (time.perf_counter() - t0) * 1000, result


def attacker(device_id: str, sig_alg: str, rate: float, barrier, out: mp.Queue, k: int) -> None:
    from crypto.signature import SignerIdentity, build_binding_payload

    client = _api(str(Path(tempfile.mkdtemp()) / "attacker.json"), sig_alg)
    h, frame_bytes, burst = _payload_parts(k)
    key = SignerIdentity(alg=sig_alg)
    sign = lambda h_, n, ts, r, ctx: key.sign(build_binding_payload(h_, n, ts, r, **ctx))  # noqa: E731
    interval = ATTACK_PROCS / rate if rate > 0 else 0.0
    n, accepted, error = 0, 0, None
    barrier.wait()
    t_start = time.time()
    stop_at = t_start + SECONDS
    next_at = t_start + (k / ATTACK_PROCS) * interval  # stagger the attackers
    while time.time() < stop_at:
        if interval:
            wait = next_at - time.time()
            if wait > 0:
                time.sleep(min(wait, stop_at - time.time()))
                if time.time() >= stop_at:
                    break
            next_at = max(next_at + interval, time.time() - interval)  # no burst catch-up after a stall
        try:
            # Raw session start: the attacker does not need to verify the offer.
            session = client._request("POST", "/session/start", json={"device_id": device_id})
            _, result = _submit(client, session, h, frame_bytes, burst, sign)
            n += 1
            accepted += bool(result["accepted"])
        except Exception as exc:
            error = error or f"{type(exc).__name__}: {str(exc)[:300]}"
    out.put(("attacker", n, accepted, time.time() - t_start, error))


def victim(key_path: str, sig_alg: str, barrier, out: mp.Queue) -> None:
    client = _api(key_path, sig_alg)
    dev = client.device()
    parts = [_payload_parts(i) for i in range(3)]
    lat, accepted, i = [], 0, 0
    sign = lambda h_, n, ts, r, ctx: dev.sign_binding(h_, n, ts, r, **ctx)  # noqa: E731
    from distributed_experiments import digit_session

    barrier.wait()
    stop_at = time.time() + SECONDS
    while time.time() < stop_at:
        h, frame_bytes, burst = parts[i % len(parts)]
        try:
            client.reset_replay_store()
            session = digit_session(client)
            ms, result = _submit(client, session, h, frame_bytes, burst, sign)
            client.end_session(session["session_id"])
            lat.append(ms)
            accepted += bool(result["accepted"])
        except Exception:
            pass
        i += 1
    out.put(("victim", lat, accepted))


def one_run(key_path: str, device_id: str, sig_alg: str, rate: float | None, server) -> dict:
    """rate None = victim alone."""
    ctx = mp.get_context("spawn")
    out = ctx.Queue()
    n_att = 0 if rate is None else ATTACK_PROCS
    barrier = ctx.Barrier(1 + n_att)
    procs = [ctx.Process(target=victim, args=(key_path, sig_alg, barrier, out))]
    procs += [ctx.Process(target=attacker, args=(device_id, sig_alg, rate, barrier, out, k)) for k in range(n_att)]
    cpu0 = server.cpu_seconds()
    for p in procs:
        p.start()
    results = [out.get() for _ in procs]
    for p in procs:
        p.join()
    cpu = server.cpu_seconds() - cpu0
    v = next(r for r in results if r[0] == "victim")
    atts = [r for r in results if r[0] == "attacker"]
    forged = sum(r[1] for r in atts)
    return {
        "target_rate": rate, "forged_requests": forged, "forged_accepted": sum(r[2] for r in atts),
        "forged_per_s": sum(r[1] / r[3] for r in atts if r[3] > 0),
        "victim_completed": len(v[1]), "victim_accepted": v[2],
        "victim_ms_median": statistics.median(v[1]) if v[1] else None,
        "server_cpu_s": cpu, "attacker_errors": sorted({r[4] for r in atts if r[4]}),
    }


def rate_at_degradation(points: list[tuple[float, float]], target: float) -> tuple[float | None, bool]:
    """points: (achieved forged rate, latency ratio), any order. Returns
    (interpolated rate, is_lower_bound)."""
    pts = sorted(p for p in points if p[0] > 0 and p[1] is not None)
    if not pts:
        return None, True
    if pts[0][1] >= target:
        return pts[0][0], False  # already degraded at the lowest rate (upper bound, rare)
    for (r0, y0), (r1, y1) in zip(pts, pts[1:]):
        if y0 < target <= y1:
            f = (target - y0) / (y1 - y0)
            return math.exp(math.log(r0) + f * (math.log(r1) - math.log(r0))), False
    return pts[-1][0], True


def run_config(mode: str, sig_alg: str) -> dict:
    from distributed_experiments import SERVER_KEY_PATH, Server

    server = Server(mode)
    key_path = str(Path(tempfile.mkdtemp()) / "victim.json")
    repeats = []
    try:
        client = _api(key_path, sig_alg)
        client._device = None
        device_id = client.enroll_device()
        for rep in range(REPEATS):
            base = one_run(key_path, device_id, sig_alg, None, server)
            sweep = [one_run(key_path, device_id, sig_alg, rate, server) for rate in RATES[mode]]
            for row in sweep:
                row["latency_ratio"] = row["victim_ms_median"] / base["victim_ms_median"] if row["victim_ms_median"] else None
            r2, lower = rate_at_degradation([(r["forged_per_s"], r["latency_ratio"]) for r in sweep], DEGRADATION)
            repeats.append({"repeat": rep, "baseline": base, "sweep": sweep,
                            "forged_per_s_at_2x": r2, "is_lower_bound": lower})
            print(json.dumps({"config": f"{mode}/{sig_alg}", "repeat": rep, "L0": base["victim_ms_median"],
                              "points": [(round(r["forged_per_s"], 2), round(r["latency_ratio"] or 0, 2)) for r in sweep],
                              "rate_at_2x": r2, "lower_bound": lower}), flush=True)
    finally:
        server.stop()
    vals = [r["forged_per_s_at_2x"] for r in repeats if r["forged_per_s_at_2x"] is not None]
    return {"mode": mode, "sig_alg": sig_alg, "seconds": SECONDS, "attack_procs": ATTACK_PROCS, "degradation": DEGRADATION,
            "server_key_path_shared": bool(SERVER_KEY_PATH), "repeats": repeats,
            "forged_per_s_at_2x_mean": statistics.fmean(vals) if vals else None,
            "forged_per_s_at_2x_sd": statistics.stdev(vals) if len(vals) > 1 else None,
            "any_lower_bound": any(r["is_lower_bound"] for r in repeats)}


def main() -> None:
    from distributed_experiments import CONFIGS, tag

    only = sys.argv[1:]
    _prepare_payloads(max(ATTACK_PROCS, 3))
    results = json.loads(OUT.read_text()) if OUT.exists() and only else {}
    for mode, sig_alg in CONFIGS:
        name = tag(mode, sig_alg)
        if only and name not in only:
            continue
        results[name] = run_config(mode, sig_alg)
        OUT.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
