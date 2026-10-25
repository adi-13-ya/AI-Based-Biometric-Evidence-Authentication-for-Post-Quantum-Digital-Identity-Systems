"""D3 (revised): forged-request flood against a separate server process.

Each attacker is its own OS process (no shared GIL with the victim or other
attackers). An attacker prepares one realistic payload once (16 PNG frames
from a real webcam clip, ~7 MB), then loops: open a session for the victim's
enrolled device (device ids are not secret), sign with the attacker's own
ML-DSA key, encapsulate to the session's ML-KEM key, seal, submit. Anyone can
do the last three steps; only the signature check (g3) can tell this
request is not from the victim's device.

The victim process runs real verifications of webcam clips (spoken-digit
sessions, signed with the enrolled device key) for the whole run and records
each /verify-sealed latency. Every mode is run twice: victim alone, then
victim plus FLOOD_PROCS attackers.

Run: python report/scripts/flood_experiment.py
Writes report/data/v2/d3_flood.json
"""
from __future__ import annotations

import base64
import json
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
SECONDS = 20
FLOOD_PROCS = 4


def _api(key_path: str):
    os.environ["PQBIO_DEVICE_KEY"] = key_path
    import client_demo.api_client as client

    client.API_BASE_URL = f"http://127.0.0.1:{PORT}"
    return client


def _payload_parts(clip_index: int):
    from client_demo.frame_utils import frame_to_png_bytes
    from crypto.channel import evidence_hash_of
    from distributed_experiments import webcam_clips

    frames, burst = webcam_clips(clip_index + 1)[clip_index]
    frame_bytes = [frame_to_png_bytes(f) for f in frames]
    return evidence_hash_of(frame_bytes), frame_bytes, burst


def _submit(client, session, evidence_hash, frame_bytes, burst, sign) -> tuple[float, dict]:
    nonce = base64.b64decode(session["nonce"])
    ts = datetime.now(timezone.utc).isoformat()
    resp = client.challenge_answer(session["challenge"])
    header = {"evidence_hash": evidence_hash.hex(), "timestamp": ts, "challenge_response": resp,
              "signature": sign(evidence_hash, nonce, ts, resp).hex(), "frame_timestamps": None, "burst_frames": burst}
    envelope = client.seal_submission(session, header, frame_bytes)
    t0 = time.perf_counter()
    result = client.post_sealed(session, envelope)
    return (time.perf_counter() - t0) * 1000, result


def attacker(device_id: str, barrier, out: mp.Queue, k: int) -> None:
    from crypto.signature import SignerIdentity, build_binding_payload

    client = _api(str(Path(tempfile.mkdtemp()) / "attacker.json"))
    h, frame_bytes, burst = _payload_parts(k)
    key = SignerIdentity(alg="ML-DSA-65")
    sign = lambda h_, n, ts, r: key.sign(build_binding_payload(h_, n, ts, r))  # noqa: E731
    n, accepted, error = 0, 0, None
    barrier.wait()  # everyone starts the measured window together, after setup
    t_start = time.time()
    stop_at = t_start + SECONDS
    while time.time() < stop_at:
        try:
            session = client._request("POST", "/session/start", json={"device_id": device_id})
            _, result = _submit(client, session, h, frame_bytes, burst, sign)
            n += 1
            accepted += bool(result["accepted"])
        except Exception as exc:
            error = error or f"{type(exc).__name__}: {str(exc)[:300]}"
    out.put(("attacker", n, accepted, time.time() - t_start, error))


def victim(key_path: str, barrier, out: mp.Queue) -> None:
    client = _api(key_path)
    dev = client.device()
    parts = [_payload_parts(i) for i in range(3)]
    lat, accepted, i = [], 0, 0
    sign = lambda h_, n, ts, r: dev.sign_binding(h_, n, ts, r)  # noqa: E731
    from distributed_experiments import digit_session

    barrier.wait()
    t_start = time.time()
    stop_at = t_start + SECONDS

    while time.time() < stop_at:
        h, frame_bytes, burst = parts[i % len(parts)]
        try:
            client.reset_replay_store()
            ms, result = _submit(client, digit_session(client), h, frame_bytes, burst, sign)
            lat.append(ms)
            accepted += bool(result["accepted"])
        except Exception:
            pass
        i += 1
    out.put(("victim", lat, accepted, time.time() - t_start))


def run(mode: str, with_flood: bool) -> dict:
    from distributed_experiments import Server

    server = Server(mode)
    key_path = str(Path(tempfile.mkdtemp()) / "victim.json")
    try:
        client = _api(key_path)
        client._device = None  # fresh identity per run: each run starts a new server with an empty registry
        device_id = client.enroll_device()
        ctx = mp.get_context("spawn")
        out = ctx.Queue()
        n_procs = 1 + (FLOOD_PROCS if with_flood else 0)
        barrier = ctx.Barrier(n_procs)
        procs = [ctx.Process(target=victim, args=(key_path, barrier, out))]
        if with_flood:
            procs += [ctx.Process(target=attacker, args=(device_id, barrier, out, k)) for k in range(FLOOD_PROCS)]
        cpu0 = server.cpu_seconds()
        for p in procs:
            p.start()
        results = [out.get() for _ in procs]
        for p in procs:
            p.join()
        cpu = server.cpu_seconds() - cpu0
    finally:
        server.stop()
    lat = next(r[1] for r in results if r[0] == "victim")
    v_acc = next(r[2] for r in results if r[0] == "victim")
    forged = sum(r[1] for r in results if r[0] == "attacker")
    forged_acc = sum(r[2] for r in results if r[0] == "attacker")
    return {
        "mode": mode, "flood": with_flood, "seconds": SECONDS, "attackers": FLOOD_PROCS if with_flood else 0,
        "forged_requests": forged, "forged_accepted": forged_acc,
        # summed per-attacker rates over each attacker's own active time
        "forged_per_s": sum(r[1] / r[3] for r in results if r[0] == "attacker" and r[3] > 0),
        "victim_active_s": next(r[3] for r in results if r[0] == "victim"),
        "victim_completed": len(lat), "victim_accepted": v_acc,
        "victim_ms_median": statistics.median(lat) if lat else None,
        "victim_ms_p95": sorted(lat)[int(0.95 * (len(lat) - 1))] if lat else None,
        "server_cpu_s": cpu,
        "attacker_errors": sorted({r[4] for r in results if r[0] == "attacker" and r[4]}),
    }


def main() -> None:
    rows = []
    modes = sys.argv[1:] or ["integrated", "layered"]
    floods = [True] if "--flood-only" in modes else [False, True]
    modes = [m for m in modes if not m.startswith("--")] or ["integrated", "layered"]
    for mode in modes:
        for flood in floods:
            r = run(mode, flood)
            print(json.dumps(r))
            rows.append(r)
    out = REPO / "report" / "data" / "v2" / "d3_flood.json"
    out.write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
