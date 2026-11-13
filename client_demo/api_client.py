"""HTTP client for the FastAPI server (server/main.py).

Everything the client is trusted with happens in this process: it checks
the server's signature on the session offer against the pinned server key,
encapsulates to the session's ML-KEM public key, hashes the captured
evidence, signs the binding payload (which covers the ML-KEM ciphertext)
with the device's own key (client_demo/device.py), and seals header and
frames with AES-256-GCM (/verify-sealed). The server only ever sees the
device's public key. liboqs is therefore required on the client; there is
no mode in which the server signs evidence for it. Message formats:
docs/PROTOCOL.md.
"""
from __future__ import annotations

import base64
import hashlib
import os
import re
from datetime import datetime, timedelta, timezone

import numpy as np
import requests

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
_TIMEOUT_SECONDS = 30


class ApiError(RuntimeError):
    pass


def _request(method: str, path: str, **kwargs) -> dict:
    url = f"{API_BASE_URL}{path}"
    try:
        resp = requests.request(method, url, timeout=_TIMEOUT_SECONDS, **kwargs)
    except requests.exceptions.RequestException as exc:
        raise ApiError(f"Could not reach server at {url}: {exc}") from exc
    if resp.status_code >= 400:
        raise ApiError(f"{method} {path} -> HTTP {resp.status_code}: {resp.text}")
    return resp.json()


def health() -> dict:
    return _request("GET", "/health")


_device = None


def device():
    """This client's DeviceIdentity, created (and its key file written) on
    first use."""
    global _device
    if _device is None:
        from client_demo.device import DeviceIdentity

        _device = DeviceIdentity(alg=os.getenv("SIG_ALG", "ML-DSA-65"))
    return _device


def enroll_device() -> str:
    """Enroll this device's public key with the server once; returns device_id.
    Sends PQBIO_ENROLLMENT_TOKEN as X-Enrollment-Token if it is set."""
    dev = device()
    device_id = dev.device_id_for(API_BASE_URL)
    if device_id is None:
        token = os.getenv("PQBIO_ENROLLMENT_TOKEN")
        headers = {"X-Enrollment-Token": token} if token else {}
        device_id = _request("POST", "/device/enroll", json={"sig_alg": dev.alg, "public_key": dev.public_key.hex()},
                             headers=headers)["device_id"]
        dev.remember_enrollment(API_BASE_URL, device_id)
    return device_id


def server_key() -> tuple[str, bytes]:
    """The server identity key session offers must verify against: from
    PQBIO_SERVER_PUBKEY ("<alg>:<hex>", distributed out of band) if set,
    else the key pinned for this server, else - trust on first use - fetched
    from GET /server/identity once and pinned."""
    env = os.getenv("PQBIO_SERVER_PUBKEY")
    if env:
        alg, _, hex_key = env.partition(":")
        return alg, bytes.fromhex(hex_key)
    dev = device()
    pinned = dev.pinned_server_key(API_BASE_URL)
    if pinned is None:
        ident = _request("GET", "/server/identity")
        dev.pin_server_key(API_BASE_URL, ident["sig_alg"], bytes.fromhex(ident["public_key"]))
        pinned = dev.pinned_server_key(API_BASE_URL)
    return pinned["sig_alg"], bytes.fromhex(pinned["public_key"])


def verify_session_offer(session: dict) -> None:
    """Raise ApiError unless the server identity key signed this offer. Done
    before anything is encapsulated to session["kem_public_key"]."""
    from crypto.signature import build_session_offer, verify

    alg, public_key = server_key()
    offer = build_session_offer(
        session_id=session["session_id"], session_nonce=base64.b64decode(session["nonce"]), challenge=session["challenge"],
        kem_alg=session["kem_alg"], kem_public_key=bytes.fromhex(session["kem_public_key"]), device_id=session["device_id"],
        issued_at=session["issued_at"], freshness_window_seconds=session["freshness_window_seconds"],
    )
    try:
        ok = session.get("server_sig_alg") == alg and verify(offer, bytes.fromhex(session["server_signature"]), public_key, alg=alg)
    except (KeyError, ValueError):
        ok = False
    if not ok:
        raise ApiError("session offer is not signed by the pinned server key (possible man in the middle); refusing to send evidence")


def session_start() -> dict:
    try:
        session = _request("POST", "/session/start", json={"device_id": enroll_device()})
    except ApiError as exc:
        if "Unknown device_id" not in str(exc):
            raise
        # The server forgot us (in-memory registry restarted): enroll again.
        device().forget_enrollment(API_BASE_URL)
        session = _request("POST", "/session/start", json={"device_id": enroll_device()})
    verify_session_offer(session)
    return session


def end_session(session_id: str) -> dict:
    return _request("DELETE", f"/session/{session_id}")


def reset_replay_store() -> dict:
    """Demo rehearsal helper; the server only allows it with DEMO_MODE=true."""
    return _request("POST", "/demo/reset-replay-store")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def now_iso(offset_seconds: float = 0.0) -> str:
    return (datetime.now(timezone.utc) + timedelta(seconds=offset_seconds)).isoformat()


def can_seal() -> bool:
    """True if this client can run ML-KEM itself (liboqs importable), and so
    can send the verification payload over the sealed post-quantum channel."""
    try:
        import oqs  # noqa: F401
    except Exception:
        return False
    return True


def challenge_answer(challenge: str) -> str:
    """The response a user who saw `challenge` would give: the announced
    digits for a "say the number" prompt, otherwise a description of the
    movement (verified from the frames themselves, not from this text)."""
    match = re.search(r"\d+", challenge)
    if match and "number" in challenge.lower():
        return match.group()
    return "performed: " + challenge.lower()


# Ways the demo can deliberately break the transaction binding, to show that
# each one is caught cryptographically before any AI runs.
TAMPER_MODES = {
    "stale_timestamp": "Evidence signed 2 minutes ago (outside the 30 s freshness window)",
    "swap_frames": "Different frames submitted than the ones that were signed",
    "reuse_nonce": "Second submission on an already-used session nonce",
    "wrong_challenge": "Wrong answer to the liveness challenge (pre-recorded evidence)",
    "corrupt_channel": "One bit flipped in the ML-KEM/AES-GCM sealed payload in transit",
}


def run_verification(
    frames: list,
    frame_timestamps: list[float] | None = None,
    challenge_response: str | None = None,
    session: dict | None = None,
    tamper: str | None = None,
    sealed: bool | None = None,
    burst_frames: int = 0,
) -> tuple[dict, dict]:
    """Full client flow for one submission: start a session (unless one is
    given), hash the frames, sign the binding payload with this device's
    ML-DSA key, and submit over the sealed ML-KEM + AES-256-GCM channel
    (plaintext /verify only if `sealed=False`, e.g. for measurements).
    Returns (session, verify_response). `tamper` selects one of TAMPER_MODES."""
    from client_demo.frame_utils import frame_to_png_bytes
    from crypto.channel import evidence_hash_of

    session = session or session_start()
    session_id = session["session_id"]
    if challenge_response is None:
        challenge_response = challenge_answer(session["challenge"])
    if tamper == "wrong_challenge":
        challenge_response = "0000" if "number" in session["challenge"].lower() else ""

    sealed = can_seal() if sealed is None else sealed
    kem = encapsulate(session) if sealed else None
    frame_bytes = [frame_to_png_bytes(f) for f in frames]
    evidence_hash = evidence_hash_of(frame_bytes).hex()
    timestamp = now_iso(-120 if tamper == "stale_timestamp" else 0)
    signature = device().sign_binding(
        bytes.fromhex(evidence_hash), base64.b64decode(session["nonce"]), timestamp, challenge_response,
        **binding_context(session, kem[0] if kem else b"", burst_frames, frame_timestamps),
    ).hex()

    if tamper == "swap_frames":
        # e.g. a different face swapped in after a genuine one was signed
        frame_bytes = [frame_to_png_bytes(np.ascontiguousarray(f[:, ::-1])) for f in frames]
    header = {
        "evidence_hash": evidence_hash,
        "timestamp": timestamp,
        "challenge_response": challenge_response,
        "signature": signature,
        "frame_timestamps": frame_timestamps,
        "burst_frames": burst_frames,
    }

    if not sealed:
        if tamper == "reuse_nonce":
            _submit_plain(session, header, frame_bytes)  # first, legitimate use of the nonce
        return session, _submit_plain(session, header, frame_bytes)
    if tamper == "reuse_nonce":
        _submit_sealed(session, header, frame_bytes, kem)  # first, legitimate use of the nonce
    return session, _submit_sealed(session, header, frame_bytes, kem, corrupt=(tamper == "corrupt_channel"))


def binding_context(session: dict, kem_ciphertext: bytes, burst_frames: int = 0, frame_timestamps: list[float] | None = None) -> dict:
    """The signed fields besides hash, nonce, timestamp and challenge answer
    (crypto/signature.py: build_binding_payload)."""
    return {"session_id": session["session_id"], "kem_ciphertext": kem_ciphertext,
            "burst_frames": burst_frames, "frame_timestamps": frame_timestamps}


def encapsulate(session: dict) -> tuple[bytes, bytes]:
    """(ML-KEM ciphertext, shared secret) for this session's offered key.
    Callers verify the offer first (session_start does)."""
    from crypto.kem import client_encapsulate

    return client_encapsulate(bytes.fromhex(session["kem_public_key"]), alg=session["kem_alg"])


def _submit_plain(session: dict, header: dict, frame_bytes: list[bytes]) -> dict:
    payload = {**header, "frames_b64": [base64.b64encode(b).decode("ascii") for b in frame_bytes]}
    return _request("POST", f"/session/{session['session_id']}/verify", json=payload)


def seal_submission(session: dict, header: dict, frame_bytes: list[bytes], kem: tuple[bytes, bytes]) -> bytes:
    """The PQB1 envelope (crypto/channel.py): ML-KEM ciphertext, sealed
    header, sealed frames - header first so the server can reject a bad
    signature before it reads the frames. `kem` is the (ciphertext, shared
    secret) from encapsulate(), the same ciphertext the header's signature
    covers."""
    from crypto.channel import seal_envelope

    kem_ciphertext, shared_secret = kem
    return seal_envelope(shared_secret, base64.b64decode(session["nonce"]), session["session_id"], kem_ciphertext, header, frame_bytes)


def post_sealed(session: dict, envelope: bytes) -> dict:
    return _request(
        "POST", f"/session/{session['session_id']}/verify-sealed",
        data=envelope, headers={"Content-Type": "application/octet-stream"},
    )


def _submit_sealed(session: dict, header: dict, frame_bytes: list[bytes], kem: tuple[bytes, bytes], corrupt: bool = False) -> dict:
    envelope = bytearray(seal_submission(session, header, frame_bytes, kem))
    if corrupt:
        envelope[-1] ^= 0x01  # one bit of the sealed frames
    result = post_sealed(session, bytes(envelope))
    result["sealed_bytes"] = len(envelope)
    return result
