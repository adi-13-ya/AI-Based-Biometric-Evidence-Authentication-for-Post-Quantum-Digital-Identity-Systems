"""FastAPI server tests: device enrollment, session lifecycle, crypto,
Section 7 transaction binding, and the full AI-detection + fusion pipeline
wired into /verify.

The tests play the capture device: they hold their own ML-DSA key
(`_DEVICE`), enroll its public key and sign every binding payload locally,
exactly like client_demo/device.py. The server is never asked to sign.

`/health` is designed to degrade gracefully (not raise) when liboqs is
unavailable, so it's tested unconditionally. Everything that creates a
session needs a real ML-KEM/ML-DSA keypair and is skipped without liboqs -
see tests/conftest.py. Fusion's exact ACCEPT/REJECT outcome on mock-mode
heuristic scores of synthetic noise isn't asserted here (that's
deterministic-input territory - see tests/test_fusion.py); these tests only
assert that the pipeline *ran* and returned well-formed output once binding
passes.
"""
from __future__ import annotations

import base64
import re
from datetime import datetime, timedelta, timezone
from io import BytesIO

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from conftest import skip_if_no_oqs

from crypto.channel import evidence_hash_of
from server.main import app

client = TestClient(app)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _frame_b64(seed: int = 0) -> str:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    buf = BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _hash_of(*frames_b64: str) -> str:
    return evidence_hash_of([base64.b64decode(f) for f in frames_b64]).hex()


def _challenge_response_for(challenge: str) -> str:
    """Whatever `verify_challenge` (detection/liveness_challenge.py) needs
    to accept this specific challenge without real motion frames: the exact
    announced digits for a "say the number" challenge, or any non-empty
    string for a turn/nod challenge (which falls back to a response-present
    check without motion frames)."""
    match = re.search(r"\d+", challenge)
    if match and "number" in challenge.lower():
        return match.group()
    return "done"


_DEVICE = None


def _device():
    global _DEVICE
    if _DEVICE is None:
        from crypto.signature import SignerIdentity

        _DEVICE = SignerIdentity(alg="ML-DSA-65")
    return _DEVICE


def _start() -> dict:
    """Enroll the test device (idempotent) and open a session for it."""
    dev = _device()
    device_id = client.post("/device/enroll", json={"sig_alg": dev.alg, "public_key": dev.public_key.hex()}).json()["device_id"]
    resp = client.post("/session/start", json={"device_id": device_id})
    assert resp.status_code == 200, resp.text
    return resp.json()


def _sign(session: dict, evidence_hash_hex: str, timestamp: str, challenge_response: str, signer=None) -> str:
    from crypto.signature import build_binding_payload

    payload = build_binding_payload(bytes.fromhex(evidence_hash_hex), base64.b64decode(session["nonce"]), timestamp, challenge_response)
    return (signer or _device()).sign(payload).hex()


def test_health_reports_status_without_raising():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] in {"ok", "degraded"}
    assert "kem_alg" in body and "sig_alg" in body
    assert "liboqs_ready" in body


def test_demo_reset_is_refused_outside_demo_mode(monkeypatch):
    from server.config import settings

    monkeypatch.setattr(settings, "demo_mode", False)
    assert client.post("/demo/reset-replay-store").status_code == 403
    monkeypatch.setattr(settings, "demo_mode", True)
    assert client.post("/demo/reset-replay-store").json() == {"replay_store_cleared": True}


@skip_if_no_oqs
class TestSessionFlow:
    def test_session_start_returns_expected_fields(self):
        body = _start()
        for field in ("session_id", "nonce", "challenge", "kem_public_key", "device_id"):
            assert body[field]
        assert "client_public_key" not in body

    def test_session_start_requires_an_enrolled_device(self):
        assert client.post("/session/start", json={"device_id": "0" * 32}).status_code == 404

    def test_enroll_rejects_malformed_key(self):
        assert client.post("/device/enroll", json={"sig_alg": "ML-DSA-65", "public_key": "00" * 10}).status_code == 400

    def test_server_exposes_no_signing_endpoint(self):
        """Distributed trust: the server must not be able to sign for a client."""
        session = _start()
        sid = session["session_id"]
        assert client.post(f"/session/{sid}/evidence", json={}).status_code in (404, 405)
        assert client.post(f"/session/{sid}/channel").status_code in (404, 405)

    def test_signature_by_another_device_is_rejected(self):
        from crypto.signature import SignerIdentity

        session = _start()
        frame = _frame_b64(seed=9)
        timestamp = _now_iso()
        resp = _challenge_response_for(session["challenge"])
        with SignerIdentity(alg="ML-DSA-65") as other:
            signature = _sign(session, _hash_of(frame), timestamp, resp, signer=other)
        body = client.post(
            f"/session/{session['session_id']}/verify",
            json={"evidence_hash": _hash_of(frame), "timestamp": timestamp, "challenge_response": resp, "signature": signature, "frames_b64": [frame]},
        ).json()
        assert body["binding_checks"]["signature_valid"] is False
        assert body["detection_ran"] is False

    def test_sign_then_verify_runs_full_pipeline(self):
        session = _start()
        session_id = session["session_id"]
        challenge_response = _challenge_response_for(session["challenge"])

        frame = _frame_b64(seed=1)
        evidence_hash = _hash_of(frame)
        timestamp = _now_iso()
        signature = _sign(session, evidence_hash, timestamp, challenge_response)

        verify_resp = client.post(
            f"/session/{session_id}/verify",
            json={
                "evidence_hash": evidence_hash,
                "timestamp": timestamp,
                "challenge_response": challenge_response,
                "signature": signature,
                "frames_b64": [frame],
            },
        )
        assert verify_resp.status_code == 200
        body = verify_resp.json()
        assert body["binding_accepted"] is True
        assert body["binding_checks"] == {
            "signature_valid": True,
            "nonce_unused": True,
            "timestamp_fresh": True,
            "challenge_fulfilled": True,
            "evidence_matches_hash": True,
        }
        assert body["channel"] == "plaintext"
        assert body["detection_ran"] is True
        assert set(body["detection_scores"].keys()) == {
            "presentation_attack",
            "morph",
            "deepfake",
            "replay",
            "injection",
        }
        assert 0.0 <= body["fused_score"] <= 1.0
        assert isinstance(body["accepted"], bool)
        assert "total_ms" in body["timing_ms"]

    def test_verify_without_frames_after_binding_passes_is_400(self):
        session = _start()
        session_id = session["session_id"]
        challenge_response = _challenge_response_for(session["challenge"])
        timestamp = _now_iso()

        signature = _sign(session, "00" * 32, timestamp, challenge_response)

        verify_resp = client.post(
            f"/session/{session_id}/verify",
            json={
                "evidence_hash": "00" * 32,
                "timestamp": timestamp,
                "challenge_response": challenge_response,
                "signature": signature,
                # frames_b64 omitted
            },
        )
        assert verify_resp.status_code == 400

    def test_verify_rejects_mismatched_evidence_hash_before_detection(self):
        session = _start()
        session_id = session["session_id"]
        timestamp = _now_iso()

        signature = _sign(session, "00" * 32, timestamp, "done")

        verify_resp = client.post(
            f"/session/{session_id}/verify",
            json={
                "evidence_hash": "11" * 32,  # different evidence than what was signed
                "timestamp": timestamp,
                "challenge_response": "done",
                "signature": signature,
            },
        )
        assert verify_resp.status_code == 200
        body = verify_resp.json()
        assert body["binding_accepted"] is False
        assert body["binding_checks"]["signature_valid"] is False
        assert body["detection_ran"] is False
        assert body["accepted"] is False

    def test_verify_rejects_stale_timestamp_before_detection(self):
        session = _start()
        session_id = session["session_id"]
        stale_timestamp = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()

        signature = _sign(session, "00" * 32, stale_timestamp, "done")

        verify_resp = client.post(
            f"/session/{session_id}/verify",
            json={
                "evidence_hash": "00" * 32,
                "timestamp": stale_timestamp,
                "challenge_response": "done",
                "signature": signature,
            },
        )
        body = verify_resp.json()
        assert body["binding_accepted"] is False
        assert body["binding_checks"]["timestamp_fresh"] is False
        # short-circuit: the signature is never even verified for a stale submission
        assert "signature_valid" not in body["binding_checks"]
        assert body["detection_ran"] is False

    def test_verify_rejects_reused_nonce(self):
        session = _start()
        session_id = session["session_id"]
        challenge_response = _challenge_response_for(session["challenge"])
        timestamp = _now_iso()
        frame = _frame_b64(seed=2)

        signature = _sign(session, _hash_of(frame), timestamp, challenge_response)
        verify_payload = {
            "evidence_hash": _hash_of(frame),
            "timestamp": timestamp,
            "challenge_response": challenge_response,
            "signature": signature,
            "frames_b64": [frame],
        }

        first = client.post(f"/session/{session_id}/verify", json=verify_payload)
        assert first.json()["binding_accepted"] is True

        # Replaying the exact same (already-signed, already-verified)
        # evidence against the same session must be rejected - this is the
        # core anti-replay property from spec Section 7.
        second = client.post(f"/session/{session_id}/verify", json=verify_payload)
        body = second.json()
        assert body["binding_accepted"] is False
        assert body["binding_checks"]["nonce_unused"] is False
        assert body["detection_ran"] is False

    def test_verify_rejects_frames_substituted_after_signing(self):
        session = _start()
        session_id = session["session_id"]
        challenge_response = _challenge_response_for(session["challenge"])
        timestamp = _now_iso()
        signed_frame, swapped_frame = _frame_b64(seed=3), _frame_b64(seed=4)

        signature = _sign(session, _hash_of(signed_frame), timestamp, challenge_response)

        body = client.post(
            f"/session/{session_id}/verify",
            json={
                "evidence_hash": _hash_of(signed_frame),  # valid signature over the genuine frame...
                "timestamp": timestamp,
                "challenge_response": challenge_response,
                "signature": signature,
                "frames_b64": [swapped_frame],  # ...but different frames submitted
            },
        ).json()
        assert body["binding_checks"]["signature_valid"] is True
        assert body["binding_checks"]["evidence_matches_hash"] is False
        assert body["detection_ran"] is False
        assert body["accepted"] is False

    def test_verify_unknown_session_returns_404(self):
        resp = client.post(
            "/session/does-not-exist/verify",
            json={"evidence_hash": "00" * 32, "timestamp": _now_iso(), "signature": "00"},
        )
        assert resp.status_code == 404

    def test_end_session_closes_it(self):
        session = _start()
        session_id = session["session_id"]
        resp = client.delete(f"/session/{session_id}")
        assert resp.status_code == 200
        assert resp.json()["closed"] is True
        # session no longer usable
        assert client.post(
            f"/session/{session_id}/verify", json={"evidence_hash": "00" * 32, "timestamp": _now_iso(), "signature": "00"}
        ).status_code == 404


@skip_if_no_oqs
class TestSealedChannel:
    def _envelope(self, session: dict, aad_session_id: str | None = None, signer=None) -> bytearray:
        from crypto.channel import seal_envelope
        from crypto.kem import client_encapsulate

        challenge_response = _challenge_response_for(session["challenge"])
        timestamp = _now_iso()
        frame = _frame_b64(seed=5)
        header = {
            "evidence_hash": _hash_of(frame),
            "timestamp": timestamp,
            "challenge_response": challenge_response,
            "signature": _sign(session, _hash_of(frame), timestamp, challenge_response, signer=signer),
        }
        kem_ct, shared_secret = client_encapsulate(bytes.fromhex(session["kem_public_key"]), alg=session["kem_alg"])
        return bytearray(seal_envelope(shared_secret, base64.b64decode(session["nonce"]), aad_session_id or session["session_id"],
                                       kem_ct, header, [base64.b64decode(frame)]))

    def _post(self, session: dict, envelope) -> dict:
        resp = client.post(f"/session/{session['session_id']}/verify-sealed", content=bytes(envelope),
                           headers={"Content-Type": "application/octet-stream"})
        assert resp.status_code == 200, resp.text
        return resp.json()

    def test_sealed_payload_is_decrypted_and_verified(self):
        session = _start()
        body = self._post(session, self._envelope(session))
        assert body["channel"] == "ML-KEM-768 + AES-256-GCM"
        assert body["binding_accepted"] is True
        assert body["detection_ran"] is True
        assert {"kem_decapsulate_ms", "aead_decrypt_ms"} <= set(body["timing_ms"])

    def test_tampered_frames_are_rejected_at_channel(self):
        session = _start()
        envelope = self._envelope(session)
        envelope[-1] ^= 0x01
        body = self._post(session, envelope)
        assert body["binding_checks"] == {"channel_authentic": False}
        assert body["accepted"] is False

    def test_payload_sealed_for_another_session_is_rejected(self):
        session = _start()
        body = self._post(session, self._envelope(session, aad_session_id="some-other-session"))
        assert body["binding_checks"] == {"channel_authentic": False}

    def test_forged_signature_is_rejected_before_frames_are_opened(self):
        """Header-first early rejection: with a forged signature the server
        must stop at the header. The frames are deliberately corrupted too;
        if the server had decrypted them it would report a channel failure."""
        from crypto.signature import SignerIdentity

        session = _start()
        with SignerIdentity(alg="ML-DSA-65") as attacker:
            envelope = self._envelope(session, signer=attacker)
        envelope[-1] ^= 0x01
        body = self._post(session, envelope)
        assert body["binding_checks"] == {"nonce_unused": True, "timestamp_fresh": True, "signature_valid": False}
        assert body["detection_ran"] is False

    def test_malformed_envelope_is_400(self):
        session = _start()
        resp = client.post(f"/session/{session['session_id']}/verify-sealed", content=b"not an envelope",
                           headers={"Content-Type": "application/octet-stream"})
        assert resp.status_code == 400


@skip_if_no_oqs
def test_layered_ablation_pays_for_detection_on_forged_requests(monkeypatch):
    """PIPELINE_MODE=layered runs the AI stage before the crypto checks; the
    integrated default never runs it for a forged submission."""
    from crypto.signature import SignerIdentity
    from server.config import settings

    def forged(session):
        frame = _frame_b64(seed=21)
        timestamp = _now_iso()
        resp = _challenge_response_for(session["challenge"])
        with SignerIdentity(alg="ML-DSA-65") as attacker:
            signature = _sign(session, _hash_of(frame), timestamp, resp, signer=attacker)
        return client.post(
            f"/session/{session['session_id']}/verify",
            json={"evidence_hash": _hash_of(frame), "timestamp": timestamp, "challenge_response": resp, "signature": signature, "frames_b64": [frame]},
        ).json()

    monkeypatch.setattr(settings, "pipeline_mode", "integrated")
    body = forged(_start())
    assert body["accepted"] is False and "detection_ms" not in body["timing_ms"]

    monkeypatch.setattr(settings, "pipeline_mode", "layered")
    body = forged(_start())
    assert body["accepted"] is False and body["timing_ms"]["detection_ms"] > 0
