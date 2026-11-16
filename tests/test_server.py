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
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from conftest import skip_if_no_oqs

from crypto.channel import evidence_hash_of
from server.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def _plaintext_endpoint_for_tests(monkeypatch):
    """The plaintext /verify endpoint is off by default (the evaluated
    format is /verify-sealed); these tests drive it directly. Most tests
    exercise the protocol with one small synthetic frame, so they allow
    sample-style submissions and lift the per-device session limits; the
    tests of those mechanisms (TestAuthenticatedClientHardening,
    TestInvalidRequestsCannotKillSessions) set them back."""
    from server.config import settings

    monkeypatch.setattr(settings, "allow_plaintext_verify", True)
    monkeypatch.setattr(settings, "allow_sample_uploads", True)
    monkeypatch.setattr(settings, "max_active_sessions_per_device", 10**6)
    monkeypatch.setattr(settings, "max_sessions_per_minute_per_device", 10**6)


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


def _start_request(device_id: str, signer, timestamp: str | None = None, request_nonce: bytes | None = None) -> dict:
    """Signed session-start body (crypto/signature.py: build_session_request)."""
    import os

    from crypto.signature import build_session_request

    timestamp = timestamp or _now_iso()
    request_nonce = request_nonce or os.urandom(16)
    sig = signer.sign(build_session_request(device_id=device_id, timestamp=timestamp, request_nonce=request_nonce))
    return {"device_id": device_id, "timestamp": timestamp, "request_nonce": request_nonce.hex(), "signature": sig.hex()}


def _enroll(dev=None) -> str:
    dev = dev or _device()
    return client.post("/device/enroll", json={"sig_alg": dev.alg, "public_key": dev.public_key.hex()}).json()["device_id"]


def _start(dev=None) -> dict:
    """Enroll the test device (idempotent) and open a signed session for it."""
    dev = dev or _device()
    resp = client.post("/session/start", json=_start_request(_enroll(dev), dev))
    assert resp.status_code == 200, resp.text
    return resp.json()


def _close_headers(session: dict, dev=None) -> dict:
    from crypto.signature import build_session_close

    sig = (dev or _device()).sign(build_session_close(session_id=session["session_id"], session_nonce=base64.b64decode(session["nonce"])))
    return {"X-Session-Close-Signature": sig.hex()}


def _sign(session: dict, evidence_hash_hex: str, timestamp: str, challenge_response: str, signer=None,
          kem_ct: bytes = b"", burst_frames: int = 0, frame_timestamps=None) -> str:
    from crypto.signature import build_binding_payload

    payload = build_binding_payload(bytes.fromhex(evidence_hash_hex), base64.b64decode(session["nonce"]), timestamp, challenge_response,
                                    session_id=session["session_id"], kem_ciphertext=kem_ct,
                                    burst_frames=burst_frames, frame_timestamps=frame_timestamps)
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
        assert client.post("/session/start", json=_start_request("0" * 32, _device())).status_code == 404

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
            "submission_well_formed": True,
            "nonce_consumed": True,
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
        assert client.delete(f"/session/{session_id}").status_code == 401  # unsigned close is refused
        resp = client.delete(f"/session/{session_id}", headers=_close_headers(session))
        assert resp.status_code == 200
        assert resp.json()["closed"] is True
        # session no longer usable
        assert client.post(
            f"/session/{session_id}/verify", json={"evidence_hash": "00" * 32, "timestamp": _now_iso(), "signature": "00"}
        ).status_code == 404


@skip_if_no_oqs
class TestSealedChannel:
    def _envelope(self, session: dict, aad_session_id: str | None = None, signer=None, rewrap: bool = False,
                  header_edit: dict | None = None) -> bytearray:
        """A sealed submission. `rewrap`: sign over one encapsulation but seal
        under another (what an interceptor re-wrapping the header would do).
        `header_edit`: fields changed after signing."""
        from crypto.channel import seal_envelope
        from crypto.kem import client_encapsulate

        challenge_response = _challenge_response_for(session["challenge"])
        timestamp = _now_iso()
        frame = _frame_b64(seed=5)
        kem_ct, shared_secret = client_encapsulate(bytes.fromhex(session["kem_public_key"]), alg=session["kem_alg"])
        header = {
            "evidence_hash": _hash_of(frame),
            "timestamp": timestamp,
            "challenge_response": challenge_response,
            "signature": _sign(session, _hash_of(frame), timestamp, challenge_response, signer=signer, kem_ct=kem_ct),
            "burst_frames": 0,
            "frame_timestamps": None,
            **(header_edit or {}),
        }
        if rewrap:
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

    def test_unsigned_header_edit_is_rejected(self):
        """burst_frames and frame_timestamps steer the injection check, so they
        are signed: changing them after signing breaks the signature."""
        session = _start()
        body = self._post(session, self._envelope(session, header_edit={"burst_frames": 1}))
        assert body["binding_checks"]["signature_valid"] is False

    def test_header_rewrapped_under_another_encapsulation_is_rejected(self):
        session = _start()
        body = self._post(session, self._envelope(session, rewrap=True))
        assert body["binding_checks"]["signature_valid"] is False
        assert body["detection_ran"] is False

    def test_buffered_baseline_reads_the_frames_before_checking(self, monkeypatch):
        """PIPELINE_MODE=buffered: same checks, but the whole body is received
        and decrypted first. Corrupted frames therefore surface as a channel
        failure even though the signature is forged; the integrated gate
        (test above) stops at the signature without opening them."""
        from crypto.signature import SignerIdentity
        from server.config import settings

        monkeypatch.setattr(settings, "pipeline_mode", "buffered")
        session = _start()
        with SignerIdentity(alg="ML-DSA-65") as attacker:
            envelope = self._envelope(session, signer=attacker)
        envelope[-1] ^= 0x01
        assert self._post(session, envelope)["binding_checks"] == {"channel_authentic": False}

        session = _start()
        with SignerIdentity(alg="ML-DSA-65") as attacker:
            body = self._post(session, self._envelope(session, signer=attacker))
        assert body["binding_checks"]["signature_valid"] is False and body["detection_ran"] is False

        session = _start()
        body = self._post(session, self._envelope(session))
        assert body["binding_accepted"] is True and body["detection_ran"] is True

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


@skip_if_no_oqs
class TestServerAuthenticationAndEnrollment:
    def test_session_offer_is_signed_by_the_server_identity(self):
        from crypto.signature import build_session_offer, verify

        ident = client.get("/server/identity").json()
        session = _start()
        offer = build_session_offer(
            session_id=session["session_id"], session_nonce=base64.b64decode(session["nonce"]), challenge=session["challenge"],
            kem_alg=session["kem_alg"], kem_public_key=bytes.fromhex(session["kem_public_key"]), device_id=session["device_id"],
            issued_at=session["issued_at"], freshness_window_seconds=session["freshness_window_seconds"],
        )
        assert session["server_key_id"] == ident["key_id"]
        assert verify(offer, bytes.fromhex(session["server_signature"]), bytes.fromhex(ident["public_key"]), alg=ident["sig_alg"])

    def test_client_refuses_a_substituted_kem_key(self, monkeypatch, tmp_path):
        """A man in the middle who swaps the ephemeral ML-KEM key cannot re-sign
        the offer, so the client stops before encapsulating to it."""
        import client_demo.api_client as api
        from crypto.kem import ServerKemSession

        ident = client.get("/server/identity").json()
        monkeypatch.setenv("PQBIO_SERVER_PUBKEY", f"{ident['sig_alg']}:{ident['public_key']}")
        session = _start()
        api.verify_session_offer(session)  # genuine offer passes
        with ServerKemSession(alg=session["kem_alg"]) as mitm:
            mitm.generate_keypair()
            forged = {**session, "kem_public_key": mitm.public_key.hex()}
        with pytest.raises(api.ApiError):
            api.verify_session_offer(forged)

    def test_enrollment_token_gates_enrollment(self, monkeypatch):
        from crypto.signature import SignerIdentity
        from server.config import settings

        monkeypatch.setattr(settings, "enrollment_token", "onboarding-secret")
        with SignerIdentity(alg="ML-DSA-65") as dev:
            body = {"sig_alg": dev.alg, "public_key": dev.public_key.hex()}
            assert client.post("/device/enroll", json=body).status_code == 403
            assert client.post("/device/enroll", json=body, headers={"X-Enrollment-Token": "wrong"}).status_code == 403
            assert client.post("/device/enroll", json=body, headers={"X-Enrollment-Token": "onboarding-secret"}).status_code == 200

    def test_plaintext_verify_is_off_by_default(self, monkeypatch):
        from server.config import settings

        monkeypatch.setattr(settings, "allow_plaintext_verify", False)
        session = _start()
        body = {"evidence_hash": "00" * 32, "timestamp": _now_iso(), "signature": "00"}
        assert client.post(f"/session/{session['session_id']}/verify", json=body).status_code == 404

    def test_classical_device_runs_the_same_protocol(self):
        """ECDSA-P256 devices exist for the like-for-like classical baseline."""
        from crypto.signature import SignerIdentity

        with SignerIdentity(alg="ECDSA-P256") as dev:
            session = _start(dev)
            assert session["sig_alg"] == "ECDSA-P256"
            frame = _frame_b64(seed=31)
            ts = _now_iso()
            resp = _challenge_response_for(session["challenge"])
            body = client.post(f"/session/{session['session_id']}/verify", json={
                "evidence_hash": _hash_of(frame), "timestamp": ts, "challenge_response": resp,
                "signature": _sign(session, _hash_of(frame), ts, resp, signer=dev), "frames_b64": [frame],
            }).json()
        assert body["binding_accepted"] is True


@skip_if_no_oqs
def test_concurrent_duplicates_consume_the_nonce_once():
    """Of many simultaneous copies of one submission, exactly one gets past
    the nonce check."""
    import threading

    session = _start()
    from server.session_store import store

    sess = store.get(session["session_id"])
    barrier = threading.Barrier(16)
    wins = []

    def race():
        barrier.wait()
        wins.append(sess.consume_nonce())

    threads = [threading.Thread(target=race) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert wins.count(True) == 1
