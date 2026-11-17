"""Session safety and authenticated-client hardening (v1.3).

1. Invalid requests cannot invalidate a legitimate session. The nonce is
   consumed only after the device signature and the evidence hash verify,
   session close is signed, and session start is signed. Every test sends
   invalid traffic at a live session and then shows that the device's own
   submission still passes the binding gate.
2. An authenticated (enrolled, validly signing) client cannot change the
   shape of a submission to skip checks or to cost the server unbounded work,
   and the liveness check fails closed.
"""
from __future__ import annotations

import base64
import io
import os
import threading
from datetime import timedelta, datetime, timezone

import numpy as np
import pytest
from PIL import Image

from conftest import skip_if_no_oqs
from test_server import _close_headers, _device, _enroll, _hash_of, _now_iso, _sign, _start, _start_request, client

pytestmark = skip_if_no_oqs


@pytest.fixture(autouse=True)
def _defaults(monkeypatch):
    """Production defaults (strict layout, signed session start, motion
    challenges), with the per-minute limit lifted unless a test sets it."""
    from server.config import settings

    monkeypatch.setattr(settings, "allow_sample_uploads", False)
    monkeypatch.setattr(settings, "require_signed_session_start", True)
    monkeypatch.setattr(settings, "challenge_kinds", "motion")
    monkeypatch.setattr(settings, "max_sessions_per_minute_per_device", 10**6)
    monkeypatch.setattr(settings, "max_active_sessions_per_device", 10**6)


def _png(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def _frames(n: int = 16, size: int = 64, seed: int = 0) -> list[bytes]:
    rng = np.random.default_rng(seed)
    return [_png(rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)) for _ in range(n)]


def _envelope(session: dict, frames: list[bytes], burst: int = 8, signer=None, timestamp: str | None = None,
              swap_after_signing: list[bytes] | None = None, aad_session_id: str | None = None) -> bytes:
    from crypto.channel import evidence_hash_of, seal_envelope
    from crypto.kem import client_encapsulate

    ts = timestamp or _now_iso()
    kem_ct, shared = client_encapsulate(bytes.fromhex(session["kem_public_key"]), alg=session["kem_alg"])
    h = evidence_hash_of(frames).hex()
    header = {"evidence_hash": h, "timestamp": ts, "challenge_response": "done", "burst_frames": burst, "frame_timestamps": None,
              "signature": _sign(session, h, ts, "done", signer=signer, kem_ct=kem_ct, burst_frames=burst)}
    return seal_envelope(shared, base64.b64decode(session["nonce"]), aad_session_id or session["session_id"], kem_ct, header,
                         swap_after_signing or frames)


def _post(session: dict, body: bytes):
    return client.post(f"/session/{session['session_id']}/verify-sealed", content=body,
                       headers={"Content-Type": "application/octet-stream"})


def _legit_passes_crypto(session: dict) -> dict:
    """The device's own, well-formed submission. Noise frames have no face,
    so the strict motion challenge fails *after* the nonce is consumed; what
    matters here is that every cryptographic check and the nonce passed."""
    body = _post(session, _envelope(session, _frames(seed=99))).json()
    assert body["binding_checks"].get("nonce_unused") is True, body
    assert body["binding_checks"].get("signature_valid") is True, body
    assert body["binding_checks"].get("nonce_consumed") is True, body
    return body


# ------------------------------------------------------------------ 1. sessions
class TestInvalidRequestsCannotKillSessions:
    def _attacks(self, session: dict):
        from crypto.signature import SignerIdentity

        other = SignerIdentity(alg="ML-DSA-65")
        other_session = _start()
        frames = _frames()
        yield "garbage envelope", lambda: client.post(f"/session/{session['session_id']}/verify-sealed", content=b"PQB1" + os.urandom(4000))
        yield "random ciphertext", lambda: _post(session, bytes(bytearray(_envelope(session, frames))[:8]) + os.urandom(20000))
        yield "forged signature (attacker key)", lambda: _post(session, _envelope(session, frames, signer=other))
        yield "stale timestamp", lambda: _post(session, _envelope(session, frames, timestamp=(datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()))
        yield "frames substituted after signing", lambda: _post(session, _envelope(session, frames, swap_after_signing=_frames(seed=7)))
        yield "envelope for another session", lambda: _post(session, _envelope(other_session, frames))
        tampered = bytearray(_envelope(session, frames))
        tampered[-50] ^= 1
        yield "tampered frames ciphertext", lambda: _post(session, bytes(tampered))
        yield "unsigned close", lambda: client.delete(f"/session/{session['session_id']}")
        yield "close signed by another key", lambda: client.delete(f"/session/{session['session_id']}", headers=_close_headers(session, other))

    def test_each_invalid_request_leaves_the_session_usable(self):
        names = [name for name, _ in self._attacks(_start())]
        for name in names:
            session = _start()
            attack = dict(self._attacks(session))[name]
            for _ in range(3):
                assert attack().status_code in (200, 400, 401), name
            _legit_passes_crypto(session)

    def test_concurrent_invalid_flood_does_not_block_the_device(self):
        from crypto.signature import SignerIdentity

        session = _start()
        other = SignerIdentity(alg="ML-DSA-65")
        forged = [_envelope(session, _frames(seed=i), signer=other) for i in range(12)]
        results = []
        barrier = threading.Barrier(len(forged) + 1)

        def send(body):
            barrier.wait()
            results.append(_post(session, body).json()["binding_checks"])

        threads = [threading.Thread(target=send, args=(b,)) for b in forged]
        for t in threads:
            t.start()
        barrier.wait()
        for t in threads:
            t.join()
        assert all(r.get("signature_valid") is False for r in results)
        _legit_passes_crypto(session)

    def test_valid_submission_still_consumes_the_nonce(self):
        session = _start()
        _legit_passes_crypto(session)
        replay = _post(session, _envelope(session, _frames(seed=99))).json()
        assert replay["binding_checks"]["nonce_unused"] is False

    def test_concurrent_valid_copies_pass_exactly_once(self):
        session = _start()
        body = _envelope(session, _frames(seed=3))
        out = []
        barrier = threading.Barrier(8)

        def send():
            barrier.wait()
            out.append(_post(session, body).json()["binding_checks"])

        threads = [threading.Thread(target=send) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert sum(1 for c in out if c.get("nonce_consumed") is True) == 1

    def test_signed_close_works(self):
        session = _start()
        assert client.delete(f"/session/{session['session_id']}", headers=_close_headers(session)).status_code == 200


class TestSignedSessionStart:
    def test_unsigned_start_is_refused(self):
        assert client.post("/session/start", json={"device_id": _enroll()}).status_code == 401

    def test_start_signed_by_another_key_is_refused(self):
        from crypto.signature import SignerIdentity

        assert client.post("/session/start", json=_start_request(_enroll(), SignerIdentity(alg="ML-DSA-65"))).status_code == 401

    def test_replayed_and_stale_starts_are_refused(self):
        body = _start_request(_enroll(), _device())
        assert client.post("/session/start", json=body).status_code == 200
        assert client.post("/session/start", json=body).status_code == 401
        old = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
        assert client.post("/session/start", json=_start_request(_enroll(), _device(), timestamp=old)).status_code == 401

    def test_per_device_limits(self, monkeypatch):
        from crypto.signature import SignerIdentity
        from server.config import settings

        dev = SignerIdentity(alg="ML-DSA-65")
        monkeypatch.setattr(settings, "max_active_sessions_per_device", 2)
        _start(dev), _start(dev)
        assert client.post("/session/start", json=_start_request(_enroll(dev), dev)).status_code == 429
        dev2 = SignerIdentity(alg="ML-DSA-65")
        monkeypatch.setattr(settings, "max_active_sessions_per_device", 10**6)
        monkeypatch.setattr(settings, "max_sessions_per_minute_per_device", 3)
        for _ in range(3):
            _start(dev2)
        assert client.post("/session/start", json=_start_request(_enroll(dev2), dev2)).status_code == 429
        # Another device is unaffected.
        _start(SignerIdentity(alg="ML-DSA-65"))

    def test_expired_session_is_gone(self, monkeypatch):
        from server.config import settings

        session = _start()
        monkeypatch.setattr(settings, "session_ttl_seconds", -1)
        assert _post(session, _envelope(session, _frames())).status_code == 404


# ----------------------------------------------------- 2. authenticated client
class TestAuthenticatedClientHardening:
    def _checks(self, frames, burst=8):
        session = _start()
        return _post(session, _envelope(session, frames, burst=burst)).json()

    def test_single_frame_is_rejected_before_decoding(self):
        body = self._checks(_frames(1), burst=0)
        assert body["binding_checks"]["submission_well_formed"] is False
        assert body["binding_checks"]["signature_valid"] is True
        assert "challenge_fulfilled" not in body["binding_checks"]

    def test_burst_length_is_fixed(self):
        assert self._checks(_frames(), burst=2)["binding_checks"]["submission_well_formed"] is False

    def test_oversized_and_bomb_frames_are_rejected_from_the_header(self):
        bomb = _png(np.zeros((4000, 4000, 3), dtype=np.uint8))  # 16 MP, compresses to a few kB
        assert len(bomb) < 200_000
        body = self._checks(_frames(15) + [bomb])
        assert body["binding_checks"]["submission_well_formed"] is False
        assert "pixels" in body["binding_reasons"][0] or "differ" in body["binding_reasons"][0]

    def test_non_image_frame_is_rejected(self):
        assert self._checks(_frames(15) + [b"not an image"])["binding_checks"]["submission_well_formed"] is False

    def test_body_size_is_capped(self, monkeypatch):
        from server.config import settings

        session = _start()
        monkeypatch.setattr(settings, "max_body_bytes", 50_000)
        assert _post(session, _envelope(session, _frames(size=128))).status_code == 413

    def test_motion_challenge_fails_closed_without_a_face(self):
        body = self._checks(_frames())  # noise: no face for MediaPipe
        assert body["binding_checks"]["submission_well_formed"] is True
        assert body["binding_checks"]["challenge_fulfilled"] is False
        assert body["detection_ran"] is False

    def test_lenient_fallback_only_with_sample_uploads(self):
        from detection.liveness_challenge import evaluate_challenge

        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        assert evaluate_challenge("Turn your head slightly left", "done", [frame, frame], strict=False)[0] is True
        assert evaluate_challenge("Turn your head slightly left", "done", [frame, frame], strict=True)[0] is False
        assert evaluate_challenge("Nod your head once", "done", [frame], strict=True)[0] is False

    def test_digit_challenge_is_not_issued_by_default(self):
        assert all("number" not in _start()["challenge"].lower() for _ in range(25))

    def test_pad_judges_several_frames_and_keeps_the_worst(self, monkeypatch):
        from detection.base import DetectionResult
        from server import main

        scores = [0.95] + [0.1] * 7  # a live first frame spliced in front of attack frames

        class Fake:
            def predict(self, frame):
                return DetectionResult(module="presentation_attack", score=scores[int(frame[0, 0, 0])], is_mock=False)

        monkeypatch.setattr(main, "_pad_detector", Fake())
        spread = [np.full((4, 4, 3), i, dtype=np.uint8) for i in range(8)]
        result = main._pad_over_frames(spread)
        assert result.details["pad_frame_indices"][0] == 0 and len(result.details["pad_frame_indices"]) == 3
        assert result.score == pytest.approx(0.1)


class TestAblationFlags:
    """The measurement-only ablations really remove the defence (so the
    report's ablation rows measure what they claim)."""

    def test_early_nonce_consumption_lets_invalid_requests_end_the_session(self, monkeypatch):
        from crypto.signature import SignerIdentity
        from server.config import settings

        monkeypatch.setattr(settings, "ablate_early_nonce_consume", True)
        session = _start()
        _post(session, _envelope(session, _frames(), signer=SignerIdentity(alg="ML-DSA-65")))
        body = _post(session, _envelope(session, _frames(seed=99))).json()
        assert body["binding_checks"]["nonce_unused"] is False
        session2 = _start()
        assert client.delete(f"/session/{session2['session_id']}").status_code == 200  # unsigned close accepted

    def test_shape_check_can_be_disabled(self, monkeypatch):
        from server.config import settings

        monkeypatch.setattr(settings, "enforce_submission_shape", False)
        session = _start()
        body = _post(session, _envelope(session, _frames(1), burst=0)).json()
        assert "submission_well_formed" not in body["binding_checks"]
