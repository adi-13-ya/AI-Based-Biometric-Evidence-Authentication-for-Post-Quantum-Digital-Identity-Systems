"""Crypto round-trip tests (spec Definition of Done: "Unit tests: crypto
round-trip (keygen/encap/decap, sign/verify)").

ML-KEM/ML-DSA tests are skipped (not faked) when liboqs isn't available in
this environment - see tests/conftest.py and docs/SETUP.md. The classical
baseline tests always run natively since `cryptography` has no such
dependency.
"""
from __future__ import annotations

import pytest

from conftest import skip_if_no_oqs

from crypto import classical_baseline
from crypto.signature import build_binding_payload


@skip_if_no_oqs
class TestMlKem:
    def test_round_trip_shared_secret_matches(self):
        from crypto.kem import ServerKemSession, client_encapsulate

        with ServerKemSession(alg="ML-KEM-768") as server:
            server_public_key = server.generate_keypair()
            ciphertext, client_secret = client_encapsulate(server_public_key, alg="ML-KEM-768")
            server_secret = server.decapsulate(ciphertext)

        assert client_secret == server_secret
        assert len(client_secret) == 32

    def test_derive_channel_key_is_deterministic_and_correct_length(self):
        from crypto.kem import derive_channel_key

        secret = b"\x01" * 32
        key1 = derive_channel_key(secret, salt=b"fixed-salt")
        key2 = derive_channel_key(secret, salt=b"fixed-salt")
        assert key1 == key2
        assert len(key1) == 32

    def test_unsupported_alg_raises(self):
        from crypto.kem import ServerKemSession

        with pytest.raises(ValueError):
            ServerKemSession(alg="not-a-real-alg")

    def test_tampered_ciphertext_does_not_match(self):
        from crypto.kem import ServerKemSession, client_encapsulate

        with ServerKemSession(alg="ML-KEM-768") as server:
            server_public_key = server.generate_keypair()
            ciphertext, client_secret = client_encapsulate(server_public_key, alg="ML-KEM-768")
            tampered = bytes([ciphertext[0] ^ 0xFF]) + ciphertext[1:]
            server_secret = server.decapsulate(tampered)

        # ML-KEM is IND-CCA2; a tampered ciphertext decapsulates to an
        # unrelated (implicit-rejection) secret rather than raising.
        assert server_secret != client_secret


@skip_if_no_oqs
class TestMlDsa:
    def test_sign_and_verify_round_trip(self):
        from crypto.signature import SignerIdentity, verify

        with SignerIdentity(alg="ML-DSA-65") as signer:
            message = build_binding_payload(b"\x00" * 32, b"nonce-bytes-123", "2026-09-07T00:00:00Z", "7")
            signature = signer.sign(message)
            assert verify(message, signature, signer.public_key, alg="ML-DSA-65") is True

    def test_verify_rejects_tampered_message(self):
        from crypto.signature import SignerIdentity, verify

        with SignerIdentity(alg="ML-DSA-65") as signer:
            message = build_binding_payload(b"\x00" * 32, b"nonce-bytes-123", "2026-09-07T00:00:00Z", "7")
            signature = signer.sign(message)
            tampered_message = message[:-1] + b"\x01"
            assert verify(tampered_message, signature, signer.public_key, alg="ML-DSA-65") is False

    def test_verify_rejects_wrong_public_key(self):
        from crypto.signature import SignerIdentity, verify

        with SignerIdentity(alg="ML-DSA-65") as signer_a, SignerIdentity(alg="ML-DSA-65") as signer_b:
            message = build_binding_payload(b"\x00" * 32, b"nonce-bytes-123", "2026-09-07T00:00:00Z", "7")
            signature = signer_a.sign(message)
            assert verify(message, signature, signer_b.public_key, alg="ML-DSA-65") is False

    def test_unsupported_alg_raises(self):
        from crypto.signature import SignerIdentity

        with pytest.raises(ValueError):
            SignerIdentity(alg="not-a-real-alg")


def test_build_binding_payload_is_tagged_and_length_prefixed():
    payload = build_binding_payload(b"HASH", b"NONCE", "ts-1", "chal-1")
    assert payload.startswith(b"PQBIO-EVIDENCE-v2")
    assert b"\x00\x00\x00\x04HASH\x00\x00\x00\x05NONCE\x00\x00\x00\x04ts-1\x00\x00\x00\x06chal-1" in payload


def test_binding_payload_field_boundaries_are_signed():
    """v1 concatenation let a byte move between timestamp and challenge
    response without changing the message; v2 does not."""
    assert build_binding_payload(b"H", b"N", "ts-1", "7") != build_binding_payload(b"H", b"N", "ts-17", "")


def test_binding_payload_covers_session_kem_and_metadata():
    base = build_binding_payload(b"H", b"N", "t", "r", session_id="s", kem_ciphertext=b"ct", burst_frames=8, frame_timestamps=[0.0, 0.033])
    for change in ({"session_id": "s2"}, {"kem_ciphertext": b"ct2"}, {"burst_frames": 7}, {"frame_timestamps": [0.0, 0.034]},
                   {"frame_timestamps": None}):
        kwargs = {"session_id": "s", "kem_ciphertext": b"ct", "burst_frames": 8, "frame_timestamps": [0.0, 0.033], **change}
        assert build_binding_payload(b"H", b"N", "t", "r", **kwargs) != base


def test_ecdsa_p256_baseline_round_trip():
    from crypto.signature import SignerIdentity, verify

    with SignerIdentity(alg="ECDSA-P256") as signer:
        sig = signer.sign(b"msg")
        assert len(signer.public_key) == 65
        assert verify(b"msg", sig, signer.public_key, alg="ECDSA-P256") is True
        assert verify(b"msh", sig, signer.public_key, alg="ECDSA-P256") is False
        again = SignerIdentity(alg="ECDSA-P256", secret_key=signer.export_secret_key())
        assert again.public_key == signer.public_key


class TestClassicalBaseline:
    def test_rsa_round_trip(self):
        private_key, public_key = classical_baseline.rsa_generate_keypair()
        plaintext = b"\x02" * 32  # AES-256 key sized payload
        ciphertext = classical_baseline.rsa_encrypt(public_key, plaintext)
        assert classical_baseline.rsa_decrypt(private_key, ciphertext) == plaintext

    def test_ecdsa_round_trip(self):
        private_key, public_key = classical_baseline.ecdsa_generate_keypair()
        message = b"evidence-hash-session-id-timestamp"
        signature = classical_baseline.ecdsa_sign(private_key, message)
        assert classical_baseline.ecdsa_verify(public_key, message, signature) is True

    def test_ecdsa_rejects_tampered_message(self):
        private_key, public_key = classical_baseline.ecdsa_generate_keypair()
        message = b"evidence-hash-session-id-timestamp"
        signature = classical_baseline.ecdsa_sign(private_key, message)
        assert classical_baseline.ecdsa_verify(public_key, message + b"x", signature) is False

    def test_public_key_size_bytes_is_positive(self):
        _, public_key = classical_baseline.ecdsa_generate_keypair()
        assert classical_baseline.public_key_size_bytes(public_key) > 0
