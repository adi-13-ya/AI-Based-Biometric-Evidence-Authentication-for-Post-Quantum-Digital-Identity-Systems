"""ML-DSA (FIPS 204) signing/verification, via oqs-python / liboqs.

Two signed messages exist, each with its own domain tag so a signature over
one can never be replayed as the other (docs/PROTOCOL.md, Section 3):

  * the evidence binding (device key; build_binding_payload), and
  * the session offer (server identity key; build_session_offer), which
    authenticates the server's ephemeral ML-KEM public key to the client.

Both are length-prefixed field lists, so field boundaries are part of what
is signed (the v1 payload was a bare concatenation, in which a byte could
move between timestamp and challenge_response without changing the message).

ECDSA-P256 is also accepted as a device algorithm. It is quantum-vulnerable
and exists only so the classical-vs-post-quantum comparison can run the
same protocol and check order (report/scripts/distributed_experiments.py);
it is never a default.

Trust split: `SignerIdentity` is only ever instantiated on the capture
device (client_demo/device.py), which generates the keypair, keeps the
secret key in its own key file and enrolls only the public key with the
server (POST /device/enroll). The server imports nothing but `verify` and
`build_binding_payload` from this module, so it cannot produce a valid
binding signature on a client's behalf.
"""
from __future__ import annotations

import os

# `oqs` is imported lazily inside each function/class below (not at module
# level) so this module - and pure helpers like `build_binding_payload` - can
# be imported where liboqs isn't installed (see crypto/kem.py for the same
# rationale). The real oqs/liboqs is still always what performs signing.

DEFAULT_SIG_ALG = os.getenv("SIG_ALG", "ML-DSA-65")

PQ_SIG_ALGS = {"ML-DSA-44", "ML-DSA-65", "ML-DSA-87"}
CLASSICAL_SIG_ALGS = {"ECDSA-P256"}  # measurement baseline only
SUPPORTED_SIG_ALGS = PQ_SIG_ALGS | CLASSICAL_SIG_ALGS

BINDING_TAG = b"PQBIO-EVIDENCE-v2"
SESSION_OFFER_TAG = b"PQBIO-SESSION-OFFER-v1"
SESSION_REQUEST_TAG = b"PQBIO-SESSION-REQUEST-v1"
SESSION_CLOSE_TAG = b"PQBIO-SESSION-CLOSE-v1"


def _check_alg(alg: str) -> None:
    if alg not in SUPPORTED_SIG_ALGS:
        raise ValueError(f"Unsupported signature alg {alg!r}; expected one of {sorted(SUPPORTED_SIG_ALGS)}")


class SignerIdentity:
    """Holds one party's signing keypair. Pass `secret_key` and
    `public_key` to reload a keypair saved with `export_secret_key()`;
    otherwise a fresh one is generated. Must be closed (or used as a
    context manager) when done."""

    def __init__(self, alg: str = DEFAULT_SIG_ALG, secret_key: bytes | None = None, public_key: bytes | None = None):
        _check_alg(alg)
        self.alg = alg
        if alg in CLASSICAL_SIG_ALGS:
            self._sig = _EcdsaP256(secret_key)
            self.public_key: bytes = self._sig.public_key
            return
        import oqs

        if secret_key is not None:
            if public_key is None:
                raise ValueError("reloading a keypair needs its public key too")
            self._sig = oqs.Signature(alg, secret_key=secret_key)
            self.public_key = public_key
        else:
            self._sig = oqs.Signature(alg)
            self.public_key = self._sig.generate_keypair()

    def sign(self, message: bytes) -> bytes:
        return self._sig.sign(message)

    def export_secret_key(self) -> bytes:
        return self._sig.export_secret_key()

    def close(self) -> None:
        self._sig.free()

    def __enter__(self) -> "SignerIdentity":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


class _EcdsaP256:
    """ECDSA-P256 with the same interface as oqs.Signature. Public key is the
    65-byte uncompressed point, secret key the 32-byte scalar."""

    def __init__(self, secret_key: bytes | None = None):
        from cryptography.hazmat.primitives.asymmetric import ec

        if secret_key is None:
            self._key = ec.generate_private_key(ec.SECP256R1())
        else:
            self._key = ec.derive_private_key(int.from_bytes(secret_key, "big"), ec.SECP256R1())
        from cryptography.hazmat.primitives import serialization

        self.public_key = self._key.public_key().public_bytes(serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint)

    def sign(self, message: bytes) -> bytes:
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import ec

        return self._key.sign(message, ec.ECDSA(hashes.SHA256()))

    def export_secret_key(self) -> bytes:
        return self._key.private_numbers().private_value.to_bytes(32, "big")

    def free(self) -> None:
        pass


def _ecdsa_verify(message: bytes, signature: bytes, public_key: bytes) -> bool:
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import ec

    try:
        key = ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256R1(), public_key)
        key.verify(signature, message, ec.ECDSA(hashes.SHA256()))
        return True
    except (InvalidSignature, ValueError):
        return False


def verify(message: bytes, signature: bytes, public_key: bytes, alg: str = DEFAULT_SIG_ALG) -> bool:
    _check_alg(alg)
    if alg in CLASSICAL_SIG_ALGS:
        return _ecdsa_verify(message, signature, public_key)
    import oqs

    with oqs.Signature(alg) as verifier:
        return verifier.verify(message, signature, public_key)


def encode_fields(tag: bytes, *fields: bytes) -> bytes:
    """tag || for each field: u32 big-endian length || field."""
    return tag + b"".join(len(f).to_bytes(4, "big") + f for f in fields)


def encode_meta(burst_frames: int = 0, frame_timestamps: list[float] | None = None) -> bytes:
    """Canonical bytes of the header fields the injection check reads, so
    they are signed too: u32 burst_frames, then u32 count (0xFFFFFFFF for
    none) and one IEEE-754 float64 per capture timestamp."""
    import struct

    out = struct.pack(">I", int(burst_frames))
    if frame_timestamps is None:
        return out + struct.pack(">I", 0xFFFFFFFF)
    return out + struct.pack(">I", len(frame_timestamps)) + b"".join(struct.pack(">d", float(t)) for t in frame_timestamps)


def build_binding_payload(
    evidence_hash: bytes,
    session_nonce: bytes,
    timestamp: str,
    challenge_response: str = "",
    *,
    session_id: str = "",
    kem_ciphertext: bytes = b"",
    burst_frames: int = 0,
    frame_timestamps: list[float] | None = None,
) -> bytes:
    """The evidence-binding message the device signs (docs/PROTOCOL.md, 3.2):

        "PQBIO-EVIDENCE-v2" || len-prefixed(
            evidence_hash, session_nonce, timestamp, challenge_response,
            session_id, SHA-256(kem_ciphertext) or empty, encode_meta(...))

    `session_nonce` is the random single-use nonce the server issued at
    session start (server/session_store.py Session.nonce), and the server
    always rebuilds this message from its own stored copy. Binding the
    ML-KEM ciphertext ties the signature to one encapsulation, so a party
    that intercepts a submission cannot re-wrap the signed header under an
    encapsulation of its own. It is empty on the plaintext test endpoint.
    """
    import hashlib

    return encode_fields(
        BINDING_TAG,
        evidence_hash,
        session_nonce,
        timestamp.encode("utf-8"),
        challenge_response.encode("utf-8"),
        session_id.encode("utf-8"),
        hashlib.sha256(kem_ciphertext).digest() if kem_ciphertext else b"",
        encode_meta(burst_frames, frame_timestamps),
    )


def build_session_offer(
    *,
    session_id: str,
    session_nonce: bytes,
    challenge: str,
    kem_alg: str,
    kem_public_key: bytes,
    device_id: str,
    issued_at: float,
    freshness_window_seconds: int,
) -> bytes:
    """The message the server's long-term identity key signs at session start
    (docs/PROTOCOL.md, 3.1). The client verifies it against the pinned server
    key before it encapsulates to `kem_public_key`, which is how a client
    knows the ephemeral ML-KEM key is the server's and not a man in the
    middle's."""
    return encode_fields(
        SESSION_OFFER_TAG,
        session_id.encode("utf-8"),
        session_nonce,
        challenge.encode("utf-8"),
        kem_alg.encode("ascii"),
        kem_public_key,
        device_id.encode("ascii"),
        repr(float(issued_at)).encode("ascii"),
        str(int(freshness_window_seconds)).encode("ascii"),
    )


def build_session_request(*, device_id: str, timestamp: str, request_nonce: bytes) -> bytes:
    """What the device signs to open a session (docs/PROTOCOL.md, 3.1). Only
    the holder of the enrolled key can open sessions for a device, so nobody
    else can spend that device's per-device session budget."""
    return encode_fields(SESSION_REQUEST_TAG, device_id.encode("ascii"), timestamp.encode("utf-8"), request_nonce)


def build_session_close(*, session_id: str, session_nonce: bytes) -> bytes:
    """What the device signs to close its own session early. An unsigned
    close request would let anyone who learns a session id end it."""
    return encode_fields(SESSION_CLOSE_TAG, session_id.encode("utf-8"), session_nonce)


def sig_details(alg: str = DEFAULT_SIG_ALG) -> dict:
    """Algorithm metadata (key/signature sizes) - used by the evaluation
    harness for the PQC-vs-classical payload-size comparison."""
    _check_alg(alg)
    if alg in CLASSICAL_SIG_ALGS:
        return {"name": alg, "length_public_key": 65, "length_secret_key": 32, "length_signature": 72, "claimed_nist_level": 0}
    import oqs

    with oqs.Signature(alg) as sig:
        return dict(sig.details)
