"""ML-DSA (FIPS 204) signing/verification, via oqs-python / liboqs.

Used to sign the transaction-binding payload described in spec Section 7:
    binding_payload = evidence_hash || session_nonce || timestamp || challenge_response

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

SUPPORTED_SIG_ALGS = {"ML-DSA-44", "ML-DSA-65", "ML-DSA-87"}


def _check_alg(alg: str) -> None:
    if alg not in SUPPORTED_SIG_ALGS:
        raise ValueError(f"Unsupported signature alg {alg!r}; expected one of {sorted(SUPPORTED_SIG_ALGS)}")


class SignerIdentity:
    """Holds one party's ML-DSA signing keypair. Pass `secret_key` and
    `public_key` to reload a keypair saved with `export_secret_key()`;
    otherwise a fresh one is generated. Must be closed (or used as a
    context manager) when done."""

    def __init__(self, alg: str = DEFAULT_SIG_ALG, secret_key: bytes | None = None, public_key: bytes | None = None):
        import oqs

        _check_alg(alg)
        self.alg = alg
        if secret_key is not None:
            if public_key is None:
                raise ValueError("reloading a keypair needs its public key too")
            self._sig = oqs.Signature(alg, secret_key=secret_key)
            self.public_key: bytes = public_key
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


def verify(message: bytes, signature: bytes, public_key: bytes, alg: str = DEFAULT_SIG_ALG) -> bool:
    import oqs

    _check_alg(alg)
    with oqs.Signature(alg) as verifier:
        return verifier.verify(message, signature, public_key)


def build_binding_payload(
    evidence_hash: bytes,
    session_nonce: bytes,
    timestamp: str,
    challenge_response: str = "",
) -> bytes:
    """Serialize the Section 7 binding payload:
    evidence_hash || session_nonce || timestamp || challenge_response

    `session_nonce` is the random single-use nonce the server issued at
    session start (server/session_store.py Session.nonce) - NOT the
    session_id/UUID used to look the session up over HTTP. Using the actual
    nonce here (rather than the session_id) is what session/binding.py's
    single-use enforcement is protecting.

    Note: Section 5.2's illustrative snippet omits challenge_response; Section
    7's full binding design (the mechanism actually implemented in
    binding/session_binding.py) includes it, so that is what this builds.
    """
    return evidence_hash + session_nonce + timestamp.encode("utf-8") + challenge_response.encode("utf-8")


def sig_details(alg: str = DEFAULT_SIG_ALG) -> dict:
    """Algorithm metadata (key/signature sizes) - used by the evaluation
    harness for the PQC-vs-classical payload-size comparison."""
    import oqs

    _check_alg(alg)
    with oqs.Signature(alg) as sig:
        return dict(sig.details)
