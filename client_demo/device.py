"""The capture device's own ML-DSA identity (client side of the trust split).

The device generates its ML-DSA keypair locally, stores the secret key in a
key file only it reads (mode 0600), and sends the server nothing but the
public key, once, at enrollment. Every evidence submission is signed here,
in the client process; the server only verifies. Nothing in this module is
imported by server/, and nothing in server/ can sign.

Key file: $PQBIO_DEVICE_KEY, default ~/.pqbio/device_<alg>.json
    {"alg": ..., "public_key": hex, "secret_key": hex, "device_id": {server_url: id},
     "server_keys": {server_url: {"sig_alg": ..., "public_key": hex}}}

`server_keys` are the pinned server identity keys that session offers must
verify against (docs/PROTOCOL.md, Section 4).

Enrollment here is trust-on-first-use, which is enough to separate the two
roles for the prototype. A deployment would bind enrollment to an onboarding
event (KYC check, or a hardware attestation of a key held in a TEE/secure
element) - see the report's limitations.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from crypto.signature import DEFAULT_SIG_ALG, SignerIdentity, build_binding_payload


def default_key_path(alg: str) -> Path:
    override = os.getenv("PQBIO_DEVICE_KEY")
    return Path(override) if override else Path.home() / ".pqbio" / f"device_{alg}.json"


class DeviceIdentity:
    def __init__(self, alg: str = DEFAULT_SIG_ALG, key_path: Path | None = None):
        self.alg = alg
        self.key_path = Path(key_path) if key_path else default_key_path(alg)
        self._enrollments: dict[str, str] = {}
        self._server_keys: dict[str, dict] = {}
        if self.key_path.exists():
            data = json.loads(self.key_path.read_text())
            if data["alg"] != alg:
                raise ValueError(f"{self.key_path} holds an {data['alg']} key, not {alg}")
            self._signer = SignerIdentity(alg, bytes.fromhex(data["secret_key"]), bytes.fromhex(data["public_key"]))
            self._enrollments = data.get("device_id", {})
            self._server_keys = data.get("server_keys", {})
        else:
            self._signer = SignerIdentity(alg)
            self._save()

    @property
    def public_key(self) -> bytes:
        return self._signer.public_key

    def device_id_for(self, server_url: str) -> str | None:
        return self._enrollments.get(server_url)

    def remember_enrollment(self, server_url: str, device_id: str) -> None:
        self._enrollments[server_url] = device_id
        self._save()

    def forget_enrollment(self, server_url: str) -> None:
        self._enrollments.pop(server_url, None)
        self._save()

    def pinned_server_key(self, server_url: str) -> dict | None:
        return self._server_keys.get(server_url)

    def pin_server_key(self, server_url: str, sig_alg: str, public_key: bytes) -> None:
        self._server_keys[server_url] = {"sig_alg": sig_alg, "public_key": public_key.hex()}
        self._save()

    def sign_binding(self, evidence_hash: bytes, session_nonce: bytes, timestamp: str, challenge_response: str, **context) -> bytes:
        """Signature over the evidence-binding message (crypto/signature.py:
        build_binding_payload). `context`: session_id, kem_ciphertext,
        burst_frames, frame_timestamps."""
        return self._signer.sign(build_binding_payload(evidence_hash, session_nonce, timestamp, challenge_response, **context))

    def _save(self) -> None:
        self.key_path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({
            "alg": self.alg,
            "public_key": self._signer.public_key.hex(),
            "secret_key": self._signer.export_secret_key().hex(),
            "device_id": self._enrollments,
            "server_keys": self._server_keys,
        })
        fd = os.open(self.key_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as fh:
            fh.write(payload)

    def close(self) -> None:
        self._signer.close()
