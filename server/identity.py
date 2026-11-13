"""The server's long-term signing identity.

Used for one thing only: signing each session offer (session id, nonce,
challenge, ephemeral ML-KEM public key, device id, issue time), so the
client can tell the ML-KEM key it is about to encapsulate to really comes
from this server (docs/PROTOCOL.md, Section 4). It never signs evidence:
evidence signatures come only from capture devices (client_demo/device.py),
and the two messages carry different domain tags (crypto/signature.py).

Key file: $SERVER_KEY_PATH, default ~/.pqbio/server_identity_<alg>.json,
mode 0600, generated on first start. Clients pin the public key: either
distributed out of band (PQBIO_SERVER_PUBKEY) or, in the prototype,
trust-on-first-use from GET /server/identity.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from crypto.signature import SignerIdentity
from server.config import settings


def key_id_of(sig_alg: str, public_key: bytes) -> str:
    return hashlib.sha256(sig_alg.encode("ascii") + b"\x00" + public_key).hexdigest()[:32]


class ServerIdentity:
    def __init__(self, alg: str, key_path: Path):
        self.alg = alg
        self.key_path = key_path
        self._signer: SignerIdentity | None = None

    def _load(self) -> SignerIdentity:
        # Loaded on first use so importing the app does not need liboqs.
        if self._signer is None:
            if self.key_path.exists():
                data = json.loads(self.key_path.read_text())
                self._signer = SignerIdentity(self.alg, bytes.fromhex(data["secret_key"]), bytes.fromhex(data["public_key"]))
            else:
                self._signer = SignerIdentity(self.alg)
                self.key_path.parent.mkdir(parents=True, exist_ok=True)
                fd = os.open(self.key_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
                with os.fdopen(fd, "w") as fh:
                    fh.write(json.dumps({"alg": self.alg, "public_key": self._signer.public_key.hex(),
                                         "secret_key": self._signer.export_secret_key().hex()}))
        return self._signer

    @property
    def public_key(self) -> bytes:
        return self._load().public_key

    @property
    def key_id(self) -> str:
        return key_id_of(self.alg, self.public_key)

    def sign_offer(self, offer: bytes) -> bytes:
        return self._load().sign(offer)


def _default_path() -> Path:
    return Path(settings.server_key_path) if settings.server_key_path else Path.home() / ".pqbio" / f"server_identity_{settings.server_sig_alg}.json"


identity = ServerIdentity(settings.server_sig_alg, _default_path())
