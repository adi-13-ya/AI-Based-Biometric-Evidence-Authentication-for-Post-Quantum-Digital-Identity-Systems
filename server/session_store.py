"""In-memory session state.

A prototype-scale store (dict keyed by session_id) is sufficient for a
single-process demo server; a production system would back this with Redis
or similar so nonce single-use is enforced across replicas. Holds the native
ML-KEM keypair for the lifetime of the session, so sessions must be closed
to free it. A session is opened for one enrolled device and keeps only that
device's ML-DSA *public* key (server/device_registry.py); no signing key
exists on the server.
"""
from __future__ import annotations

import base64
import os
import secrets
import threading
import time
import uuid
from dataclasses import dataclass, field

from crypto.kem import ServerKemSession
from server.config import settings

CHALLENGE_PROMPTS = [
    "Turn your head slightly left",
    "Turn your head slightly right",
    "Nod your head once",
    "Say the number {digits}",
]


def _generate_challenge() -> str:
    prompt = secrets.choice(CHALLENGE_PROMPTS)
    if "{digits}" in prompt:
        digits = "".join(secrets.choice("0123456789") for _ in range(4))
        prompt = prompt.format(digits=digits)
    return prompt


@dataclass
class Session:
    session_id: str
    nonce: bytes
    challenge: str
    created_at: float
    kem: ServerKemSession
    device_id: str
    device_public_key: bytes
    sig_alg: str
    nonce_used: bool = False
    kem_established: bool = False
    used_evidence_hashes: set = field(default_factory=set)
    _nonce_lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def consume_nonce(self) -> bool:
        """Atomically mark the nonce used. True only for the first caller:
        of any number of concurrent submissions on one session, exactly one
        can pass the nonce check (a read-then-set without the lock let two
        simultaneous copies of the same submission both pass)."""
        with self._nonce_lock:
            first = not self.nonce_used
            self.nonce_used = True
            return first

    def is_fresh(self, timestamp: float, window_seconds: int | None = None) -> bool:
        window = settings.freshness_window_seconds if window_seconds is None else window_seconds
        return abs(time.time() - timestamp) <= window

    def close(self) -> None:
        self.kem.close()


class SessionStore:
    def __init__(self):
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    def create(self, device_id: str, device_public_key: bytes, sig_alg: str) -> Session:
        session_id = str(uuid.uuid4())
        nonce = os.urandom(settings.nonce_bytes)
        challenge = _generate_challenge()

        kem = ServerKemSession(alg=settings.kem_alg)
        kem.generate_keypair()

        session = Session(
            session_id=session_id,
            nonce=nonce,
            challenge=challenge,
            created_at=time.time(),
            kem=kem,
            device_id=device_id,
            device_public_key=device_public_key,
            sig_alg=sig_alg,
        )
        with self._lock:
            self._sessions[session_id] = session
        return session

    def get(self, session_id: str) -> Session | None:
        with self._lock:
            return self._sessions.get(session_id)

    def delete(self, session_id: str) -> None:
        with self._lock:
            session = self._sessions.pop(session_id, None)
        if session is not None:
            session.close()

    def all_active(self) -> list[Session]:
        with self._lock:
            return list(self._sessions.values())


def nonce_b64(nonce: bytes) -> str:
    return base64.b64encode(nonce).decode("ascii")


# Process-wide singleton, matching FastAPI's typical app-state pattern for a
# small prototype server.
store = SessionStore()
