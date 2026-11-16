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

MOTION_PROMPTS = [
    "Turn your head slightly left",
    "Turn your head slightly right",
    "Nod your head once",
]
DIGIT_PROMPTS = ["Say the number {digits}"]
CHALLENGE_PROMPTS = MOTION_PROMPTS + DIGIT_PROMPTS


def issued_prompts() -> list[str]:
    """Prompts the server may issue (settings.challenge_kinds). The digit
    prompt is only typed back by the client (no audio pipeline), so it binds
    a session but proves nothing about liveness; it is off by default."""
    kinds = {k.strip() for k in settings.challenge_kinds.split(",") if k.strip()}
    prompts = (MOTION_PROMPTS if "motion" in kinds else []) + (DIGIT_PROMPTS if "digits" in kinds else [])
    return prompts or MOTION_PROMPTS


def _generate_challenge() -> str:
    prompt = secrets.choice(issued_prompts())
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
        simultaneous copies of the same submission both pass).

        Called only once a submission has been authenticated: valid device
        signature and frames that hash to the signed evidence hash
        (server/main.py). A request that fails any of those checks reads
        `nonce_used` but never consumes it, so an unauthenticated request
        cannot invalidate the session for the device that owns it."""
        with self._nonce_lock:
            first = not self.nonce_used
            self.nonce_used = True
            return first

    def expired(self, now: float | None = None) -> bool:
        return (now or time.time()) - self.created_at > settings.session_ttl_seconds

    def is_fresh(self, timestamp: float, window_seconds: int | None = None) -> bool:
        window = settings.freshness_window_seconds if window_seconds is None else window_seconds
        return abs(time.time() - timestamp) <= window

    def close(self) -> None:
        self.kem.close()


class SessionLimitError(Exception):
    """A per-device session limit was reached (HTTP 429)."""


class SessionStore:
    def __init__(self):
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()
        self._starts: dict[str, list[float]] = {}  # device_id -> recent session start times
        self._start_nonces: dict[str, float] = {}  # signed session-start request nonces seen recently

    def _purge(self, now: float) -> list[Session]:
        dead = [sid for sid, s in self._sessions.items() if s.expired(now)]
        closed = [self._sessions.pop(sid) for sid in dead]
        for dev in list(self._starts):
            self._starts[dev] = [t for t in self._starts[dev] if now - t < 60]
            if not self._starts[dev]:
                del self._starts[dev]
        horizon = 2 * settings.freshness_window_seconds
        for n in [n for n, t in self._start_nonces.items() if now - t > horizon]:
            del self._start_nonces[n]
        return closed

    def claim_start_nonce(self, request_nonce: str) -> bool:
        """Record a signed session-start request nonce; False if seen within
        twice the freshness window (a replayed start request)."""
        with self._lock:
            if request_nonce in self._start_nonces:
                return False
            self._start_nonces[request_nonce] = time.time()
            return True

    def create(self, device_id: str, device_public_key: bytes, sig_alg: str) -> Session:
        now = time.time()
        with self._lock:
            closed = self._purge(now)
            active = sum(1 for s in self._sessions.values() if s.device_id == device_id and not s.nonce_used)
            recent = len(self._starts.get(device_id, []))
            if active >= settings.max_active_sessions_per_device:
                limit = f"{active} unused sessions open for this device (limit {settings.max_active_sessions_per_device})"
            elif recent >= settings.max_sessions_per_minute_per_device:
                limit = f"{recent} sessions started in the last minute (limit {settings.max_sessions_per_minute_per_device})"
            else:
                limit = ""
                self._starts.setdefault(device_id, []).append(now)
        for s in closed:
            s.close()
        if limit:
            raise SessionLimitError(limit)
        session_id = str(uuid.uuid4())
        nonce = os.urandom(settings.nonce_bytes)
        challenge = _generate_challenge()

        kem = ServerKemSession(alg=settings.kem_alg)
        kem.generate_keypair()

        session = Session(
            session_id=session_id,
            nonce=nonce,
            challenge=challenge,
            created_at=now,
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
            session = self._sessions.get(session_id)
            if session is not None and session.expired():
                return None
            return session

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
