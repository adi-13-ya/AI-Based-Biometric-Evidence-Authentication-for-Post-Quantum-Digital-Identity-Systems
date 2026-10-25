"""Enrolled capture devices: device_id -> ML-DSA public key.

The server side of the trust split (client_demo/device.py is the other
side). Only public keys are ever stored here. device_id is derived from the
key itself (first 16 bytes of SHA-256 over alg || public key, hex), so
re-enrolling the same key is idempotent and an id cannot be pointed at a
different key.

In-memory like server/session_store.py; a deployment would persist this and
gate enrollment on an onboarding/attestation step.
"""
from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass

from crypto.signature import SUPPORTED_SIG_ALGS


@dataclass(frozen=True)
class Device:
    device_id: str
    sig_alg: str
    public_key: bytes


def device_id_of(sig_alg: str, public_key: bytes) -> str:
    return hashlib.sha256(sig_alg.encode("ascii") + b"\x00" + public_key).hexdigest()[:32]


class DeviceRegistry:
    # ML-DSA public key sizes (FIPS 204, Table 2)
    _PK_LEN = {"ML-DSA-44": 1312, "ML-DSA-65": 1952, "ML-DSA-87": 2592}

    def __init__(self):
        self._devices: dict[str, Device] = {}
        self._lock = threading.Lock()

    def enroll(self, sig_alg: str, public_key: bytes) -> Device:
        if sig_alg not in SUPPORTED_SIG_ALGS:
            raise ValueError(f"unsupported signature algorithm {sig_alg!r}")
        if len(public_key) != self._PK_LEN[sig_alg]:
            raise ValueError(f"{sig_alg} public key must be {self._PK_LEN[sig_alg]} bytes, got {len(public_key)}")
        device = Device(device_id_of(sig_alg, public_key), sig_alg, public_key)
        with self._lock:
            self._devices[device.device_id] = device
        return device

    def get(self, device_id: str) -> Device | None:
        with self._lock:
            return self._devices.get(device_id)


registry = DeviceRegistry()
