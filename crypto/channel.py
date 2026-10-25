"""AES-256-GCM payload sealing over an ML-KEM shared secret (spec Section 5.2:
"shared_secret is then used as an AES-256-GCM key for the actual
video/data channel").

The channel key is k = HKDF-SHA256(K, salt=session_nonce) (crypto/kem.py),
so it is bound to one session, and the session_id is authenticated as GCM
associated data so a payload sealed for one session cannot be submitted
under another. Shared by the client (sealing) and server (opening) so both
sides derive and use the key identically.
"""
from __future__ import annotations

import hashlib
import os

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from crypto.kem import derive_channel_key

AEAD_NONCE_LEN = 12  # 96-bit GCM nonce


def seal_payload(shared_secret: bytes, session_nonce: bytes, session_id: str, plaintext: bytes) -> tuple[bytes, bytes]:
    """Encrypt `plaintext`, returning (aead_nonce, ciphertext_with_tag)."""
    key = derive_channel_key(shared_secret, salt=session_nonce)
    aead_nonce = os.urandom(AEAD_NONCE_LEN)
    return aead_nonce, AESGCM(key).encrypt(aead_nonce, plaintext, session_id.encode("utf-8"))


def open_payload(shared_secret: bytes, session_nonce: bytes, session_id: str, aead_nonce: bytes, ciphertext: bytes) -> bytes:
    """Decrypt and authenticate; raises cryptography.exceptions.InvalidTag
    on any tampering or session mismatch."""
    key = derive_channel_key(shared_secret, salt=session_nonce)
    return AESGCM(key).decrypt(aead_nonce, ciphertext, session_id.encode("utf-8"))


def evidence_hash_of(frame_bytes: list[bytes]) -> bytes:
    """Canonical evidence hash: SHA-256 over the encoded bytes of every
    submitted frame, in order - exactly the bytes the server receives, so it
    can recompute and compare without re-encoding anything. Each frame is
    length-prefixed so frame boundaries are part of what is signed."""
    h = hashlib.sha256()
    for chunk in frame_bytes:
        h.update(len(chunk).to_bytes(8, "big"))
        h.update(chunk)
    return h.digest()


# --------------------------------------------------------------------------
# Header-first sealed envelope (POST /session/{id}/verify-sealed)
#
#   b"PQB1" | u16 len(kem_ct) | kem_ct
#           | 12 B nonce_h | u32 len(header_ct) | header_ct
#           | 12 B nonce_f | frames_ct (to the end of the body)
#
# header_ct = AES-256-GCM(k, nonce_h, JSON header, AAD = session_id || 0 || "header")
# frames_ct = AES-256-GCM(k, nonce_f, frames,      AAD = session_id || 0 || "frames")
# frames    = concatenation of (u32 length || encoded frame bytes)
#
# The header (evidence hash, timestamp, challenge response, signature, ...)
# is a few kilobytes and comes first, so the server can authenticate it and
# run the nonce / freshness / signature checks while the frames are still
# arriving, and reject without reading, decrypting or base64-decoding the
# multi-megabyte frame data. Both parts use the one channel key; the AAD
# labels keep a header from being opened as frames or vice versa.
# --------------------------------------------------------------------------
import json as _json
import struct as _struct

ENVELOPE_MAGIC = b"PQB1"
_HEADER_LIMIT = 64 * 1024


def _aad(session_id: str, part: str) -> bytes:
    return session_id.encode("utf-8") + b"\x00" + part.encode("ascii")


def pack_frames(frame_bytes: list[bytes]) -> bytes:
    return b"".join(_struct.pack(">I", len(b)) + b for b in frame_bytes)


def unpack_frames(blob: bytes) -> list[bytes]:
    frames, i = [], 0
    while i < len(blob):
        if i + 4 > len(blob):
            raise ValueError("truncated frame length")
        (n,) = _struct.unpack(">I", blob[i: i + 4])
        if i + 4 + n > len(blob):
            raise ValueError("truncated frame")
        frames.append(blob[i + 4: i + 4 + n])
        i += 4 + n
    return frames


def seal_envelope(shared_secret: bytes, session_nonce: bytes, session_id: str, kem_ciphertext: bytes,
                  header: dict, frame_bytes: list[bytes]) -> bytes:
    key = AESGCM(derive_channel_key(shared_secret, salt=session_nonce))
    nonce_h, nonce_f = os.urandom(AEAD_NONCE_LEN), os.urandom(AEAD_NONCE_LEN)
    header_ct = key.encrypt(nonce_h, _json.dumps(header).encode("utf-8"), _aad(session_id, "header"))
    frames_ct = key.encrypt(nonce_f, pack_frames(frame_bytes), _aad(session_id, "frames"))
    return b"".join([
        ENVELOPE_MAGIC, _struct.pack(">H", len(kem_ciphertext)), kem_ciphertext,
        nonce_h, _struct.pack(">I", len(header_ct)), header_ct, nonce_f, frames_ct,
    ])


def parse_envelope_head(buf: bytes) -> tuple[bytes, bytes, bytes, int] | None:
    """(kem_ct, nonce_h, header_ct, offset of nonce_f) once the head of the
    envelope is complete, None if more bytes are needed. Raises ValueError on
    a malformed head."""
    if len(buf) < 6:
        return None
    if buf[:4] != ENVELOPE_MAGIC:
        raise ValueError("not a PQB1 envelope")
    (kem_len,) = _struct.unpack(">H", buf[4:6])
    i = 6 + kem_len
    if len(buf) < i + AEAD_NONCE_LEN + 4:
        return None
    nonce_h = bytes(buf[i: i + AEAD_NONCE_LEN])
    (hdr_len,) = _struct.unpack(">I", buf[i + AEAD_NONCE_LEN: i + AEAD_NONCE_LEN + 4])
    if hdr_len > _HEADER_LIMIT:
        raise ValueError("header too large")
    j = i + AEAD_NONCE_LEN + 4
    if len(buf) < j + hdr_len:
        return None
    return bytes(buf[6:6 + kem_len]), nonce_h, bytes(buf[j: j + hdr_len]), j + hdr_len


def open_envelope_header(shared_secret: bytes, session_nonce: bytes, session_id: str, nonce_h: bytes, header_ct: bytes) -> dict:
    key = AESGCM(derive_channel_key(shared_secret, salt=session_nonce))
    return _json.loads(key.decrypt(nonce_h, header_ct, _aad(session_id, "header")))


def open_envelope_frames(shared_secret: bytes, session_nonce: bytes, session_id: str, tail: bytes) -> list[bytes]:
    """`tail` = everything after the header: nonce_f || frames_ct."""
    if len(tail) < AEAD_NONCE_LEN + 16:
        raise ValueError("missing frames")
    key = AESGCM(derive_channel_key(shared_secret, salt=session_nonce))
    return unpack_frames(key.decrypt(tail[:AEAD_NONCE_LEN], tail[AEAD_NONCE_LEN:], _aad(session_id, "frames")))
