# Protocol specification (v1.3)

This is the exact protocol that every result in the report was measured on.
Where the code and this document disagree, the code is wrong. Source
files are named in each section.

**Scope.** The system is *biometric evidence screening*. It decides whether
a submission is authentic (it came unaltered from an enrolled device, for
this session, recently) and whether its content shows the signs of five
attack types. It does **not** match the face against a trusted identity
reference (an ID document photo or an enrolled template), so it does not
establish *who* the person is. See Section 5.

## 1. Parties and keys

| Party | Key | Algorithm | Where it lives | What it signs or decrypts |
|---|---|---|---|---|
| Capture device | device key | ML-DSA-65 (ECDSA-P256 for the classical baseline only) | `~/.pqbio/device_<alg>.json`, mode 0600, never sent | Evidence binding (3.2) |
| Server | identity key (long-term) | ML-DSA-65 | `SERVER_KEY_PATH`, mode 0600 | Session offers (3.1) only |
| Server | ephemeral KEM key, one per session | ML-KEM-768 | Memory, freed with the session | Decapsulates that session's ciphertext |

The server stores device **public** keys only (`server/device_registry.py`)
and has no endpoint that signs evidence. The identity key and the device
keys sign different messages with different domain tags, so neither
signature can stand in for the other.

## 2. Message flow

```
Device                                           Server
  | 0. POST /device/enroll {sig_alg, pk_dev}  ->   | store pk_dev under device_id = H(alg||0||pk_dev)[:16]
  |                                                |   (needs X-Enrollment-Token if ENROLLMENT_TOKEN is set)
  | 1. POST /session/start {device_id, ts_s,  ->   | verify sig_req, ts_s fresh, rn unseen; per-device
  |    rn, sig_req = Sign(sk_dev, REQ)}            |   limits; then new session_id (UUID4), nonce N (32 B),
  |                                                |   challenge c, ML-KEM-768 keypair (pk_kem, sk_kem)
  |   <- offer + sig_srv = Sign(sk_srv, OFFER)     |
  | 2. Verify sig_srv with the pinned pk_srv;      |
  |    abort if invalid                            |
  | 3. (ct, K) = Encaps(pk_kem)                    |
  | 4. capture frames F_1..F_n, answer r           |
  | 5. h = EvidenceHash(F), ts = now (ISO-8601)    |
  |    sig_dev = Sign(sk_dev, BIND)                |
  | 6. k = HKDF-SHA256(K, salt=N, info=I)          |
  |    POST /session/{id}/verify-sealed  PQB1  ->  | Section 6
  |   <- decision, failed check, explanations      |
```

`I = "pq-biometric-auth channel key v1"`. A session ends when its submission
is authenticated, after `SESSION_TTL_SECONDS` (120 s), or on
`DELETE /session/{id}` with `X-Session-Close-Signature = Sign(sk_dev, CLOSE)`. The client pins `pk_srv` from
`PQBIO_SERVER_PUBKEY` (out of band) or, in the prototype, trust-on-first-use
from `GET /server/identity`, stored in the device key file
(`client_demo/api_client.py: server_key`).

## 3. Signed messages

Both use `encode_fields(tag, f_1, ..., f_m) = tag || (u32_be(len f_i) || f_i)...`
(`crypto/signature.py`). Length prefixes make field boundaries part of
what is signed. (v1.0 signed a bare concatenation `h || N || ts || r`, in
which a byte could move between `ts` and `r` without changing the message.)

### 3.1 Session offer (server identity key)

```
OFFER = encode_fields("PQBIO-SESSION-OFFER-v1",
          session_id, N, challenge, kem_alg, pk_kem, device_id,
          repr(issued_at), str(freshness_window_seconds))
```

### 3.1a Session request and session close (device key, v1.3)

```
REQ   = encode_fields("PQBIO-SESSION-REQUEST-v1", device_id, ts_s, rn)   rn: 16 random bytes
CLOSE = encode_fields("PQBIO-SESSION-CLOSE-v1",  session_id, N)
```

Only the holder of the device key can open a session for a device (and so
spend that device's session budget, Section 6) or close one early. `rn` is
remembered for twice the freshness window, so a captured start request
cannot be replayed. Before v1.3 both endpoints were unauthenticated:
anyone who knew a session id could close it.

### 3.2 Evidence binding (device key)

```
BIND = encode_fields("PQBIO-EVIDENCE-v2",
          h, N, ts, r, session_id, SHA-256(ct), META)
META = u32 burst_frames || u32 count (0xFFFFFFFF if absent) || f64 frame_timestamps...
EvidenceHash(F) = SHA-256( for each frame: u64_be(len F_i) || F_i )
```

`F_i` are the encoded image bytes (PNG/JPEG) exactly as sent, so the server
recomputes `h` without re-encoding. The server **never takes `N` from the
request**. It rebuilds `BIND` from the nonce it stored itself and from
`session_id` and `ct` as received.

What each signed field prevents:

| Field | Without it, an attacker could |
|---|---|
| `h` | substitute frames after signing |
| `N` | replay a genuine signed submission in another session |
| `ts` | submit evidence captured earlier |
| `r` | reuse a recording made for a different challenge |
| `session_id` | (redundant with `N`; kept so the message is self-describing) |
| `SHA-256(ct)` | re-wrap a signed header under an encapsulation of its own (an interceptor that learned `K` cannot forward it with a different `ct`) |
| `META` | change the inputs of the injection classifier (`burst_frames`, `frame_timestamps`), which v1.0 did not sign |

## 4. Wire format of `/verify-sealed` (PQB1)

`crypto/channel.py`. The only submission format evaluated in the report.
(The plaintext `POST /session/{id}/verify` exists for unit tests only and
returns 404 unless `ALLOW_PLAINTEXT_VERIFY=true`.)

```
offset  field          size                 protection
0       "PQB1"         4 B                  -
4       len(ct)        u16                  -
6       ct             1,088 B (ML-KEM-768) -
        nonce_h        12 B                 -
        len(hdr_ct)    u32 (<= 64 KiB)      -
        hdr_ct         ~6.9 KB (ML-DSA-65)  AES-256-GCM(k, nonce_h, HEADER, AAD = session_id||0x00||"header")
        nonce_f        12 B                 -
        frm_ct         to end of body       AES-256-GCM(k, nonce_f, FRAMES, AAD = session_id||0x00||"frames")

HEADER = JSON {evidence_hash, timestamp, challenge_response, signature,
               burst_frames, frame_timestamps}
FRAMES = for each frame: u32_be(len F_i) || F_i
```

What is encrypted and what is signed:

| Item | Encrypted (AES-GCM) | Signed by device | Authenticated by GCM tag |
|---|---|---|---|
| `ct` | no (it is the KEM ciphertext) | yes, as SHA-256(ct) | indirectly: a wrong `ct` yields a wrong `k` |
| header fields | yes | yes, except `signature` itself | yes |
| frames | yes | yes, through `h` | yes |
| `session_id` | no (in the URL) | yes | yes (AAD) |

**How the header is bound to the body.** The two GCM messages are
independent ciphertexts under the same key `k`, with different AAD labels,
so neither can be opened as the other. The binding between them is
`h`. The header carries the signed `h`, and after decrypting the frames the
server recomputes `EvidenceHash(FRAMES)` and compares it with `h`
(constant-time) before any frame is decoded. A body from another
submission, even one that is validly sealed for this session, fails that
comparison (`evidence_matches_hash`).

## 5. What a valid submission proves

| Claim | Established by | Status |
|---|---|---|
| The ML-KEM key the client used belongs to the server | Offer signature (3.1) under the pinned key | Yes, if `pk_srv` is pinned correctly (TOFU in the prototype) |
| The bytes came from a device holding an enrolled key | Device signature (3.2) | Yes |
| They were made for this session and challenge, within the window | `N`, `r`, `ts` inside the signature | Yes |
| The frames are the ones that were signed | `h` recomputed by the server | Yes |
| The device is a real camera, or the frames came from its sensor | Hardware attestation of a key in a TEE or secure element, or a signed camera pipeline | **No.** A device key held in software signs whatever it is given, including a video file. Covered only partly, and statistically, by the liveness challenge and the injection classifier. |
| The enrolled device belongs to the claimed person | Onboarding (KYC check) at enrollment | **No.** Enrollment is trust-on-first-use, or a shared token (`ENROLLMENT_TOKEN`) standing in for the onboarding step. |
| The face is the claimed person's | Matching against a trusted reference | **No.** Not implemented. |
| The face shows no presentation, morph, deepfake, replay or injection artefacts | The five detectors (Section 7) | Statistically, at the measured error rates. The morph detector is not benchmarked. |

Hence *screening*. A deployment adds attestation at enrollment and an
identity match, and keeps everything above unchanged.

**Enrollment.** `device_id` is derived from the key, so re-enrolling a key
is idempotent and an id cannot be pointed at another key. Device ids are not
secret, which is why session start is signed (3.1a). The flood experiment of
the report (D3) disables that requirement (`REQUIRE_SIGNED_SESSION_START=false`)
to model the worst case of an attacker who already holds session ids.

## 6. Server processing order (`server/main.py: verify_evidence_sealed`)

Integrated (default, `PIPELINE_MODE=integrated`):

| Step | What the server does with the body | Check (stops at first failure) |
|---|---|---|
| 1 | Reads chunks into a buffer **only until the head is complete** (4 + 2 + 1,088 + 12 + 4 + ~6.9 KB). The frames have not been read. | malformed head -> HTTP 400 |
| 2 | ML-KEM decapsulation, HKDF, GCM-open of the header | `channel_authentic` |
| 3 | **Reads** the nonce state; does not change it | `nonce_unused` |
| 4 | Parses `ts` | `timestamp_fresh` (30 s) |
| 5 | Rebuilds `BIND`, verifies `sig_dev` with the enrolled key | `signature_valid` |
| 6 | **If any of 2-5 failed:** answers immediately. The rest of the body is drained (up to `MAX_BODY_BYTES`) and discarded: it is not stored, decrypted, hashed or decoded. | - |
| 7 | Otherwise reads the rest of the body (HTTP 413 above `MAX_BODY_BYTES`, 48 MiB), GCM-opens the frames | `channel_authentic` |
| 8 | Recomputes `h` over the decrypted frame bytes | `evidence_matches_hash` |
| 9 | Reads only the image headers: exactly `SPREAD_FRAMES + BURST_FRAMES` (8 + 8) PNG/JPEG frames of one size, `burst_frames = 8`, at most `MAX_FRAME_PIXELS` (1920x1080) | `submission_well_formed` |
| 10 | **Atomically consumes the nonce** (`Session.consume_nonce`). The request is now authenticated: valid device signature, frames equal to the signed ones. | `nonce_consumed` |
| 11 | **First pixel decode**, then the MediaPipe head-motion check, failing closed if motion cannot be measured | `challenge_fulfilled` |
| 12 | Five detectors (PAD on the first frame and 2 server-chosen frames, worst score), then the per-module veto (Section 7) | detector veto |

The checks never change; only the point at which the body is read moves.
The two baselines in the report:

- **Buffered** (`PIPELINE_MODE=buffered`). Steps 1 and 7 are merged: the
  whole body is received and kept, header and frames are decrypted, then
  steps 3-5, 8, 9 and 10 run in the same order. It authenticates before any
  inference, but only after buffering and decrypting the whole request.
- **Layered** (`PIPELINE_MODE=layered`, ablation). Whole body and decryption
  as in buffered, then decode, liveness and the five detectors, and only
  then the cryptographic checks. This is "two independent layers".

**Nonce consumption and concurrency (v1.3).** The nonce is consumed only
at step 10, after the device signature and the evidence hash have been
verified. A request that fails at any earlier step (GCM failure, stale
timestamp, bad signature, substituted or truncated frames, malformed shape)
reads the nonce but never consumes it, so it cannot invalidate the session
of the device that owns it; nor can an unsigned close (3.1a). Up to v1.2
the nonce was consumed at step 3, before the signature, so anyone who knew
a session id, or saw the device's envelope on the path and sent a corrupted
copy first, could end the session (measured in the report, Section 4.6).
Consumption is atomic (`Session.consume_nonce`, per-session lock): of any
number of concurrent valid copies, exactly one passes `nonce_consumed`
(`tests/test_session_safety.py`). A submission that passes step 10 consumes
the nonce whatever the liveness check and the detectors decide, so an
authenticated submission cannot be retried against the same challenge.
Sessions and nonces are in-process memory. A multi-replica deployment needs
an atomic shared store (for example Redis `SET NX`).

**Per-device limits.** A device may have at most
`MAX_ACTIVE_SESSIONS_PER_DEVICE` (3) unused sessions and start at most
`MAX_SESSIONS_PER_MINUTE_PER_DEVICE` (12) per minute (HTTP 429 beyond).
Each session admits one inference, so this bounds the model work any one
enrolled key, including a stolen or malicious one, can cause.

**Not covered.** The body still crosses the network after an early reject,
so bandwidth is not saved, only server CPU and memory. Rate limits are per
device: an attacker with many enrolled keys is limited only by enrollment
(`ENROLLMENT_TOKEN`, or attestation in a deployment).

## 7. Decision rule (`detection/fusion.py`)

Each detector `i` returns `s_i` in [0, 1] (1 = genuine) and has a threshold
`t_i` set on development data (`.env.example`). The submission is
**accepted iff `s_i >= t_i` for every `i`**. Each threshold is a separate
veto. The threshold-normalised weighted mean
`sum_i w_i * norm(s_i, t_i)` (where `norm` maps `t_i` to 0.5) is
**reported only**. It never accepts or rejects anything.

v1.0 decided "normalised mean >= 0.5 **and** no veto". When every
threshold is set, as the server always does, no veto implies every
normalised score >= 0.5, so the mean is >= 0.5. The mean condition was
therefore never decisive, and v1.0's decisions equal veto-only on every
input (`tests/test_fusion.py: test_veto_only_matches_v1_rule_when_all_thresholds_set`).
No v1.0 result had to be rerun for the fusion change.
