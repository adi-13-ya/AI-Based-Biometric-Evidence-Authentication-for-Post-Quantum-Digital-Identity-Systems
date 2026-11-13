"""FastAPI server: device enrollment, session lifecycle, ML-KEM decapsulation,
Section 7 transaction-binding enforcement, and the AI attack-detection +
fusion pipeline (KEM decap -> binding gate -> liveness -> detectors ->
decision engine).

Trust split: the server holds ML-KEM decapsulation keys, the *public*
ML-DSA keys of enrolled devices, and its own identity key, which signs
session offers only (server/identity.py). Evidence is signed on the capture
device (client_demo/device.py); there is no endpoint that signs evidence on
a client's behalf. The exact wire format and check order are specified in
docs/PROTOCOL.md.
"""
from __future__ import annotations

import base64
import hmac
import io
import logging
import time

import numpy as np
from cryptography.exceptions import InvalidTag
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from PIL import Image
from pydantic import BaseModel, Field

from binding.session_binding import check_transaction_binding, complete_binding, credential_checks
from crypto.channel import evidence_hash_of, open_envelope_frames, open_envelope_header, parse_envelope_head
from crypto.kem import kem_details
from crypto.signature import build_session_offer, sig_details
from detection.deepfake_detector import DeepfakeDetector
from detection.fusion import fuse
from detection.injection_detector import InjectionDetector
from detection.liveness_challenge import evaluate_challenge
from detection.morph_detector import MorphDetector
from detection.presentation_attack import PresentationAttackDetector
from detection.replay_detector import store as replay_store
from server.config import settings
from server.device_registry import registry as device_registry
from server.explain import explain
from server.identity import identity as server_identity
from server.session_store import Session, nonce_b64, store

logger = logging.getLogger(__name__)

app = FastAPI(title="PQ-Biometric-Auth Server", version="0.1.0")

# Process-wide detector instances (spec Section 6: pretrained-first, with an
# automatic mock-mode fallback logged per-module if a checkpoint/dependency
# is unavailable - see detection/README-models.md). Instantiated once at
# import time rather than per-request since loading a real checkpoint would
# be too slow to redo on every submission.
_pad_detector = PresentationAttackDetector(
    mode=settings.detection_mode, device=settings.detector_device, weights=settings.pad_weights
)
_morph_detector = MorphDetector(mode=settings.detection_mode, device=settings.detector_device)
_deepfake_detector = DeepfakeDetector(mode=settings.detection_mode, device=settings.detector_device)
_injection_detector = InjectionDetector(mode=settings.detection_mode, device=settings.detector_device)

_FUSION_WEIGHTS = {
    "presentation_attack": settings.weight_presentation_attack,
    "morph": settings.weight_morph,
    "deepfake": settings.weight_deepfake,
    "replay": settings.weight_replay,
    "injection": settings.weight_injection,
}
_FUSION_THRESHOLDS = {
    "presentation_attack": settings.threshold_presentation_attack,
    "morph": settings.threshold_morph,
    "deepfake": settings.threshold_deepfake,
    "replay": settings.threshold_replay,
    "injection": settings.threshold_injection,
}


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------
class DeviceEnrollRequest(BaseModel):
    sig_alg: str = Field(description="ML-DSA parameter set of the device key")
    public_key: str = Field(description="hex-encoded ML-DSA public key generated on the device")


class DeviceEnrollResponse(BaseModel):
    device_id: str
    sig_alg: str


class SessionStartRequest(BaseModel):
    device_id: str = Field(description="id returned by POST /device/enroll")


class SessionStartResponse(BaseModel):
    session_id: str
    nonce: str = Field(description="base64-encoded session nonce")
    challenge: str
    kem_alg: str
    sig_alg: str
    kem_public_key: str = Field(description="hex-encoded ML-KEM public key")
    device_id: str
    issued_at: float
    freshness_window_seconds: int
    server_sig_alg: str
    server_key_id: str
    server_signature: str = Field(description="hex signature by the server identity key over the session offer (crypto/signature.py: build_session_offer)")


class ServerIdentityResponse(BaseModel):
    sig_alg: str
    public_key: str
    key_id: str


class VerifyRequest(BaseModel):
    evidence_hash: str
    timestamp: str
    challenge_response: str = ""
    signature: str = Field(description="hex-encoded ML-DSA signature to verify")
    frames_b64: list[str] = Field(
        default_factory=list,
        description="base64-encoded JPEG/PNG frames in temporal order; required only if binding checks pass",
    )
    frame_timestamps: list[float] | None = Field(
        default=None, description="capture timestamps of the recording (client-reported; used only by the fallback heuristic)"
    )
    burst_frames: int = Field(
        default=0, ge=0, description="how many trailing frames of frames_b64 are consecutive captures (injection classifier input)"
    )


class VerifyResponse(BaseModel):
    session_id: str
    channel: str = Field(default="plaintext", description="how the payload reached the server")
    binding_accepted: bool = Field(description="True only if ALL Section 7 step-6 checks pass")
    binding_checks: dict[str, bool]
    binding_reasons: list[str]
    detection_ran: bool = Field(description="False if rejected at the binding stage, before AI detection runs")
    detection_scores: dict[str, float] | None = None
    detection_is_mock: dict[str, bool] | None = None
    detection_details: dict[str, dict] | None = Field(default=None, description="per-module diagnostic signals")
    fused_score: float | None = Field(default=None, description="threshold-normalised weighted mean, reported only (the decision is the per-module veto)")
    below_threshold_modules: list[str] | None = None
    accepted: bool = Field(description="Final ACCEPT/REJECT decision")
    reason: str
    explanations: list[str] = Field(default_factory=list, description="plain-language reasons for the decision")
    liveness: dict | None = Field(default=None, description="what the liveness-challenge check measured")
    timing_ms: dict[str, float]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _get_session_or_404(session_id: str) -> Session:
    session = store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Unknown session_id {session_id!r}")
    return session


def _decode_frame_bytes(frames_b64: list[str]) -> list[bytes]:
    try:
        return [base64.b64decode(b64, validate=True) for b64 in frames_b64]
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"frames_b64 must be base64: {exc}") from exc


def _decode_frames(frame_bytes: list[bytes]) -> list[np.ndarray]:
    frames = []
    for i, raw in enumerate(frame_bytes):
        try:
            image = Image.open(io.BytesIO(raw)).convert("RGB")
            frames.append(np.array(image))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"could not decode frames_b64[{i}]: {exc}") from exc
    return frames


def _json_safe(details: dict) -> dict:
    """Detector details as plain JSON numbers/strings (numpy scalars and
    imagehash objects aren't serializable by pydantic as-is)."""
    safe = {}
    for key, value in details.items():
        if isinstance(value, (bool, str)) or value is None:
            safe[key] = value
        elif isinstance(value, (int, float, np.integer, np.floating)):
            safe[key] = round(float(value), 4)
        elif isinstance(value, list):
            safe[key] = [round(float(v), 4) for v in value]
        else:
            safe[key] = str(value)
    return safe


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.get("/health")
def health() -> dict:
    liboqs_ready = True
    error = None
    try:
        kem_details(settings.kem_alg)
        sig_details(settings.sig_alg)
    except Exception as exc:  # pragma: no cover - environment dependent
        liboqs_ready = False
        error = str(exc)

    payload = {
        "status": "ok" if liboqs_ready else "degraded",
        "kem_alg": settings.kem_alg,
        "sig_alg": settings.sig_alg,
        "liboqs_ready": liboqs_ready,
    }
    if error:
        payload["error"] = error
    return payload


@app.get("/server/identity", response_model=ServerIdentityResponse)
def get_server_identity() -> ServerIdentityResponse:
    """The public key that signs session offers. Clients pin it (out of band,
    or trust-on-first-use in the prototype) and verify every offer with it."""
    return ServerIdentityResponse(sig_alg=server_identity.alg, public_key=server_identity.public_key.hex(), key_id=server_identity.key_id)


@app.post("/device/enroll", response_model=DeviceEnrollResponse)
def enroll_device(req: DeviceEnrollRequest, x_enrollment_token: str | None = Header(default=None)) -> DeviceEnrollResponse:
    """Register a capture device's public key. The secret key never leaves
    the device. Open (trust-on-first-use) unless ENROLLMENT_TOKEN is set, in
    which case the caller must present it - the hook where a deployment
    would require an onboarding check or hardware key attestation."""
    if settings.enrollment_token and not hmac.compare_digest(
        (x_enrollment_token or "").encode("utf-8"), settings.enrollment_token.encode("utf-8")
    ):
        raise HTTPException(status_code=403, detail="enrollment requires a valid X-Enrollment-Token")
    try:
        device = device_registry.enroll(req.sig_alg, bytes.fromhex(req.public_key))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return DeviceEnrollResponse(device_id=device.device_id, sig_alg=device.sig_alg)


@app.post("/session/start", response_model=SessionStartResponse)
def session_start(req: SessionStartRequest) -> SessionStartResponse:
    device = device_registry.get(req.device_id)
    if device is None:
        raise HTTPException(status_code=404, detail=f"Unknown device_id {req.device_id!r}; enroll first")
    session = store.create(device.device_id, device.public_key, device.sig_alg)
    offer = build_session_offer(
        session_id=session.session_id, session_nonce=session.nonce, challenge=session.challenge,
        kem_alg=settings.kem_alg, kem_public_key=session.kem.public_key, device_id=device.device_id,
        issued_at=session.created_at, freshness_window_seconds=settings.freshness_window_seconds,
    )
    return SessionStartResponse(
        session_id=session.session_id,
        nonce=nonce_b64(session.nonce),
        challenge=session.challenge,
        kem_alg=settings.kem_alg,
        sig_alg=device.sig_alg,
        kem_public_key=session.kem.public_key.hex(),
        device_id=device.device_id,
        issued_at=session.created_at,
        freshness_window_seconds=settings.freshness_window_seconds,
        server_sig_alg=server_identity.alg,
        server_key_id=server_identity.key_id,
        server_signature=server_identity.sign_offer(offer).hex(),
    )


@app.post("/session/{session_id}/verify", response_model=VerifyResponse)
def verify_evidence(session_id: str, req: VerifyRequest) -> VerifyResponse:
    """Plaintext variant of the pipeline, for unit tests only (disabled
    unless ALLOW_PLAINTEXT_VERIFY=true). No result in the report uses it;
    the evaluated format is /verify-sealed."""
    if not settings.allow_plaintext_verify:
        raise HTTPException(status_code=404, detail="plaintext /verify is disabled; use /verify-sealed")
    session = _get_session_or_404(session_id)
    return _verify_core(session_id, session, req, timing_ms={}, channel="plaintext",
                        binding_context=_binding_context(session_id, req, b""))


def _binding_context(session_id: str, req: VerifyRequest, kem_ct: bytes) -> dict:
    """The signed fields besides evidence hash, nonce, timestamp and
    challenge answer (crypto/signature.py: build_binding_payload)."""
    return {"session_id": session_id, "kem_ciphertext": kem_ct,
            "burst_frames": req.burst_frames, "frame_timestamps": req.frame_timestamps}


def _channel_reject(session_id: str, session: Session, timing_ms: dict, reason: str) -> VerifyResponse:
    session.consume_nonce()
    timing_ms["total_ms"] = sum(timing_ms.values())
    return VerifyResponse(
        session_id=session_id,
        channel=f"{settings.kem_alg} + AES-256-GCM",
        binding_accepted=False,
        binding_checks={"channel_authentic": False},
        binding_reasons=[reason],
        detection_ran=False,
        accepted=False,
        reason="rejected at channel stage: " + reason,
        explanations=explain(accepted=False, binding_checks={"channel_authentic": False}, binding_reasons=[reason]),
        timing_ms=timing_ms,
    )


def _sealed_head(session_id: str, session: Session, head: tuple) -> tuple:
    """Decapsulate, open the small signed header and - in the integrated
    pipeline - run the credential checks (nonce, freshness, signature).
    Returns (response or None, shared_secret, request, credentials, timing_ms, binding_context)."""
    kem_ct, nonce_h, header_ct, _ = head
    timing_ms: dict = {}
    t0 = time.perf_counter()
    try:
        shared_secret = session.kem.decapsulate(kem_ct)
    except Exception:
        shared_secret = None
    t1 = time.perf_counter()
    header = None
    if shared_secret is not None:
        try:
            header = open_envelope_header(shared_secret, session.nonce, session_id, nonce_h, header_ct)
        except (InvalidTag, ValueError):
            header = None
    timing_ms["kem_decapsulate_ms"] = (t1 - t0) * 1000
    timing_ms["aead_decrypt_ms"] = (time.perf_counter() - t1) * 1000
    reason = "sealed payload failed ML-KEM decapsulation / AES-256-GCM authentication (tampered or wrong session)"
    if header is None:
        return _channel_reject(session_id, session, timing_ms, reason), None, None, None, timing_ms, None
    try:
        req = VerifyRequest(**{**header, "frames_b64": []})
        evidence_hash, signature = bytes.fromhex(req.evidence_hash), bytes.fromhex(req.signature)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"invalid sealed header: {exc}") from exc
    context = _binding_context(session_id, req, kem_ct)
    if settings.pipeline_mode in ("layered", "buffered"):
        return None, shared_secret, req, None, timing_ms, context
    t2 = time.perf_counter()
    first_use = session.consume_nonce()  # atomic: of concurrent duplicates, only one sees first_use
    credentials = credential_checks(
        session_nonce=session.nonce, client_public_key=session.device_public_key, sig_alg=session.sig_alg,
        evidence_hash=evidence_hash, timestamp=req.timestamp, challenge_response=req.challenge_response,
        signature=signature, nonce_already_used=not first_use,
        freshness_window_seconds=settings.freshness_window_seconds, binding_context=context,
    )
    timing_ms["credential_check_ms"] = (time.perf_counter() - t2) * 1000
    if not credentials.accepted:
        timing_ms["total_ms"] = sum(timing_ms.values())
        return _binding_reject(session_id, session, credentials, timing_ms, f"{settings.kem_alg} + AES-256-GCM", {}), None, None, None, timing_ms, None
    return None, shared_secret, req, credentials, timing_ms, context


def _sealed_tail(session_id: str, session: Session, shared_secret: bytes, req: VerifyRequest, credentials, timing_ms: dict,
                 tail: bytes, context: dict) -> VerifyResponse:
    t0 = time.perf_counter()
    try:
        frame_bytes = open_envelope_frames(shared_secret, session.nonce, session_id, tail)
    except (InvalidTag, ValueError):
        timing_ms["aead_decrypt_ms"] += (time.perf_counter() - t0) * 1000
        return _channel_reject(session_id, session, timing_ms, "sealed frames failed AES-256-GCM authentication (tampered in transit)")
    timing_ms["aead_decrypt_ms"] += (time.perf_counter() - t0) * 1000
    return _verify_core(session_id, session, req, timing_ms=timing_ms, channel=f"{settings.kem_alg} + AES-256-GCM",
                        frame_bytes=frame_bytes, credentials=credentials, binding_context=context)


@app.post("/session/{session_id}/verify-sealed", response_model=VerifyResponse)
async def verify_evidence_sealed(session_id: str, request: Request) -> VerifyResponse:
    """Verification over the post-quantum channel, header first.

    Body: the PQB1 envelope (crypto/channel.py): ML-KEM ciphertext, then a
    small AES-256-GCM-sealed header (evidence hash, timestamp, challenge
    response, ML-DSA signature), then the sealed frames. The header is read
    and checked while the frames are still arriving. If the nonce, freshness
    or signature check fails, the server answers without decrypting,
    hashing or decoding the frames (the remaining body is drained unread so
    the client still receives the response). Tampering with any part, or a
    submission under another session's id, fails GCM authentication.

    With PIPELINE_MODE=buffered (measurement baseline) the whole body is
    received first and the frames are decrypted before any check runs; the
    checks themselves, and their order, are the same."""
    session = _get_session_or_404(session_id)
    stream = request.stream()
    buf = bytearray()
    head = None
    buffered = settings.pipeline_mode == "buffered"
    try:
        async for chunk in stream:
            buf += chunk
            if buffered:
                continue
            head = parse_envelope_head(buf)
            if head is not None:
                break
        if buffered:
            head = parse_envelope_head(buf)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"malformed envelope: {exc}") from exc
    if head is None:
        raise HTTPException(status_code=400, detail="truncated envelope")

    response, shared_secret, req, credentials, timing_ms, context = await run_in_threadpool(_sealed_head, session_id, session, head)
    if response is not None:
        async for _ in stream:  # drain without processing
            pass
        return response
    tail = bytes(buf[head[3]:])
    async for chunk in stream:
        tail += chunk
    return await run_in_threadpool(_sealed_tail, session_id, session, shared_secret, req, credentials, timing_ms, tail, context)


def _binding_reject(session_id: str, session: Session, result, timing_ms: dict, channel: str, liveness: dict) -> VerifyResponse:
    logger.info("session=%s binding_rejected reasons=%s", session_id, result.reasons)
    return VerifyResponse(
        session_id=session_id,
        channel=channel,
        binding_accepted=False,
        binding_checks=result.checks,
        binding_reasons=result.reasons,
        detection_ran=False,
        accepted=False,
        reason="rejected at transaction-binding stage: " + "; ".join(result.reasons),
        explanations=explain(
            accepted=False,
            binding_checks=result.checks,
            binding_reasons=result.reasons,
            liveness=liveness,
            challenge=session.challenge,
        ),
        liveness=_json_safe(liveness),
        timing_ms=timing_ms,
    )


def _verify_core(session_id: str, session: Session, req: VerifyRequest, timing_ms: dict, channel: str,
                 frame_bytes: list[bytes] | None = None, credentials=None, binding_context: dict | None = None) -> VerifyResponse:
    """Server-side pipeline, cheapest stage first, stopping at the first
    rejection: binding gate (nonce single-use, timestamp freshness, ML-DSA
    signature against the enrolled device key, frames hash to the signed
    evidence_hash) -> image decoding + liveness challenge (MediaPipe) -> the
    five detectors -> fusion. Image decoding and every model run only after
    the cryptographic checks pass, so forged traffic costs the server one
    signature verification (see binding/session_binding.py)."""
    try:
        evidence_hash = bytes.fromhex(req.evidence_hash)
        signature = bytes.fromhex(req.signature)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="evidence_hash and signature must be hex-encoded") from exc

    if frame_bytes is None:
        frame_bytes = _decode_frame_bytes(req.frames_b64)  # base64 only; pixels are decoded lazily below
    frames: list[np.ndarray] = []
    liveness_ms = 0.0
    liveness: dict = {}

    liveness_ok: bool | None = None
    detection: tuple | None = None

    def run_liveness(challenge: str, response: str) -> bool:
        # Decoding the images and the MediaPipe head-motion check are timed
        # as liveness_check_ms so binding_check_ms is crypto + bookkeeping only.
        nonlocal liveness_ms, frames, liveness_ok
        if liveness_ok is None:
            start = time.perf_counter()
            frames = _decode_frames(frame_bytes)
            liveness_ok, measured = evaluate_challenge(challenge, response, frames)
            liveness.update(measured)
            liveness_ms = (time.perf_counter() - start) * 1000
        return liveness_ok

    def run_detection():
        nonlocal detection
        if detection is None:
            t2 = time.perf_counter()
            primary_frame = frames[0]
            burst = req.burst_frames if 2 <= req.burst_frames <= len(frames) else 0
            spread = frames[: len(frames) - burst] or frames
            results = [
                _pad_detector.predict(primary_frame),
                _morph_detector.predict(primary_frame),
                _deepfake_detector.predict_video(spread),
                replay_store.check_and_record(primary_frame, session_id=session_id),
                _injection_detector.predict_sequence(frames[-burst:] if burst else frames, frame_timestamps=req.frame_timestamps),
            ]
            t3 = time.perf_counter()
            fused = fuse(
                results,
                weights=_FUSION_WEIGHTS,
                per_module_thresholds=_FUSION_THRESHOLDS,
            )
            detection = (results, fused, (t3 - t2) * 1000, (time.perf_counter() - t3) * 1000)
        return detection

    if settings.pipeline_mode == "layered":
        # Ablation baseline only (PIPELINE_MODE=layered): the AI layer runs on
        # every request before, and independently of, the cryptographic
        # checks - the "two separate layers" design. Used to measure what the
        # integrated ordering saves; never the default.
        run_liveness(session.challenge, req.challenge_response)
        if frames:
            run_detection()

    # The signature covers evidence_hash, so the frames actually analysed
    # must be the ones that hash describes - otherwise an attacker could get
    # a genuine capture's hash signed and then submit different frames.
    # (Without frames there is nothing to compare; that case is a 400 below.)
    extra_checks = []
    if frame_bytes:
        extra_checks.append((
            "evidence_matches_hash",
            lambda: hmac.compare_digest(evidence_hash_of(frame_bytes), evidence_hash),
            "submitted frames do not match the signed evidence_hash (evidence substituted after signing)",
        ))

    t0 = time.perf_counter()
    if credentials is not None:  # sealed path: phase 1 already ran on the header
        binding_result = complete_binding(credentials, session_challenge=session.challenge, challenge_response=req.challenge_response,
                                          challenge_check=run_liveness, extra_checks=extra_checks)
    else:
        first_use = session.consume_nonce()  # atomic, see Session.consume_nonce
        binding_result = check_transaction_binding(
            session_nonce=session.nonce,
            session_challenge=session.challenge,
            client_public_key=session.device_public_key,
            sig_alg=session.sig_alg,
            evidence_hash=evidence_hash,
            timestamp=req.timestamp,
            challenge_response=req.challenge_response,
            signature=signature,
            nonce_already_used=not first_use,
            freshness_window_seconds=settings.freshness_window_seconds,
            challenge_check=run_liveness,
            extra_checks=extra_checks,
            binding_context=binding_context,
        )
    t1 = time.perf_counter()

    # The nonce is single-use: it was consumed atomically above (or in
    # _sealed_head) regardless of outcome, so a rejected submission cannot
    # be retried against the same nonce.

    timing_ms["binding_check_ms"] = (t1 - t0) * 1000 - (0.0 if settings.pipeline_mode == "layered" else liveness_ms) + timing_ms.pop("credential_check_ms", 0.0)
    timing_ms["liveness_check_ms"] = liveness_ms
    if detection is not None and not binding_result.accepted:
        timing_ms["detection_ms"] = detection[2]  # spent on a request that is rejected anyway (layered mode)
        timing_ms["fusion_ms"] = detection[3]

    if not binding_result.accepted:
        timing_ms["total_ms"] = sum(timing_ms.values())
        return _binding_reject(session_id, session, binding_result, timing_ms, channel, liveness)

    if not frames:
        raise HTTPException(status_code=400, detail="frames_b64 must contain at least one frame once binding checks pass")

    detection_results, fusion_result, detection_ms, fusion_ms = run_detection()
    timing_ms["detection_ms"] = detection_ms
    timing_ms["fusion_ms"] = fusion_ms
    timing_ms["total_ms"] = sum(timing_ms.values())
    detection_details = {r.module: _json_safe(r.details) for r in detection_results}

    if fusion_result.accepted:
        reason = f"accepted (every detector at or above its threshold; reported mean={fusion_result.fused_score:.2f})"
    else:
        reason = (
            f"rejected by detector veto (below_threshold_modules={fusion_result.below_threshold_modules}; "
            f"reported mean={fusion_result.fused_score:.2f})"
        )
    logger.info("session=%s decision=%s fused_score=%.3f", session_id, fusion_result.accepted, fusion_result.fused_score)

    return VerifyResponse(
        session_id=session_id,
        channel=channel,
        binding_accepted=True,
        binding_checks=binding_result.checks,
        binding_reasons=[],
        detection_ran=True,
        detection_scores=fusion_result.per_module_scores,
        detection_is_mock=fusion_result.per_module_is_mock,
        detection_details=detection_details,
        fused_score=fusion_result.fused_score,
        below_threshold_modules=fusion_result.below_threshold_modules,
        accepted=fusion_result.accepted,
        reason=reason,
        explanations=explain(
            accepted=fusion_result.accepted,
            binding_checks=binding_result.checks,
            binding_reasons=[],
            detection_scores=fusion_result.per_module_scores,
            detection_details=detection_details,
            below_threshold_modules=fusion_result.below_threshold_modules,
            fused_score=fusion_result.fused_score,
        ),
        liveness=_json_safe(liveness),
        timing_ms=timing_ms,
    )


@app.post("/demo/reset-replay-store")
def reset_replay_store() -> dict:
    """Forget previously seen evidence so a demo can be rehearsed with the
    same genuine sample. Disabled unless DEMO_MODE=true: in production this
    would let anyone erase replay protection."""
    if not settings.demo_mode:
        raise HTTPException(status_code=403, detail="only available with DEMO_MODE=true")
    replay_store.clear()
    return {"replay_store_cleared": True}


@app.delete("/session/{session_id}")
def end_session(session_id: str) -> dict:
    _get_session_or_404(session_id)
    store.delete(session_id)
    return {"session_id": session_id, "closed": True}
