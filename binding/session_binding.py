"""Transaction-binding / freshness mechanism - spec Section 7.

This is the integration point between the crypto layer (Section 5) and the
AI detection pipeline (Section 6): it's what prevents replay of even a
*genuine* previously-captured biometric sample, by making each session's
required evidence unique (a fresh nonce + a fresh liveness challenge).

Section 7 step 6 requires these checks before evidence proceeds to AI
detection:
  (b) session_nonce hasn't been used before
  (c) timestamp is within an acceptable freshness window
  (a) ML-DSA signature validity, over a payload rebuilt from the nonce the
      server itself issued
  (e) caller-supplied integrity checks, e.g. that the submitted frames hash
      to the signed evidence_hash (server/main.py)
  (d) the liveness challenge was fulfilled - the only check that runs a
      model (MediaPipe head motion)

They run in that order - cheapest first - and stop at the first failure.
The order is the point of the protocol: a forged, stale, replayed or
substituted submission is rejected after O(1) bookkeeping or one ML-DSA
verification (tens of microseconds) and never reaches the liveness model or
the detectors. Checks that were not reached because an earlier one failed
are absent from `checks`, not reported as passed. (An earlier version ran
all four unconditionally, so a request with a bad signature still paid for
the MediaPipe liveness pass.)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

from crypto.signature import build_binding_payload, verify as sig_verify

DEFAULT_FRESHNESS_WINDOW_SECONDS = 30

ChallengeCheck = Callable[[str, str], bool]


@dataclass
class BindingCheckResult:
    accepted: bool
    checks: dict[str, bool] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)

    def _fail(self, check_name: str, ok: bool, reason: str) -> None:
        self.checks[check_name] = ok
        if not ok:
            self.accepted = False
            self.reasons.append(reason)


def placeholder_challenge_check(challenge: str, challenge_response: str) -> bool:
    """Stand-in for AI-verified challenge fulfilment (Section 7 step 6d).

    Only checks that *some* response was actually submitted - this is a
    bookkeeping placeholder, not a liveness check. Replace by passing a real
    `challenge_check` (e.g. from detection/fusion.py, once it exists) into
    `check_transaction_binding`.
    """
    return bool(challenge_response and challenge_response.strip())


def parse_timestamp(timestamp: str) -> datetime:
    """Parse an ISO-8601 timestamp (accepting a trailing 'Z') into an
    aware UTC datetime. Raises ValueError on anything else."""
    normalized = timestamp[:-1] + "+00:00" if timestamp.endswith("Z") else timestamp
    dt = datetime.fromisoformat(normalized)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


ExtraCheck = tuple[str, Callable[[], bool], str]  # (name, check, reason if it fails)


def credential_checks(
    *,
    session_nonce: bytes,
    client_public_key: bytes,
    sig_alg: str,
    evidence_hash: bytes,
    timestamp: str,
    challenge_response: str,
    signature: bytes,
    nonce_already_used: bool,
    freshness_window_seconds: int = DEFAULT_FRESHNESS_WINDOW_SECONDS,
    now: datetime | None = None,
) -> BindingCheckResult:
    """Phase 1 of the gate: nonce unused, timestamp fresh, ML-DSA signature
    valid. Needs only the small signed header, not the evidence itself, so
    the sealed endpoint runs it before reading the frames.

    The payload is always rebuilt server-side from `session_nonce` (the
    value the server itself issued and stored) rather than trusting any
    nonce the caller might send - this is what makes "matches the one it
    issued" structural rather than a comparison that could be spoofed.
    Whether the nonce was already used is the caller's (server/main.py's)
    bookkeeping.
    """
    now = now or datetime.now(timezone.utc)

    def fresh_enough() -> tuple[bool, str]:
        try:
            delta_seconds = abs((now - parse_timestamp(timestamp)).total_seconds())
        except (ValueError, TypeError):
            return False, "timestamp could not be parsed as ISO-8601"
        return (
            delta_seconds <= freshness_window_seconds,
            f"timestamp is {delta_seconds:.1f}s old, outside the {freshness_window_seconds}s freshness window",
        )

    def signature_ok() -> bool:
        payload = build_binding_payload(evidence_hash, session_nonce, timestamp, challenge_response)
        try:
            return bool(sig_verify(payload, signature, client_public_key, alg=sig_alg))
        except Exception:
            return False

    return _run(BindingCheckResult(accepted=True), [
        ("nonce_unused", lambda: (not nonce_already_used, "session nonce has already been used for a prior submission (possible replay)")),
        ("timestamp_fresh", fresh_enough),
        ("signature_valid", lambda: (signature_ok(), "ML-DSA signature verification failed")),
    ])


def complete_binding(
    result: BindingCheckResult,
    *,
    session_challenge: str,
    challenge_response: str,
    challenge_check: ChallengeCheck = placeholder_challenge_check,
    extra_checks: list[ExtraCheck] | None = None,
) -> BindingCheckResult:
    """Phase 2: caller-supplied evidence checks (e.g. the frames hash to the
    signed evidence_hash), then the liveness challenge. Does nothing if
    phase 1 already failed."""
    if not result.accepted:
        return result
    stages = [(name, lambda check=check, reason=reason: (bool(check()), reason)) for name, check, reason in extra_checks or []]
    stages.append((
        "challenge_fulfilled",
        lambda: (bool(challenge_check(session_challenge, challenge_response)), "liveness challenge was not fulfilled"),
    ))
    return _run(result, stages)


def _run(result: BindingCheckResult, stages) -> BindingCheckResult:
    for name, stage in stages:
        ok, reason = stage()
        result._fail(name, ok, reason)
        if not ok:
            break
    return result


def check_transaction_binding(
    *,
    session_nonce: bytes,
    session_challenge: str,
    client_public_key: bytes,
    sig_alg: str,
    evidence_hash: bytes,
    timestamp: str,
    challenge_response: str,
    signature: bytes,
    nonce_already_used: bool,
    freshness_window_seconds: int = DEFAULT_FRESHNESS_WINDOW_SECONDS,
    now: datetime | None = None,
    challenge_check: ChallengeCheck = placeholder_challenge_check,
    extra_checks: list[ExtraCheck] | None = None,
) -> BindingCheckResult:
    """Both phases in one call, cheapest check first, stopping at the first
    failure (see module docstring)."""
    result = credential_checks(
        session_nonce=session_nonce, client_public_key=client_public_key, sig_alg=sig_alg,
        evidence_hash=evidence_hash, timestamp=timestamp, challenge_response=challenge_response,
        signature=signature, nonce_already_used=nonce_already_used,
        freshness_window_seconds=freshness_window_seconds, now=now,
    )
    return complete_binding(result, session_challenge=session_challenge, challenge_response=challenge_response,
                            challenge_check=challenge_check, extra_checks=extra_checks)
