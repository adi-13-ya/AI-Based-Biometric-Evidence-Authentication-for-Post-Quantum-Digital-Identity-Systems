"""Central configuration, loaded from environment variables / .env.

Mirrors .env.example - every field here has the same default as that file so
the app runs with sane behavior even if .env is missing.
"""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- PQC algorithm selection (Section 5) ---
    kem_alg: str = "ML-KEM-768"
    sig_alg: str = "ML-DSA-65"
    enable_classical_baseline: bool = True

    # --- Server identity (signs session offers; docs/PROTOCOL.md Section 4) ---
    server_sig_alg: str = "ML-DSA-65"
    server_key_path: str = ""  # default ~/.pqbio/server_identity_<alg>.json

    # --- Enrollment (docs/PROTOCOL.md Section 5) ---
    # When set, POST /device/enroll requires header X-Enrollment-Token with
    # this value: a stand-in for the onboarding step (KYC check, hardware
    # key attestation) that a deployment would gate enrollment on. Empty =
    # open trust-on-first-use enrollment, as in the demo.
    enrollment_token: str = ""

    # --- Session binding / freshness (Section 7) ---
    freshness_window_seconds: int = 30
    nonce_bytes: int = 32

    # --- Detection thresholds (Section 6.6 fusion) ---
    threshold_presentation_attack: float = 0.57  # dev EER point, evaluation/benchmark_pad.py
    threshold_morph: float = 0.5
    threshold_deepfake: float = 0.48  # dev EER point, evaluation/benchmark_deepfake.py
    threshold_replay: float = 0.5
    threshold_injection: float = 0.39  # BPCER = 5% point of the final model (injection_lr.json: threshold_bpcer5)

    weight_presentation_attack: float = 0.25
    weight_morph: float = 0.20
    weight_deepfake: float = 0.25
    weight_replay: float = 0.15
    weight_injection: float = 0.15

    # No fused accept threshold: each module threshold above is a separate
    # veto and the weighted mean is reported only (detection/fusion.py).

    # --- Detection model mode ---
    detection_mode: str = "pretrained"  # "pretrained" | "mock"
    detector_device: str = "cpu"
    # "released" MiniFASNet checkpoints generalise better across datasets than the
    # CASIA-fine-tuned ones (evaluation/results/pad_benchmark.json)
    pad_weights: str = "released"  # "released" | "finetuned"

    # --- Server ---
    server_host: str = "0.0.0.0"
    server_port: int = 8000
    demo_mode: bool = False  # enables POST /demo/reset-replay-store for rehearsals
    # Plaintext POST /session/{id}/verify (no ML-KEM channel) is for unit
    # tests only; the evaluated format is the sealed PQB1 envelope.
    allow_plaintext_verify: bool = False
    # Check order on /verify-sealed (docs/PROTOCOL.md Section 6):
    # "integrated" (default): authenticate the header before reading the
    #     frames; reject without buffering, decrypting or decoding them.
    # "buffered": baseline - receive and decrypt the whole request, then
    #     authenticate, then run inference.
    # "layered": ablation - inference runs before the cryptographic checks.
    pipeline_mode: str = "integrated"

    # --- Replay detector store ---
    replay_hash_window_seconds: int = 86400

    # --- Client demo ---
    api_base_url: str = "http://localhost:8000"


settings = Settings()
