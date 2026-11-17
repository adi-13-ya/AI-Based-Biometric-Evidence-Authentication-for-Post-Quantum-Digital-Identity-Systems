# Changelog

All notable changes to this project are recorded here. Format loosely
follows [Keep a Changelog](https://keepachangelog.com/); dates are when the
change was made, not a formal release schedule (this is a prototype, not a
versioned package).

## [1.4.0] - paper version: final-system results only

Supervisor review, round 3.

- All system experiments re-run on the final implementation (D1/D2 latency
  and rejection cost, D3 matched-degradation flood, D5 signed attacks, D6).
- D3 flood redesigned for the final protocol: attackers hold no key and use
  observed session ids (forged requests do not consume them); genuine traffic
  from 12 devices; deployed per-device limits.
- D6 rewritten: every defence measured by an ablation flag in the same code
  base (`ABLATE_EARLY_NONCE_CONSUME`, `ENFORCE_SUBMISSION_SHAPE`,
  `STRICT_LIVENESS`, `PAD_FRAMES=1`) instead of an older source tree; new
  concurrency test (k = 2, 8, 32 copies; valid vs 16 forged), forged session
  starts, and floods by 1-8 enrolled keys.
- Measurement configuration: typed-digit challenge for recorded clips
  (`CHALLENGE_KINDS=digits`), stated in the paper.
- Figures: architecture corrected (QRNG not used for nonces; released, not
  fine-tuned, MiniFASNet; veto rule; nonce claim at g4); legends no longer
  name versions.
- Report rewritten as a paper: research questions, precise adversaries,
  informal security argument (no formal proof claimed), session life cycle,
  design rules (Table 16), claims and evidence (Table 17); course and version
  history removed.
- 155 tests.

## [1.3.0] - session safety and authenticated malicious clients

Supervisor review, second round (items i-iv).

### Fixed (security)
- **Invalid requests could end a legitimate session.** The nonce was consumed
  before the device signature was checked, and a GCM failure consumed it too,
  so anyone who knew a session id (or saw the device's envelope and sent a
  corrupted copy first) could burn the session. The nonce is now read
  (`nonce_unused`) at the head and consumed atomically only after the
  signature and the evidence hash verify (`nonce_consumed`).
- **Unauthenticated session close and start.** `DELETE /session/{id}` now needs
  `X-Session-Close-Signature` (device signature over `PQBIO-SESSION-CLOSE-v1`);
  `POST /session/start` needs the device's signature over
  `PQBIO-SESSION-REQUEST-v1` (device_id, timestamp, request nonce), fresh and
  not replayed. `REQUIRE_SIGNED_SESSION_START=false` restores the old start
  for the flood experiment's worst case.
- **Submission shape controlled by the client.** A single-frame submission
  skipped the motion challenge (response-only fallback) and the injection
  classifier; `burst_frames` chose which frames the injection classifier saw;
  frame size and count were unbounded. New stage `submission_well_formed`
  (after the evidence hash, before any decode): exactly 8 + 8 frames,
  `burst_frames = 8`, one size, at most 1920x1080; body capped at 48 MiB (413).
- **Liveness failed open.** A motion challenge with no detectable face fell
  back to "any non-empty response". It now fails closed unless
  `ALLOW_SAMPLE_UPLOADS=true` (demo rehearsals only).
- **PAD looked at frame 0 only**, so a live frame spliced in front of attack
  frames passed it. PAD now scores the first frame and two server-chosen
  frames and keeps the worst (`PAD_FRAMES=3`).
- The spoken-digit challenge is typed back (no audio), so it is not issued by
  default (`CHALLENGE_KINDS=motion`).

### Added
- Per-device limits: 3 unused sessions, 12 session starts per minute (HTTP
  429); unused sessions expire after 120 s.
- `tests/test_session_safety.py` (19 tests); 153 tests in total.
- `report/scripts/malicious_client.py` (D6): session burning and
  authenticated-client attacks against v1.2.1 and v1.3 side by side.
- Report: Section 4.6, claims restated (abstract, contributions,
  conclusions); negative cross-domain results kept as a finding.

### Changed
- `client_demo/api_client.py` signs session start and close; `end_session`
  takes the session dict.
- Measurement scripts set `CHALLENGE_KINDS=motion,digits` and lift the
  per-device limits so D1-D5 stay reproducible.

## [1.2.1] - framing and attribution after the supervisor's answers

- Report title now ends "for Biometric Evidence Screening"; the problem statement (1.2) says the system is a screening stage, and Section 6.3 maps the three parts of the problem statement onto the objectives.
- Section 5.3 states that all five configurations used the same evidence (seeded clips, identical attack submissions and flood payloads), the same machine and the same workload, and credits each benefit to the comparison that measures it (buffered vs layered, integrated vs buffered, ML-DSA-65 vs ECDSA-P256, Sections 5.1-5.2 for PQC cost).
- Pinned server verification key confirmed as sufficient for the prototype; a certificate or attestation chain remains future work.
- Previous report kept as report/archive/QUANTUM-FINAL-REPORT-REVISED_v1.1.docx.

## [1.2.0] - validly signed attacks, cluster-level intervals, related work, quantum material to an appendix

Review feedback addressed (items 5-6).

### Changed

- **Injection threshold reconciled: 0.35 -> 0.39** (`server/config.py`,
  `.env.example`). 0.39 is the BPCER = 5% point stored with the final
  classifier (`detection/weights/injection_lr.json: threshold_bpcer5`) and the
  value the report states; the configuration had been left at an earlier
  0.35. A test now fails if the two drift apart.
- `scripts/demo_check.py`: the printed-photo, deepfake and injection demo
  samples (synthetic edits of one portrait, made for the v0.2 heuristics)
  must be rejected by *a* detector rather than by the one they are named
  after. Per-attack attribution is measured on public data instead
  (`report/scripts/signed_attacks.py`).
- Report: title, abstract and conclusions centred on rejecting invalid
  submissions before inference; new Section 2.5 (signed capture, challenge
  binding, attestation, early rejection) and rebuilt Table 1; new
  Section 4.5 / Table 10 (validly signed attacks); injection claim limited to
  re-encoded or rate-converted streams; Shor, Grover and QRNG moved to
  Appendix A; figures, tables and sections renumbered.

### Added

- `report/scripts/signed_attacks.py`: public-data attacks (webcam
  presentation attacks, emulated injections, CASIA-FASD and DeepFakeFace
  stills, replays) signed by the *enrolled* device key and sent end to end;
  records the binding check or detector vetoes that reject each one
  (`report/data/v2/d5_signed_attacks.csv|json`).
- `evaluation/cluster_ci.py`: 95% cluster-bootstrap intervals with subjects
  (CASIA-FASD), videos (webcam), persons (DeepFakeFace) and source videos
  (injection) as the resampling unit (`evaluation/results/cluster_ci.json`).

### Not done (needs resources this prototype does not have)

- No real virtual-camera driver (OBS Virtual Camera, v4l2loopback) was
  tested; the injection evaluation remains an emulation on recorded footage,
  and the claim is limited accordingly.
- The morph detector is still not benchmarked (AMSL and similar datasets need
  an access request); it is labelled as such everywhere.

## [1.1.0] - protocol specification, server authentication, veto-only fusion, matched-degradation baselines

Review feedback addressed (items 1-4): one exact protocol specification;
how clients authenticate the server's ML-KEM key; enrollment, nonce
consumption and concurrent replay; screening vs identification; the fusion
rule; a buffered baseline and a classical-signature baseline under the same
check order; attacker effort at matched degradation of genuine service.

### Changed (breaking)

- **Signed evidence message v2** (`crypto/signature.py: build_binding_payload`).
  It is now domain-tagged (`PQBIO-EVIDENCE-v2`) and length-prefixed (v1 was a
  bare concatenation, so field boundaries were not signed). It also covers
  `session_id`, SHA-256 of the ML-KEM ciphertext, and `burst_frames` /
  `frame_timestamps`, which steer the injection classifier and were
  previously unsigned. Clients now encapsulate *before* signing.
  `api_client.seal_submission` takes the `(ciphertext, shared_secret)` from
  `encapsulate()`.
- **Session offers are signed by a server identity key**
  (`server/identity.py`, ML-DSA-65, `SERVER_KEY_PATH`). `POST /session/start`
  returns `server_sig_alg`, `server_key_id`, `server_signature`. The client
  verifies the signature against a pinned key (`PQBIO_SERVER_PUBKEY`, or
  trust-on-first-use via the new `GET /server/identity`) and refuses to
  encapsulate otherwise.
- **Fusion is veto-only** (`detection/fusion.py`). Accept iff every module
  scores at or above its own threshold. The threshold-normalised weighted
  mean is reported only. `FUSION_ACCEPT_THRESHOLD` and `fuse(accept_threshold=)`
  are removed. With all five thresholds set, this makes exactly the same
  decisions as v1.0's "mean >= 0.5 and no veto" (proved in the module
  docstring, tested on 5,000 random inputs), so no v1.0 result changes.
- **Plaintext `POST /session/{id}/verify` is off by default**
  (`ALLOW_PLAINTEXT_VERIFY=false`). It is a unit-test path; the evaluated
  format is PQB1 on `/verify-sealed`.

### Fixed

- **Concurrent replay.** The nonce flag was read during the checks and set
  afterwards, so two simultaneous copies of one submission could both pass
  `nonce_unused`. `Session.consume_nonce()` now marks it under a lock and
  returns True only to the first caller (`tests/test_server.py:
  test_concurrent_duplicates_consume_the_nonce_once`).

### Added

- `docs/PROTOCOL.md`: keys, signed and encrypted fields, PQB1 byte layout,
  how the header is bound to the body, when the server reads, buffers,
  decrypts and decodes, what a valid submission does and does not prove.
- `PIPELINE_MODE=buffered`: receive and decrypt the whole request, then
  the same checks in the same order, then inference (baseline). `layered`
  is kept as the inference-first ablation.
- `ECDSA-P256` as a device signature algorithm, for the classical baseline
  under the same protocol (never a default; flagged quantum-vulnerable).
- `ENROLLMENT_TOKEN`: when set, `POST /device/enroll` requires a matching
  `X-Enrollment-Token` header (stand-in for an onboarding or attestation
  step). Off by default.
- `report/scripts/distributed_experiments.py`: D1/D2 across the five
  configurations (integrated, buffered, layered x ML-DSA-65; integrated,
  buffered x ECDSA-P256). `report/scripts/flood_experiment.py` rewritten:
  the forged-request rate needed to double genuine median latency,
  rate-paced attackers, 3 repeats per configuration
  (`report/data/v2/d3_flood_matched.json`).
- 16 tests (133 total).

### Corrected in the report

- The v1.0 flood ratio was 3,306 / 41 = 80.6x, not 82.6x. It is replaced by
  the matched-degradation result, because the two v1.0 runs degraded the
  genuine user by different amounts.

## [1.0.0] - trained models, distributed trust, public-benchmark evaluation

Review feedback addressed: detectors were hand-tuned heuristics, the
injection check accepted virtual-camera streams, the replay score's linear
interpolation was wrong, and the server signed on the client's behalf.

### Changed (breaking)

- **Clients sign their own evidence.** The capture device generates an
  ML-DSA-65 keypair (`client_demo/device.py`, key file mode 0600) and
  enrolls only the public key (`POST /device/enroll`). `POST /session/start`
  now takes `{"device_id"}`. Removed `POST /session/{id}/evidence` (server-
  side signing) and `POST /session/{id}/channel`. The server stores public
  keys only (`server/device_registry.py`) and cannot sign. liboqs is now
  required on the client.
- **`/verify-sealed` takes a binary, header-first envelope** (`PQB1`,
  `crypto/channel.py`): ML-KEM ciphertext, a small AES-GCM-sealed header
  (evidence hash, timestamp, challenge response, signature), then the sealed
  frames as raw bytes. The server streams the body, checks nonce, freshness
  and signature on the header, and rejects without decrypting, hashing or
  decoding the frames. Replaces the JSON body with base64 frames, whose
  parsing (~60 ms CPU per 7 MB request) happened before any check.
- **Binding checks short-circuit in cost order** (`binding/session_binding.py`):
  nonce, timestamp, ML-DSA signature, evidence hash, then the liveness
  challenge. Frames are decoded and MediaPipe runs only after every
  cryptographic check passes; before, a forged request still paid for the
  liveness model. Checks that were not reached are absent from
  `binding_checks`.
- **Presentation attack detection uses MiniFASNet** (Silent-Face-Anti-Spoofing,
  V2 at 2.7x + V1SE at 4.0x, RetinaFace crops), fine-tuned on CelebA-Spoof
  train (`evaluation/finetune_pad.py`); `PAD_WEIGHTS=released` selects the
  original checkpoints.
- **Deepfake detection uses EfficientNet-B0** fine-tuned on DeepFakeFace
  face swaps (`evaluation/train_deepfake.py`); clips are scored by mean frame
  probability.
- **Injection detection is a trained classifier** over content features
  (repeated frames, temporal sensor noise, 8/16-px blocking, static blocks)
  instead of client-reported frame timing, which an attacker controls
  (`evaluation/train_injection.py`, weights stored as JSON). Clients send an
  8-frame consecutive burst after the spread frames (`burst_frames`).
- **Replay score is binary.** The old score linearly interpolated the
  Hamming distance to the nearest *unrelated* capture; it was non-monotonic
  (d = 8 scored 0.75, d = 9 scored 0.52) and sat just above the veto.
- **Fusion normalises each score to its own calibrated threshold** before
  averaging; `FUSION_ACCEPT_THRESHOLD` default 0.5. Per-module thresholds
  come from development data (`.env.example`).

### Added

- `evaluation/metrics.py`: per-species APCER, BPCER, EER, BPCER@APCER,
  thresholds fixed on development data, Wilson intervals.
- `evaluation/fetch_data.py` (reproducible public subsets via HTTP range
  requests), `pad_data.py`, `benchmark_pad.py`, `benchmark_deepfake.py`.
- `report/scripts/distributed_experiments.py`: two-process deployment,
  integrated vs layered pipeline (`PIPELINE_MODE`, ablation only) under a
  forged-request flood.

## [0.3.0] - live webcam fixes

### Fixed (live webcam use)

- **Every real webcam frame was rejected as a presentation attack.** PAD
  divided whole-frame sharpness by 500, a scale only a sharp still photo
  reaches; webcam frames (15-50) scored 0.35-0.46, under the veto. PAD now
  combines three recapture cues measured on the face region: true-black
  level (0.5), sharpness at a fixed face size (0.3) and saturation (0.2).
  Simulated webcam frames (dim, soft, plain wall, small face) score 1.00;
  the printed-photo sample 0.01, the screen replay 0.41.
- **A person's second genuine recording was rejected as a replay.** Two
  recordings of the same scene match on perceptual hash; replay now also
  requires the native-resolution fine detail to match (same capture >= 0.85,
  new recording <= 0.41 after 2 px of movement). Same-scene new captures
  score 0.75 and never veto.
- **Injection timing misjudged short sequences** (2% of 4-interval live
  sequences read as machine-perfect). Timing is judged only on >= 8
  intervals, with a robust MAD-based jitter; video files and recordings now
  send every frame's timestamp.
- **Client quality check** flagged normal webcam sharpness as "too blurry"
  (threshold 40 -> 10).

### Added

- **Recorded liveness challenge.** Live Capture records a 4 s clip from the
  browser camera over WebRTC (`client_demo/recorder.py`, streamlit-webrtc
  re-added), with a REC bar and on-video prompts; 8 frames plus all
  timestamps are sent. Two-photo fallback kept.
- **Explanations.** Every response carries `explanations` (plain-language
  reasons with measured values, `server/explain.py`) and `liveness` (what
  the challenge check measured); the dashboard shows them under each REJECT
  and `demo_check.py --explain` prints them.
- `detection/face_region.py` shared by PAD and morph; `GUIDE.md`
  (presenter's guide, replaces `docs/DEMO_GUIDE.md`); README restructured
  for a public audience.

### Changed

- Demo videos are 1 s / 30 frames, long enough for the timing check.
- Synthetic test frames use correlated colour channels (camera-like).
- `docker-compose.yml`: removed the obsolete `version` key.

## [0.2.0] - sealed channel and real-face fixes

### Fixed (found by testing with a real face and the real client)

- **Morph detector rejected genuine faces.** Landmark symmetry scored a real
  frontal face 0.43 (below the 0.5 veto) and fell back to whole-frame
  mirror asymmetry (score 0) whenever MediaPipe missed the face. Rewritten
  around the artifacts pixel-averaged morphs actually leave: ghost-peak
  prominence in the 2-D autocorrelation and mirror symmetry of the high-pass
  detail in the face region (`detection/morph_detector.py`).
- **Replay detector widened its own radius.** Non-matching captures scored
  (d - 8)/24, so anything closer than ~20 bits fell under the 0.5 veto and
  3/40 genuine sessions were rejected. Non-matches now score >= 0.5.
- **Injection detector rewarded synthetic timing.** Exactly regular frame
  timestamps (pre-recorded file / virtual camera) scored 0.75; they now
  score 0.
- **Frames were not tied to the signature.** The server never checked that
  the frames it analysed hash to the signed `evidence_hash`; it now does
  (`evidence_matches_hash` binding check, canonical length-prefixed hash in
  `crypto/channel.py`).
- **The ML-KEM channel key was never used.** New `POST /session/{id}/verify-sealed`:
  the client encapsulates and seals the payload with AES-256-GCM
  (HKDF(K, salt=nonce), session id as AAD); `/verify` remains for clients
  without liboqs.
- **Live capture never delivered a frame** (streamlit-webrtc callback wrote
  `st.session_state` from a worker thread). Replaced with `st.camera_input`;
  `streamlit-webrtc`/`av` removed from `requirements-client.txt`.
- **PAD sharpness depended on camera resolution**; now measured at a 640 px
  long side.
- **Benchmark "genuine" frames were white noise**, which the FFT deepfake
  heuristic correctly flags; `evaluation/benchmark.py` now uses
  natural-statistics frames.

### Added

- `scripts/make_demo_assets.py` (demo samples from one consenting face
  photo) and `scripts/demo_check.py` (12-scenario pre-demo self-check).
- Dashboard: *Tamper with Protocol* tab, per-layer result panel, local
  capture quality check, replay-memory reset; `DEMO_MODE` setting and
  `POST /demo/reset-replay-store` (403 unless enabled).
- `liveness_check_ms` timing, separate from the cryptographic
  `binding_check_ms`; per-detector `detection_details` in responses.
- `report/` (final report, figures, data, experiment scripts including the
  Qiskit circuits) and `docs/ARCHITECTURE.md`.
- Tests: sealed channel, frame substitution, timing and replay-veto
  regressions, morph artifacts, quality check, demo-mode guard (94 total).

### Changed

- Documentation moved to `docs/` (setup, demo guide, spec, study guide).

## [0.1.0] - initial build

Initial build, per `docs/PQ-Biometric-Auth-Spec.md`, in the order specified by
the build instructions. Each stage below was a separate commit; see git log
for full detail.

### Added

- **Repo scaffolding**: README.md, SETUP.md, .env.example, requirements.txt
  + requirements-client.txt (server/Docker vs. native-Windows-client split),
  docker-compose.yml, Dockerfile, .gitignore.
- **Crypto layer** (`crypto/`): ML-KEM-768 (`kem.py`) and ML-DSA-65
  (`signature.py`) wrappers over `oqs-python`/liboqs, plus an RSA-3072/
  ECDSA-P256 classical baseline (`classical_baseline.py`) for the
  PQC-vs-classical overhead comparison.
- **FastAPI server** (`server/`): session lifecycle, ML-KEM channel
  establishment, ML-DSA evidence signing/verification.
- **Session binding** (`binding/session_binding.py`): the Section 7
  transaction-binding/freshness mechanism - nonce single-use, timestamp
  freshness window, signature validity, and challenge fulfilment.
- **AI detection pipeline** (`detection/`): presentation-attack, face-morph,
  deepfake, replay, and injection detectors, each with a documented
  mock-mode fallback when a pretrained checkpoint isn't available (see
  `detection/README-models.md`), plus the weighted-average fusion/decision
  engine (`fusion.py`) and the liveness-challenge verifier
  (`liveness_challenge.py`) that binding/session_binding.py's challenge
  check delegates to.
- **Streamlit demo client** (`client_demo/`): live webcam capture mode and a
  simulate-attack mode over `demo_assets/attacks/`, with real-time
  per-detector scores, fused decision, latency breakdown, and an
  append-only attempt log.
- **Evaluation harness** (`evaluation/`): crypto/detection latency, PQC vs.
  classical overhead, process CPU/memory overhead, and (server permitting)
  throughput vs. concurrency, written as CSV + charts.
- **Documentation**: `DEMO_GUIDE.md` (numbered live-demo script),
  `demo_assets/README-assets.md`, `detection/README-models.md`,
  `datasets/download_samples.md`, this file, and `CLAUDE.md`.

### Notes on this environment

Built on a Windows machine without Docker Desktop installed and without a
working native liboqs build toolchain (confirmed: `cmake` unavailable to
liboqs-python's self-build step). Every ML-KEM/ML-DSA-dependent test is
present and correct but skips automatically in this environment rather than
being faked - see `tests/conftest.py` and `SETUP.md`. All non-PQC-dependent
tests pass. See `README.md` "Assumptions" for the full list of decisions
made where the spec didn't pin an exact choice.
