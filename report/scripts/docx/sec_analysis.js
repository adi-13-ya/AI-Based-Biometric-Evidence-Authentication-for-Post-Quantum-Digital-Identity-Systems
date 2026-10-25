// Section 6 (analysis) and Conclusions.
const { AlignmentType } = require("docx");
const { P, H1, H2, bullets, table, tableCaption } = require("./lib");

const L = AlignmentType.LEFT;

module.exports = (N) => {
  const rel = (d) => N[`pad_released_${d}`];
  const ft = (d) => N[`pad_finetuned_${d}`];
  const leg = (d) => N[`pad_legacy_${d}`];
  const dfE = N.df_efficientnet_b0;
  const injC = N.inj_classifier, injA = N.inj_legacy_adaptive_attacker;
  const I = N.sys_integrated, Ly = N.sys_layered;
  const FI0 = N.flood_integrated_off, FI1 = N.flood_integrated_on, FL1 = N.flood_layered_on;

  return [
    H1("6. Analysis of the Results"),
    H2("6.1 Novelty and insights"),
    P(`**Binding the evidence, not just the channel, and ordering the checks by cost.** The protocol's distinctive property is that the cryptographic layer authenticates the same bytes the detectors will judge, and judges them first. Section 5.3 measures what that buys: a forged submission costs ${I.forged_signature.server} ms of server time instead of ${Ly.forged_signature.server} ms, and under a flood the server absorbs about ${N.flood_absorb_ratio} times as many forged submissions for the same slowdown of genuine users. A layered design cannot get this benefit without moving the signature check in front of the models, at which point it has become the integrated design.`),
    P("**Early rejection has to include parsing.** Our first integrated implementation already ran the signature check before the models, and its handler time for a forged request was about 2 ms. The flood experiment showed that this number was misleading. Parsing a 10 MB JSON body and base64-decoding 16 frames took about 60 ms of CPU before any check ran, and under load that work, not the models, saturated the server. Moving to a binary, header-first envelope, in which the signed header is authenticated before the frames are read, is what made the early rejection real. We report this because a measurement of the handler alone would have overstated the defence."),
    P(`**Client-reported signals cannot be trusted by an injection detector.** The v0.2 timing check was perfect against a naive attacker and accepted ${injA.inj_25fps_crf23} of injected clips once the attacker forged jitter. Anything the client computes, including timestamps, device metadata and quality scores, is under the control of an injection attacker by definition. The classifier therefore uses only the pixels, and frame repeats from rate conversion turned out to be the most reliable cue. The burst of consecutive frames costs the client nothing and is covered by the same signature, so it cannot be swapped for a better-looking burst after signing.`),
    P(`**Fine-tuning on a benchmark can make a deployed system worse.** Fine-tuning MiniFASNet on CASIA-FASD cut the intra-dataset EER from ${rel("casia_test").eer} to ${ft("casia_test").eer} and lowered AUC on both cross-dataset tests (CelebA-Spoof ${rel("celeba_test").auc} to ${ft("celeba_test").auc}, webcam ${rel("webcam").auc} to ${ft("webcam").auc}). A single intra-dataset number would have pointed to the wrong model. Choosing the deployed weights by cross-dataset behaviour, and fixing thresholds on development data, is the reason the error rates in Section 4 can be taken at face value.`),
    P("**The cost of post-quantum security is bytes, not milliseconds.** ML-KEM-768 is faster than RSA-3072 and comparable to X25519, and all post-quantum work on the server is a few milliseconds against hundreds of milliseconds of model inference. For a camera-based flow the extra kilobytes do not matter; for constrained transports such as NFC or QR-coded credentials they would decide the parameter set."),
    H2("6.2 Logic errors found and fixed"),
    P("The review asked us to check the logic, in particular the linear interpolation in the scoring. Table 15 lists every error we found, how it showed up, and the fix. Each fix is covered by a unit test in tests/."),
    tableCaption("**Table 15.** Logic errors in v0.2 and their fixes."),
    table(
      ["Where", "Error", "Effect", "Fix"],
      [
        ["Replay score", "Hamming distance to an unrelated capture linearly interpolated into a score; second version non-monotonic", "Genuine users rejected (3/40); fragile to any threshold change", "Binary score: 0 for a confirmed replay, else 1"],
        ["Fusion", "Raw scores averaged though thresholds differ per module; fused threshold 0.6 inconsistent with the 0.5 veto", "Fused check could reject what every module passed", "Threshold-normalised scores; fused threshold 0.5 equivalent to the veto"],
        ["Thresholds", "Set by hand on one portrait and simulated frames", "Near-chance accuracy on real data (Section 4)", "Chosen on development data, frozen for test"],
        ["Binding order", "All checks ran unconditionally; liveness model ran even after a failed signature", "Forged requests paid for MediaPipe", "Cost-ordered gate that stops at the first failure"],
        ["Frame handling", "Frames decoded before any check; JSON + base64 body parsed before the gate", "~60 ms of CPU per forged request before rejection", "Header-first binary envelope; frames read only after the signature"],
        ["Trust model", "Server held the per-session client key and signed on request", "Signature proved nothing about origin", "Device-held keys, public-key enrollment, no signing endpoint"],
        ["Injection", "Decision driven by client-reported timestamps", "100% of injected clips accepted with forged jitter", "Content-only features of a consecutive-frame burst"],
        ["Deepfake heuristic", "FFT ratio ramp saturates at 1.0 on face crops", "Accepts every fake (AUC 0.50)", "Trained EfficientNet-B0"],
      ],
      [1300, 2700, 2426, 2600],
      { size: 17, align: [L, L, L, L] },
    ),
    H2("6.3 Objectives and outcomes"),
    tableCaption("**Table 16.** Objectives (Section 1.3) against measured outcomes."),
    table(
      ["Objective", "Outcome", "Met"],
      [
        ["O1 Post-quantum primitives only", `ML-KEM-768 + ML-DSA-65 + AES-256-GCM; server-side PQC ${N.pqc_server_ms} ms (${N.pqc_share} of server time)`, "Yes"],
        ["O2 Cost-ordered gate, measured", `Forged request ${I.forged_signature.server} ms vs ${Ly.forged_signature.server} ms layered; ${N.flood_absorb_ratio}× more forged requests absorbed under a flood`, "Yes"],
        ["O3 Distributed trust", "Device-held ML-DSA keys; server stores public keys and cannot sign; two-process measurements", "Yes (enrollment is trust-on-first-use)"],
        ["O4 Trained models, ISO metrics", `PAD worst-species APCER ${rel("casia_test").apcer_worst} at BPCER ${rel("casia_test").bpcer} on CASIA-FASD; cross-dataset AUC ${rel("celeba_test").auc} / ${rel("webcam").auc}; deepfake AUC ${dfE.auc_gen.insight} on the trained generator; injection AUC ${injC.auc}`, "Partly: models generalise poorly across domains"],
        ["O5 Logic errors fixed", "Eight errors fixed and tested (Table 15)", "Yes"],
        ["O6 Quantum threat simulated", "Shor 0.499, Grover 0.960 ideal; noise sweep to 5%", "Yes"],
      ],
      [2200, 5226, 1600],
      { size: 17, align: [L, L, L] },
    ),
    H2("6.4 Limitations and threats to validity"),
    ...bullets([
      `**Cross-domain accuracy.** MiniFASNet accepts ${rel("celeba_test").apcer_all} of CelebA-Spoof attacks and ${rel("webcam").apcer_all} of webcam attack videos at the frozen threshold, and the deepfake model does not generalise to unseen diffusion generators. These detectors are not sufficient on their own for unconstrained webcams.`,
      `**Sample sizes.** The webcam set has 10 bona fide videos, so its BPCER interval is wide (${rel("webcam").bpcer_ci}). CelebA-Spoof was evaluated on 2 of 65 test shards and DFF on ${N.df_datasets.test.images} test images, because the network available during this work limited downloads; the subsets are reproducible with evaluation/fetch_data.py.`,
      "**Injection benchmark is an emulation.** Attack clips were produced by re-encoding and frame-rate conversion of real webcam footage, not captured from a real virtual-camera driver. A lossless file at the camera's frame rate is not detected.",
      "**DFF shortcut risk.** Real and fake images in DFF differ in resolution and compression history; cropping, resizing and JPEG augmentation reduce but do not exclude a shortcut.",
      "**Morph detection** is the one remaining signal-processing module and is not benchmarked, because the public morph datasets require an access request we could not complete in time.",
      "**Deployment assumptions.** Enrollment is trust-on-first-use, the server is a single in-memory process, and all measurements ran on one machine over loopback, so network latency is not included.",
      "**Quantum results are simulations** on a parametric noise model; the QRNG is evaluated but not wired in as the nonce source.",
    ]),
    H1("Conclusions"),
    P("This work set out to make remote face verification secure against a quantum adversary and against forged evidence, and to do so as one protocol rather than two layers. The device signs the evidence hash, the server's nonce, a timestamp and the challenge answer with a key it alone holds; the server authenticates that header before it reads any frames, and only then runs the liveness check and the detectors. In response to the review, the heuristic detectors were replaced by trained models evaluated on public data with ISO/IEC 30107-3 metrics, the scoring logic was corrected, the injection check was rebuilt so that it no longer trusts client-reported timing, and signing was moved off the server."),
    P(`The measurements support the central claim. Post-quantum cryptography costs a few milliseconds of server time, ML-KEM-768 is faster than the RSA-3072 it replaces, and the cost-ordered, header-first gate rejects forged submissions for ${I.forged_signature.server} ms instead of ${Ly.forged_signature.server} ms, so an attacker must send about ${N.flood_absorb_ratio} times more forged requests to slow honest users as much as a layered design allows. The detectors improve on every heuristic they replace: MiniFASNet reaches an EER of ${rel("casia_test").eer} on CASIA-FASD where the heuristic had ${leg("casia_test").eer}, and the injection classifier holds its accuracy against an attacker who forges timing. The results are equally clear about what is not solved. PAD accuracy drops on other cameras, and deepfake detection does not transfer to unseen generators. In a deployment, the cryptographic checks, the liveness challenge and the detectors must therefore all be required; none of them is enough alone.`),
    P("The next steps follow from these limits: training PAD and deepfake models on more diverse data, including webcam captures and several generator families; testing injection detection with real virtual-camera drivers; tying enrollment to hardware key attestation and adding per-device rate limits; drawing session nonces from a QRNG with HKDF conditioning; adopting hybrid X25519 + ML-KEM key exchange for the transition period [30]; and running the Shor and Grover circuits on real quantum hardware."),
  ];
};
