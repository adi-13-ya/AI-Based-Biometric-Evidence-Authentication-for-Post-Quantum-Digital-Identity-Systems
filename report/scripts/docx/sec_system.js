// Section 5: System evaluation - latency, early rejection under attack, PQC cost, quantum simulations.
const fs = require("fs");
const { AlignmentType } = require("docx");
const { P, H1, H2, table, tableCaption, figure } = require("./lib");

const C = AlignmentType.CENTER;
const L = AlignmentType.LEFT;
const f3 = (x) => { const v = Number(x); return v < 1 ? v.toFixed(3) : v < 10 ? v.toFixed(2) : v < 100 ? v.toFixed(1) : v.toFixed(0); };

function csv(path) {
  const [head, ...rows] = fs.readFileSync(path, "utf8").trim().split(/\r?\n/);
  const keys = head.split(",");
  return rows.map((r) => {
    const cells = r.match(/("[^"]*"|[^,]*)(,|$)/g).map((c) => c.replace(/,$/, "").replace(/^"|"$/g, ""));
    return Object.fromEntries(keys.map((k, i) => [k, cells[i]]));
  });
}

module.exports = (N, FIG, DATA) => {
  const prim = Object.fromEntries(N.primitives.map((r) => [r.scheme, r]));
  const kem = prim["ML-KEM-768"], rsa = prim["RSA-3072 (OAEP)"], x255 = prim["X25519 (ECDH)"], dsa = prim["ML-DSA-65"], ecd = prim["ECDSA-P256"];
  const kemRT = Number(kem.op1_ms) + Number(kem.op2_ms);
  const rsaRT = Number(rsa.op1_ms) + Number(rsa.op2_ms);
  const dsaRT = Number(dsa.op1_ms) + Number(dsa.op2_ms);
  const ecdRT = Number(ecd.op1_ms) + Number(ecd.op2_ms);
  const pqBytes = Number(kem.public_key_B) + Number(kem.ct_or_sig_B) + Number(dsa.ct_or_sig_B);
  const clBytes = Number(rsa.public_key_B) + Number(rsa.ct_or_sig_B) + Number(ecd.ct_or_sig_B);
  const E = N.e2e;
  const I = N.sys_integrated, Ly = N.sys_layered;
  const FI0 = N.flood_integrated_off, FI1 = N.flood_integrated_on, FL0 = N.flood_layered_off, FL1 = N.flood_layered_on;
  const mods = Object.fromEntries(N.modules.map((m) => [m.module, m]));
  const sweep = csv(`${DATA}/quantum_noise_sweep.csv`);
  const kinds = [["forged_signature", "Forged signature (attacker's key)"], ["reused_nonce", "Reused nonce"], ["stale_timestamp", "Stale timestamp (120 s)"], ["substituted_frames", "Substituted frames"]];

  return [
    H1("5. System Evaluation: Latency, Early Rejection and Cost"),
    P("All measurements in this section come from the two-process deployment described in Section 3.3: the server as a uvicorn process with a single worker, and the client, or clients, as separate processes with their own ML-DSA keys, connected over HTTP on localhost. Genuine submissions use real webcam clips from the bona fide videos of the webcam sample, each with 8 spread frames and an 8-frame burst at 640 × 480, encoded as PNG (scripts: report/scripts/distributed_experiments.py and flood_experiment.py)."),
    H2("5.1 Post-quantum versus classical primitives"),
    tableCaption("**Table 10.** Primitive latency (median of 500 runs, ms) and sizes (bytes)."),
    table(
      ["Scheme", "Quantum-safe", "KeyGen", "Encaps / Sign", "Decaps / Verify", "Public key", "Ciphertext / signature"],
      ["ML-KEM-512", "ML-KEM-768", "ML-KEM-1024", "RSA-3072 (OAEP)", "X25519 (ECDH)", "ML-DSA-44", "ML-DSA-65", "ML-DSA-87", "ECDSA-P256"].map((k) => {
        const r = prim[k];
        return [k, r.quantum_safe, f3(r.keygen_ms), f3(r.op1_ms), r.op2_ms ? f3(r.op2_ms) : "—", r.public_key_B, r.ct_or_sig_B];
      }),
      [1826, 1300, 1000, 1300, 1300, 1100, 1200],
      { size: 17, align: [L, C, C, C, C, C, C] },
    ),
    P(`ML-KEM-768 encapsulation plus decapsulation takes ${f3(kemRT)} ms, ${(rsaRT / kemRT).toFixed(0)} times faster than RSA-3072 key transport (${f3(rsaRT)} ms, dominated by private-key decryption) and comparable to X25519 (${f3(x255.op1_ms)} ms). ML-DSA-65 signing plus verification takes ${f3(dsaRT)} ms against ${f3(ecdRT)} ms for ECDSA-P256. The cost of post-quantum security is size, not time: an ML-KEM-768 public key, ciphertext and ML-DSA-65 signature total ${pqBytes.toLocaleString("en-US")} bytes against ${clBytes.toLocaleString("en-US")} bytes for RSA-3072 plus ECDSA, which is negligible next to a ${E.payload_bytes} evidence payload. Raising the security level from 3 to 5 adds well under 0.1 ms per operation.`),
    ...figure(`${FIG}/fig_r1_pqc_vs_classical.png`, "**Figure 9.** Critical-path latency (left) and bytes on the wire (right, log scale) for post-quantum (blue) and quantum-vulnerable classical (grey) primitives.", 0.95),
    H2("5.2 End-to-end latency in the distributed deployment"),
    tableCaption(`**Table 11.** Median stage latency over ${N.n_genuine} genuine sealed sessions (ms).`),
    table(
      ["Where", "Stage", "Median (ms)"],
      [
        ["Device", "Session start (HTTP; server ML-KEM key generation)", E.session_start_ms],
        ["Device", "Encode 16 frames as PNG", E.encode_frames_ms],
        ["Device", "SHA-256 evidence hash", E.hash_ms],
        ["Device", "ML-DSA-65 sign", E.sign_ms],
        ["Device", "ML-KEM-768 encapsulate + AES-256-GCM seal", E.encaps_seal_ms],
        ["Server", "ML-KEM-768 decapsulate", E.server_kem_decapsulate_ms],
        ["Server", "AES-256-GCM open (header and frames)", E.server_aead_decrypt_ms],
        ["Server", "Binding gate: nonce, freshness, ML-DSA verify, evidence hash", E.server_binding_check_ms],
        ["Server", "Decode frames + liveness challenge (MediaPipe)", E.server_liveness_check_ms],
        ["Server", "Five detectors", E.server_detection_ms],
        ["Server", "Fusion", E.server_fusion_ms],
        ["Server", "Total server processing", E.server_total_ms],
        ["Both", "Submit to response, as seen by the device (HTTP)", E.verify_http_ms],
      ],
      [1200, 5826, 2000],
      { size: 18, align: [L, L, C] },
    ),
    P(`On the server, all post-quantum work (decapsulation, AES-GCM and the binding gate including ML-DSA verification) takes ${N.pqc_server_ms} ms, ${N.pqc_share} of server processing; on the device, hashing, signing and sealing take ${N.client_crypto_ms} ms. The time goes to the models: ${E.server_detection_ms} ms for the five detectors and ${E.server_liveness_check_ms} ms for decoding and the MediaPipe liveness check. Table 12 breaks the detector cost down. The largest single client cost is PNG encoding (${E.encode_frames_ms} ms), which also makes the payload large (${E.payload_bytes}); lossless frames are kept because the injection classifier must see the camera's own compression, not one added by the client. ${I.accept_rate} of genuine sessions were accepted; the rejected clips were vetoed by PAD or the deepfake model, in line with the cross-dataset error rates of Section 4.`),
    tableCaption("**Table 12.** Per-module inference cost on CPU (single request, in-process)."),
    table(
      ["Module", "Model", "Weights (MB)", "Median (ms)", "p95 (ms)"],
      [
        ["Presentation attack", mods.presentation_attack.model, mods.presentation_attack.size_mb, f3(mods.presentation_attack.median_ms), f3(mods.presentation_attack.p95_ms)],
        ["Deepfake (8 frames)", mods["deepfake (8 frames)"].model, mods["deepfake (8 frames)"].size_mb, f3(mods["deepfake (8 frames)"].median_ms), f3(mods["deepfake (8 frames)"].p95_ms)],
        ["Injection (8-frame burst)", mods["injection (8-frame burst)"].model, "< 0.01", f3(mods["injection (8-frame burst)"].median_ms), f3(mods["injection (8-frame burst)"].p95_ms)],
        ["Morph", mods.morph.model, "—", f3(mods.morph.median_ms), f3(mods.morph.p95_ms)],
        ["Replay", mods.replay.model, "—", f3(mods.replay.median_ms), f3(mods.replay.p95_ms)],
        ["Liveness challenge", mods["liveness (MediaPipe)"].model, "—", f3(mods["liveness (MediaPipe)"].median_ms), f3(mods["liveness (MediaPipe)"].p95_ms)],
      ],
      [2026, 3000, 1300, 1350, 1350],
      { size: 18, align: [L, L, C, C, C] },
    ),
    ...figure(`${FIG}/fig_v2_latency.png`, "**Figure 10.** Median stage latency on the capture device (left) and the server (right), log scale. Every cryptographic stage is below 5 ms; the models dominate."),
    H2("5.3 Early rejection under attack"),
    P(`This experiment tests the claim behind the integrated protocol. The same server code is run in two modes. In the integrated mode (the default), the header-first envelope lets the server check nonce, freshness and signature before it reads the frames. In the layered mode (PIPELINE_MODE=layered, an ablation switch), the server decodes the frames and runs liveness and all five detectors on every request before the cryptographic checks, as a deployment that treats anti-spoofing and transport security as separate layers would. Each of four protocol attacks was sent 30 times with a full-size payload. A forged request carries a valid ML-KEM ciphertext, since anyone can encapsulate to the session key, and a signature from the attacker's own key.`),
    tableCaption("**Table 13.** Cost of rejected requests: server processing time and latency seen by the sender (medians, ms). Every request was rejected in both modes."),
    table(
      ["Attack", "Rejected by", "Server time, integrated", "Server time, layered", "Ratio", "HTTP, integrated", "HTTP, layered"],
      kinds.map(([k, label]) => [label, I[k].check, I[k].server, Ly[k].server, (Ly[k].server_raw / I[k].server_raw).toFixed(0) + "×", I[k].http, Ly[k].http]),
      [2200, 1626, 1100, 1100, 800, 1100, 1100],
      { size: 17, align: [L, L, C, C, C, C, C] },
    ),
    P(`In the integrated mode a forged signature costs the server ${I.forged_signature.server} ms; in the layered mode the same request costs ${Ly.forged_signature.server} ms, ${N.dos_ratio_forged} times more, because the models run before anyone asks whether the evidence is authentic. Reused nonces and stale timestamps are cheaper still, since they fail before the signature is verified. Substituted frames are the one case where the frames must be read, because the evidence hash is the check that fails, but no image is decoded and no model runs. The latency seen by the sender includes uploading and reading the multi-megabyte body, which both modes pay; the difference is everything after that.`),
    P(`The flood experiment measures what this means for honest users. One victim process ran genuine verifications for ${FI1.seconds} s while ${FI1.attackers} attacker processes sent forged submissions as fast as they could, each opening a new session for the victim's device. Without an attack, the victim's median latency was ${FI0.victim_med} s in the integrated mode and ${FL0.victim_med} s in the layered mode. Under the flood it rose to ${FI1.victim_med} s in the integrated mode and ${FL1.victim_med} s in the layered mode, a similar slowdown. What differs is what the attacker had to spend to cause it. In the integrated mode the server answered ${FI1.forged_n} forged submissions in ${FI1.seconds} s (${FI1.forged_rate} per second); in the layered mode it answered only ${FL1.forged_n} (${FL1.forged_rate} per second), because each one occupied the worker for about half a second of model inference. To impose the same delay on honest users, an attacker therefore needs about ${N.flood_absorb_ratio} times more requests, and as much more upload bandwidth, against the integrated design. No forged request was accepted in either mode. The slowdown that remains in the integrated mode is the cost of receiving the attackers' multi-megabyte bodies on a single worker; it can be capped by bounding body size per session and rate-limiting session creation per device, measures that complement early cryptographic rejection rather than replace it [24].`),
    ...figure(`${FIG}/fig_v2_dos.png`, "**Figure 11.** Left: server processing time per rejected request (log scale) in the integrated pipeline (blue) and the layered baseline (orange). Right: median latency of genuine verifications with no attack and while four processes flood the server with forged submissions."),
    H2("5.4 Quantum circuits on ideal and noisy back-ends"),
    P(`The quantum experiments are unchanged from the first report and are included because they justify the parameter choices in Section 5.1. On the ideal Aer simulator (8,192 shots), Shor's circuit for N = 15 recovered the factors with probability ${Number(sweep[0].shor_success).toFixed(3)} (theory 0.5) and Grover's search found the 4-bit key with probability ${Number(sweep[0].grover_success).toFixed(3)} (theory 0.961, against 0.0625 for one classical guess). After transpilation to {cx, rz, sx, x}, the Shor circuit needs 792 CNOTs at depth 1,573, against 84 CNOTs for Grover. Table 14 sweeps the two-qubit depolarising error p_{2} (with one-qubit error p_{2}/10 and readout error 2p_{2}).`),
    tableCaption("**Table 14.** Success probability against two-qubit gate error (Qiskit Aer, 8,192 shots)."),
    table(
      ["p_{2}", ...sweep.map((r) => (Number(r.p2) * 100).toString().replace(/^0$/, "0") + "%")],
      [
        ["Shor P(factors)", ...sweep.map((r) => Number(r.shor_success).toFixed(3))],
        ["Grover P(key)", ...sweep.map((r) => Number(r.grover_success).toFixed(3))],
        ["QRNG P(bit = 1)", ...sweep.map((r) => Number(r.qrng_bit_bias_p1).toFixed(3))],
      ],
      [1826, ...sweep.map(() => Math.floor(7200 / sweep.length))],
      { size: 17, align: [L, ...sweep.map(() => C)] },
    ),
    P("A two-qubit error rate of 0.1%, close to today's best superconducting gates, already removes about a third of Shor's success probability, and at 1% the circuit is indistinguishable from random output. Grover's shallower circuit degrades more slowly. Gate fidelity, not qubit count, separates current devices from a cryptographic attack, so RSA-2048 requires the error-corrected machines in [6], [7]. That delays the threat but does not remove it, and harvest-now-decrypt-later makes the delay irrelevant for biometric data with a lifetime of decades [9]. Since Table 10 shows the replacement costs well under a millisecond, there is no performance argument for waiting."),
    ...figure(`${FIG}/fig_q6_noise_sweep.png`, "**Figure 12.** Shor and Grover success probability and QRNG min-entropy against two-qubit gate error.", 0.85),
  ];
};
