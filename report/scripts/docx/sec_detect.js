// Section 4: Detection accuracy on public benchmarks.
const { AlignmentType } = require("docx");
const { P, H1, H2, table, tableCaption, figure } = require("./lib");

const C = AlignmentType.CENTER;
const L = AlignmentType.LEFT;

module.exports = (N, FIG) => {
  const pad = (s, d) => N[`pad_${s}_${d}`];
  const sp = (s, d, k) => pad(s, d).apcer_species[k];
  const padRow = (label, s) => [
    label,
    sp(s, "casia_test", "warped_photo"), sp(s, "casia_test", "cut_photo"), sp(s, "casia_test", "video_replay"),
    pad(s, "casia_test").bpcer, pad(s, "casia_test").eer, pad(s, "casia_test").auc,
    pad(s, "celeba_test").apcer_all, pad(s, "celeba_test").bpcer, pad(s, "celeba_test").auc,
    pad(s, "webcam").apcer_all, pad(s, "webcam").bpcer, pad(s, "webcam").auc,
  ];
  const dfE = N.df_efficientnet_b0;
  const dfL = N.df_legacy;
  const injC = N.inj_classifier;
  const injN = N.inj_legacy_naive_attacker;
  const injA = N.inj_legacy_adaptive_attacker;
  const conds = ["inj_25fps_crf23", "inj_24fps_crf28", "inj_30fps_crf23", "inj_30fps_crf18", "inj_lossless_30"];
  const rel = (d) => pad("released", d);
  const ft = (d) => pad("finetuned", d);
  const leg = (d) => pad("legacy", d);

  return [
    H1("4. Detection Accuracy on Public Benchmarks"),
    P("This section answers the review's first question: do the detectors work on data they were not tuned on? Every threshold below was fixed on development data before the test sets were scored (Section 3.5). For each module we also score the v0.2 heuristic on the same inputs, so the comparison is like for like."),
    H2("4.1 Presentation attack detection"),
    P(`Table 6 reports ISO/IEC 30107-3 metrics for three PAD systems on three test sets. CASIA-FASD test is the intra-dataset condition (30 subjects never used for training or threshold selection). CelebA-Spoof and the webcam sample are cross-dataset: different cameras, subjects, lighting and attack instruments. The webcam sample is scored per video, as a submitted clip would be, and has only 10 bona fide and 60 attack videos, so its rates carry wide intervals (for example, the released model's BPCER of ${rel("webcam").bpcer} has a 95% interval of ${rel("webcam").bpcer_ci}).`),
    tableCaption(`**Table 6.** PAD results at each system's development-EER threshold (released MiniFASNet τ = ${N.pad_released_thr}; legacy τ = ${N.pad_legacy_thr}; fine-tuned τ = ${N.pad_finetuned_thr}). APCER per PAI species for CASIA-FASD; overall APCER for the other sets.`),
    table(
      ["System", "Warped photo APCER", "Cut photo APCER", "Video replay APCER", "BPCER", "EER", "AUC", "APCER", "BPCER", "AUC", "APCER", "BPCER", "AUC"],
      [
        ["", "CASIA-FASD test", "", "", "", "", "", "CelebA-Spoof test", "", "", "Webcam (per video)", "", ""],
        padRow("v0.2 heuristic", "legacy"),
        padRow("MiniFASNet, released (deployed)", "released"),
        padRow("MiniFASNet, CASIA fine-tuned", "finetuned"),
      ],
      [1450, 640, 640, 640, 640, 600, 600, 660, 640, 600, 660, 640, 616],
      { size: 16, align: [L, C, C, C, C, C, C, C, C, C, C, C, C] },
    ),
    P(`**The heuristic does not work.** On CASIA-FASD the v0.2 rules reach an AUC of ${leg("casia_test").auc} and an EER of ${leg("casia_test").eer}; on the webcam sample the AUC is ${leg("webcam").auc}, no better than a coin. Their thresholds and ramps had been set on one portrait and a few simulated webcam frames, and they do not transfer. This confirms the review's main criticism with numbers.`),
    P(`**MiniFASNet works well in the setting closest to its training data and degrades across domains.** With the released checkpoints and no adaptation, the worst-species APCER on CASIA-FASD is ${rel("casia_test").apcer_worst} at a BPCER of ${rel("casia_test").bpcer} ${rel("casia_test").bpcer_ci}, the EER is ${rel("casia_test").eer} and the AUC ${rel("casia_test").auc}. On CelebA-Spoof the AUC falls to ${rel("celeba_test").auc} and, at the frozen threshold, ${rel("celeba_test").apcer_all} of attacks are accepted. On webcam video the AUC is ${rel("webcam").auc}, with ${rel("webcam").apcer_all} of attack videos accepted overall and every print and paper-outline video accepted. Part of this gap is face localisation: RetinaFace's confidence was below 0.6 for ${N.pad_celeba_test_lowconf} of CelebA-Spoof images and ${N.pad_webcam_lowconf} of webcam frames, against 0% on the tightly cropped CASIA frames. The rest is the familiar cross-dataset gap of PAD models [35], [36].`),
    P(`**Fine-tuning helped where it was not needed and hurt where it mattered.** Fine-tuning on CASIA-FASD lowered the intra-dataset EER from ${rel("casia_test").eer} to ${ft("casia_test").eer} (AUC ${ft("casia_test").auc}) but lowered cross-dataset AUC to ${ft("celeba_test").auc} on CelebA-Spoof and ${ft("webcam").auc} on webcam video. The model learned the specifics of CASIA's 20 training subjects and cameras. Because the deployment target is an unknown user's webcam, the server uses the released checkpoints; the fine-tuned weights remain available (PAD_WEIGHTS=finetuned) for comparison.`),
    ...figure(`${FIG}/fig_v2_pad_roc.png`, "**Figure 5.** PAD ROC curves (APCER against 1 − BPCER) on the intra-dataset test (left) and the cross-dataset CelebA-Spoof test (right). The v0.2 heuristic stays near the diagonal; fine-tuning on CASIA-FASD improves the left panel and worsens the right."),
    ...figure(`${FIG}/fig_v2_pad_species.png`, "**Figure 6.** Per-species APCER and BPCER at the development-EER threshold. On CASIA-FASD the released MiniFASNet accepts under 1% of each attack species; on webcam video, prints and paper outlines are accepted every time."),
    P(`Threshold-free operating points tell the same story. On CASIA-FASD, holding APCER at 5% costs a BPCER of ${rel("casia_test").b5} with the released model; on CelebA-Spoof the same APCER costs ${rel("celeba_test").b5}. A deployment that needs low APCER on arbitrary webcams cannot rely on this model alone, which is one reason the protocol makes every other check, cryptographic and behavioural, mandatory as well.`),
    H2("4.2 Deepfake detection"),
    P(`Table 7 reports the deepfake module on ${N.df_datasets.test.images} held-out DFF test images (persons never seen in training): 400 real portraits and 400 fakes from each generator. The threshold (τ = ${dfE.thr}) is the EER point of the development identities, where the AUC was ${dfE.dev_auc} and the EER ${dfE.dev_eer}.`),
    tableCaption("**Table 7.** Deepfake detection on DeepFakeFace, identity-disjoint test split."),
    table(
      ["System", "BPCER", "APCER insight (trained)", "APCER inpainting (unseen)", "APCER text2img (unseen)", "AUC insight", "AUC inpainting", "AUC text2img"],
      [
        ["v0.2 FFT heuristic", dfL.bpcer, dfL.apcer.insight, dfL.apcer.inpainting, dfL.apcer.text2img, dfL.auc_gen.insight, dfL.auc_gen.inpainting, dfL.auc_gen.text2img],
        ["EfficientNet-B0, fine-tuned (deployed)", dfE.bpcer, dfE.apcer.insight, dfE.apcer.inpainting, dfE.apcer.text2img, dfE.auc_gen.insight, dfE.auc_gen.inpainting, dfE.auc_gen.text2img],
      ],
      [1826, 900, 1050, 1100, 1100, 1000, 1050, 1000],
      { size: 16, align: [L, C, C, C, C, C, C, C] },
    ),
    P(`The FFT heuristic scores every face crop at the top of its range, so its AUC is ${dfL.auc_gen.insight} and it accepts essentially every fake. The fine-tuned network separates real faces from face swaps by the generator it was trained on with an AUC of ${dfE.auc_gen.insight}, accepting ${dfE.apcer.insight} of them at a BPCER of ${dfE.bpcer} ${dfE.bpcer_ci}. It does not generalise to diffusion models it has not seen: the AUC is ${dfE.auc_gen.inpainting} for inpainting and ${dfE.auc_gen.text2img} for text-to-image, the latter below chance, meaning those synthetic faces look more \"real\" to the model than real IMDB-WIKI photographs. Two caveats apply. The training set is small (${N.df_n_train} images, limited by the download budget described in Section 3.7), and the real images in DFF vary in size and compression while the fakes share one resolution, a shortcut that cropping, resizing and JPEG augmentation reduce but cannot rule out. Generalisation across generators is the main open problem in this area [16], and our result reproduces it rather than solving it.`),
    ...figure(`${FIG}/fig_v2_deepfake.png`, "**Figure 7.** Deepfake detection per generator: AUC (left, dashed line = chance) and APCER at the frozen threshold (right)."),
    H2("4.3 Injection (virtual camera) detection"),
    P(`No public benchmark of virtual-camera injection exists, so we built one from real webcam footage and stated the protocol in evaluation/train_injection.py. All ${N.inj_videos} webcam recordings supply camera clips: 8 consecutive decoded frames, as the client sends them. The same footage, treated as the attacker's video file, is encoded with H.264 at the file's frame rate and quality, decoded, and resampled to the 30 fps a virtual camera advertises. Five attack conditions vary the file's frame rate (24, 25 or 30 fps) and quality (CRF 18 to 28, or lossless). In total there are ${N.inj_clips} clips. The classifier is evaluated with 5-fold cross-validation grouped by source video, with each fold's threshold set at BPCER = 5% on its training folds; the lossless condition is held out of training.`),
    P("The baseline is the v0.2 timing heuristic, evaluated against two attackers. A naive attacker sends the perfectly regular timestamps a video file produces; an adaptive attacker adds camera-like jitter (normally distributed, σ = 1.7 ms) to its own timestamps, which costs nothing because the client produces them."),
    tableCaption("**Table 8.** Injection detection: APCER per attack condition and BPCER (out-of-fold for the classifier)."),
    table(
      ["System / attacker", "BPCER", "25→30 fps CRF 23", "24→30 fps CRF 28", "30 fps CRF 23", "30 fps CRF 18", "30 fps lossless"],
      [
        ["v0.2 timing heuristic / naive", injN.bpcer, ...conds.map((c) => injN[c])],
        ["v0.2 timing heuristic / forged jitter", injA.bpcer, ...conds.map((c) => injA[c])],
        ["Content classifier (deployed) / either", injC.bpcer, ...conds.map((c) => injC[c])],
      ],
      [2626, 900, 1100, 1100, 1100, 1100, 1100],
      { size: 16, align: [L, C, C, C, C, C, C] },
    ),
    P(`The timing heuristic is perfect against the naive attacker and useless against the adaptive one: it accepts ${injA.inj_25fps_crf23} of injected clips once the attacker adds jitter. That is the failure the review pointed out, and it cannot be fixed by tuning, because the signal is under the attacker's control. The content classifier does not read timestamps, so the two attackers are the same to it. It rejects nearly all clips that went through frame-rate conversion (APCER ${injC.inj_25fps_crf23} and ${injC.inj_24fps_crf28}) at a BPCER of ${injC.bpcer}, with an AUC of ${injC.auc} over the trained conditions. When the file already runs at 30 fps only the second-compression cues remain, and APCER rises to ${injC.inj_30fps_crf23} at CRF 23 and ${injC.inj_30fps_crf18} at CRF 18. A lossless 30 fps file is pixel-identical to a camera clip in this emulation, and ${injC.inj_lossless_30} of those clips are accepted, as expected. The largest standardised weights belong to the 8- and 16-pixel block-grid features and the static-block share, which matches the reasoning in Section 3.4.`),
    ...figure(`${FIG}/fig_v2_injection.png`, "**Figure 8.** Injection APCER per attack condition. Grey: v0.2 timing heuristic against a naive attacker (light) and an attacker who forges jitter (dark). Blue: the content classifier, which is unaffected by forged timestamps."),
    H2("4.4 Benchmark comparison"),
    P("Table 9 summarises, for each learned module, the v0.2 heuristic against the v1.0 model on identical test data, using threshold-free AUC and EER so that the comparison does not depend on where a threshold happened to fall."),
    tableCaption("**Table 9.** v0.2 heuristics against v1.0 models on the same test data."),
    table(
      ["Module", "Test data", "v0.2 AUC", "v0.2 EER", "v1.0 AUC", "v1.0 EER"],
      [
        ["PAD", "CASIA-FASD test (intra)", leg("casia_test").auc, leg("casia_test").eer, rel("casia_test").auc, rel("casia_test").eer],
        ["PAD", "CelebA-Spoof test (cross)", leg("celeba_test").auc, leg("celeba_test").eer, rel("celeba_test").auc, rel("celeba_test").eer],
        ["PAD", "Webcam videos (cross)", leg("webcam").auc, leg("webcam").eer, rel("webcam").auc, rel("webcam").eer],
        ["Deepfake", "DFF test, all generators", dfL.auc, dfL.eer, dfE.auc, dfE.eer],
        ["Injection", "Emulated virtual camera, forged jitter", injA.auc, injA.eer, injC.auc, injC.eer],
      ],
      [1300, 3026, 1175, 1175, 1175, 1175],
      { size: 18, align: [L, L, C, C, C, C] },
    ),
    P("Every learned module improves on the heuristic it replaces, and the improvements are largest exactly where the heuristics had been calibrated by eye. The table also shows where the models remain weak (PAD across cameras, deepfakes from unseen generators), and these are the two places where the integrated protocol's other checks matter most."),
  ];
};
