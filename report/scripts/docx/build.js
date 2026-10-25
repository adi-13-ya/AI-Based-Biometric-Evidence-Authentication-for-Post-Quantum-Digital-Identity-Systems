// Builds the final report (.docx) from the section modules and report_numbers.json.
// Usage: node build.js <repo root> <output.docx>
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, AlignmentType, Footer, PageNumber, LevelFormat, HeadingLevel, ExternalHyperlink,
} = require("docx");
const { FONT, BODY, P, runs } = require("./lib");

const REPO = process.argv[2];
const OUT = process.argv[3];
const FIG = path.join(REPO, "report", "figures");
const DATA = path.join(REPO, "report", "data");
const N = JSON.parse(fs.readFileSync(path.join(DATA, "v2", "report_numbers.json"), "utf8"));

const rel = (d) => N[`pad_released_${d}`];
const dfE = N.df_efficientnet_b0;
const injC = N.inj_classifier, injA = N.inj_legacy_adaptive_attacker;
const I = N.sys_integrated, Ly = N.sys_layered;
const FI1 = N.flood_integrated_on, FL1 = N.flood_layered_on;

const title = [
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 120 },
    children: [new TextRun({ text: "AI-Based Biometric Evidence Authentication for Post-Quantum Digital Identity Systems: Transaction Binding with Early Cryptographic Rejection", bold: true, font: FONT, size: 30 })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 40 },
    children: runs("Aditya Gujar (23BIT0224), Devangi Joshi (23BIT0178), Kevin Kurian Albert (23BIT0263)"),
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 40 },
    children: runs("aditya.070505@gmail.com, devangijoshi2004@gmail.com, lionelkevin05@gmail.com", { size: 22 }),
  }),
  new Paragraph({ spacing: { after: 120 }, children: [] }),
];

const label = (name, text) => new Paragraph({
  alignment: AlignmentType.JUSTIFIED,
  spacing: { after: 100, line: 276 },
  children: [new TextRun({ text: `${name}: `, bold: true, font: FONT, size: BODY }), ...runs(text)],
});

const abstract = [
  new Paragraph({
    alignment: AlignmentType.LEFT,
    spacing: { before: 120, after: 100 },
    children: [new TextRun({ text: "Abstract", bold: true, font: FONT, size: 30 })],
  }),
  label("Context", "Remote identity verification sends a face capture from a user's camera to a server that must decide whether it is a live, genuine presentation. Its channel is protected by RSA and elliptic-curve cryptography, which a quantum computer running Shor's algorithm will break, and its content is judged by anti-spoofing components that a secure channel cannot help."),
  label("Problem", "Deployments treat transport security and attack detection as separate layers. The server then cannot tell authentic evidence from forged evidence until after it has paid for model inference, and an earlier prototype of ours also relied on hand-tuned heuristics, signed evidence on the server on the client's behalf, and accepted virtual-camera streams."),
  label("Methodology & Novelty", "We treat post-quantum cryptography and liveness checking as one transaction-binding protocol. The capture device signs the evidence hash, a single-use 256-bit nonce, a timestamp and a liveness-challenge answer with an ML-DSA-65 key that never leaves it, and seals the submission with ML-KEM-768 and AES-256-GCM in a header-first envelope. The server checks nonce, freshness and signature on the small header before it reads, decrypts or decodes any frame, and only authentic evidence reaches the detectors: MiniFASNet for presentation attacks, an EfficientNet-B0 fine-tuned on DeepFakeFace for deepfakes, and a pixel-based classifier for injection that does not trust client-reported timing. Thresholds are fixed on development data, and ISO/IEC 30107-3 APCER and BPCER are reported on CASIA-FASD, CelebA-Spoof, a public webcam liveness sample and DeepFakeFace. Shor's and Grover's circuits are simulated with noise to ground the threat model."),
  label("Key Findings", `In a two-process deployment, a forged submission costs the server ${I.forged_signature.server} ms in the integrated pipeline against ${Ly.forged_signature.server} ms when the models run first, and under a flood the server absorbs ${N.flood_absorb_ratio} times as many forged requests for the same slowdown of genuine users. All post-quantum operations together take ${N.pqc_server_ms} ms of server time. MiniFASNet reaches a worst-species APCER of ${rel("casia_test").apcer_worst} at a BPCER of ${rel("casia_test").bpcer} on CASIA-FASD (EER ${rel("casia_test").eer}, against ${N.pad_legacy_casia_test.eer} for the heuristic it replaces), but its AUC falls to ${rel("celeba_test").auc} on CelebA-Spoof and ${rel("webcam").auc} on webcam video. The deepfake model reaches an AUC of ${dfE.auc_gen.insight} on unseen identities of its training generator and does not transfer to unseen diffusion generators. The injection classifier rejects frame-rate-converted streams at an APCER of ${injC.inj_25fps_crf23} and a BPCER of ${injC.bpcer}, while the old timing check accepts ${injA.inj_25fps_crf23} of injected clips once the attacker forges jitter.`),
  label("Significance", "Post-quantum protection of biometric evidence is affordable today, and ordering cryptographic checks ahead of inference turns that protection into a measurable defence against resource exhaustion. The cross-dataset results show that trained detectors alone are not enough for unconstrained webcams, which is why the protocol requires the cryptographic, liveness and content checks together."),
  new Paragraph({
    alignment: AlignmentType.JUSTIFIED,
    spacing: { before: 120, after: 240 },
    children: [new TextRun({ text: "Keywords: ", bold: true, font: FONT, size: BODY }), ...runs("Face presentation attack detection; Injection attack detection; ML-DSA; ML-KEM; Post-quantum cryptography; Transaction binding")],
  }),
];

const sections = [
  ...title,
  ...abstract,
  ...require("./sec_intro")(N),
  ...require("./sec_method")(N, FIG),
  ...require("./sec_detect")(N, FIG),
  ...require("./sec_system")(N, FIG, DATA),
  ...require("./sec_analysis")(N),
  ...require("./sec_back")(N),
];

const doc = new Document({
  creator: "Aditya Gujar, Devangi Joshi, Kevin Kurian Albert",
  title: "AI-Based Biometric Evidence Authentication for Post-Quantum Digital Identity Systems",
  styles: {
    default: { document: { run: { font: FONT, size: BODY } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: FONT, size: 30, bold: true, color: "000000" }, paragraph: { spacing: { before: 280, after: 140 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: FONT, size: BODY, bold: true, color: "000000" }, paragraph: { spacing: { before: 200, after: 100 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: FONT, size: BODY, bold: true, italics: true, color: "000000" }, paragraph: { spacing: { before: 140, after: 80 }, outlineLevel: 2 } },
    ],
  },
  numbering: {
    config: [
      { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 567, hanging: 283 } } } }] },
      { reference: "gates", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 567, hanging: 283 } } } }] },
    ],
  },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440, footer: 708 } } },
    footers: {
      default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 20 })] })] }),
    },
    children: sections,
  }],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(OUT, buf);
  console.log(`wrote ${OUT} (${(buf.length / 1024).toFixed(0)} KiB)`);
});
