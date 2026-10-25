// Back matter: code availability, declarations, references.
const { Paragraph, TextRun, AlignmentType } = require("docx");
const { P, H1, H2, runs, FONT } = require("./lib");

const REFS = [
  "P. W. Shor, \"Polynomial-time algorithms for prime factorization and discrete logarithms on a quantum computer,\" *SIAM J. Comput.*, vol. 26, no. 5, pp. 1484–1509, 1997.",
  "L. K. Grover, \"A fast quantum mechanical algorithm for database search,\" in *Proc. 28th Annu. ACM Symp. Theory Comput. (STOC)*, 1996, pp. 212–219.",
  "National Institute of Standards and Technology, \"Module-lattice-based key-encapsulation mechanism standard,\" FIPS 203, Aug. 2024.",
  "National Institute of Standards and Technology, \"Module-lattice-based digital signature standard,\" FIPS 204, Aug. 2024.",
  "D. Stebila and M. Mosca, \"Post-quantum key exchange for the Internet and the Open Quantum Safe project,\" in *Selected Areas in Cryptography (SAC 2016)*, LNCS vol. 10532, Springer, 2017, pp. 14–37.",
  "C. Gidney and M. Ekerå, \"How to factor 2048 bit RSA integers in 8 hours using 20 million noisy qubits,\" *Quantum*, vol. 5, p. 433, 2021.",
  "C. Gidney, \"How to factor 2048 bit RSA integers with less than a million noisy qubits,\" arXiv:2505.15917, 2025.",
  "D. J. Bernstein and T. Lange, \"Post-quantum cryptography,\" *Nature*, vol. 549, no. 7671, pp. 188–194, 2017.",
  "M. Mosca, \"Cybersecurity in an era with quantum computers: Will we be ready?\" *IEEE Security & Privacy*, vol. 16, no. 5, pp. 38–41, 2018.",
  "O. Regev, \"On lattices, learning with errors, random linear codes, and cryptography,\" *J. ACM*, vol. 56, no. 6, Art. 34, 2009.",
  "*Information technology — Biometric presentation attack detection — Part 3: Testing and reporting*, ISO/IEC 30107-3:2023.",
  "Y. Zhang, Z. Yin, Y. Li, G. Yin, J. Yan, J. Shao, and Z. Liu, \"CelebA-Spoof: Large-scale face anti-spoofing dataset with rich annotations,\" in *Proc. Eur. Conf. Comput. Vis. (ECCV)*, 2020, pp. 70–85.",
  "Z. Zhang, J. Yan, S. Liu, Z. Lei, D. Yi, and S. Z. Li, \"A face antispoofing database with diverse attacks,\" in *Proc. 5th IAPR Int. Conf. Biometrics (ICB)*, 2012, pp. 26–31.",
  "Minivision AI, \"Silent-Face-Anti-Spoofing,\" GitHub repository, 2020. [Online]. Available: [[https://github.com/minivision-ai/Silent-Face-Anti-Spoofing|https://github.com/minivision-ai/Silent-Face-Anti-Spoofing]]",
  "J. Deng, J. Guo, E. Ververas, I. Kotsia, and S. Zafeiriou, \"RetinaFace: Single-shot multi-level face localisation in the wild,\" in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2020, pp. 5203–5212.",
  "H. Song, S. Huang, Y. Dong, and W.-W. Tu, \"Robustness and generalizability of deepfake detection: A study with diffusion models,\" arXiv:2309.02218, 2023.",
  "M. Tan and Q. V. Le, \"EfficientNet: Rethinking model scaling for convolutional neural networks,\" in *Proc. 36th Int. Conf. Mach. Learn. (ICML)*, PMLR vol. 97, 2019, pp. 6105–6114.",
  "A. Rössler, D. Cozzolino, L. Verdoliva, C. Riess, J. Thies, and M. Nießner, \"FaceForensics++: Learning to detect manipulated facial images,\" in *Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV)*, 2019, pp. 1–11.",
  "D. Afchar, V. Nozick, J. Yamagishi, and I. Echizen, \"MesoNet: A compact facial video forgery detection network,\" in *Proc. IEEE Int. Workshop Inf. Forensics Security (WIFS)*, 2018, pp. 1–7.",
  "J. Frank, T. Eisenhofer, L. Schönherr, A. Fischer, D. Kolossa, and T. Holz, \"Leveraging frequency analysis for deep fake image recognition,\" in *Proc. 37th Int. Conf. Mach. Learn. (ICML)*, PMLR vol. 119, 2020, pp. 3247–3258.",
  "U. Scherhag, C. Rathgeb, J. Merkle, R. Breithaupt, and C. Busch, \"Face recognition systems under morphing attacks: A survey,\" *IEEE Access*, vol. 7, pp. 23012–23026, 2019.",
  "W. Wang and H. Farid, \"Exposing digital forgeries in video by detecting double MPEG compression,\" in *Proc. 8th ACM Workshop Multimedia and Security*, 2006, pp. 37–47.",
  "European Committee for Standardization, *Biometric data injection attack detection*, CEN/TS 18099:2024.",
  "T. Aura, P. Nikander, and J. Leiwo, \"DOS-resistant authentication with client puzzles,\" in *Security Protocols (2000)*, LNCS vol. 2133, Springer, 2001, pp. 170–177.",
  "Y. Kartynnik, A. Ablavatski, I. Grishchenko, and M. Grundmann, \"Real-time facial surface geometry from monocular video on mobile GPUs,\" arXiv:1907.06724, 2019.",
  "H. Krawczyk, \"Cryptographic extraction and key derivation: The HKDF scheme,\" in *Advances in Cryptology — CRYPTO 2010*, LNCS vol. 6223, Springer, 2010, pp. 631–648.",
  "M. Grassl, B. Langenberg, M. Roetteler, and R. Steinwandt, \"Applying Grover's algorithm to AES: Quantum resource estimates,\" in *Post-Quantum Cryptography (PQCrypto 2016)*, LNCS vol. 9606, Springer, 2016, pp. 29–43.",
  "M. A. Nielsen and I. L. Chuang, *Quantum Computation and Quantum Information*, 10th anniv. ed. Cambridge, U.K.: Cambridge Univ. Press, 2010.",
  "A. Javadi-Abhari *et al.*, \"Quantum computing with Qiskit,\" arXiv:2405.08810, 2024.",
  "D. Moody, R. Perlner, A. Regenscheid, A. Robinson, and D. Cooper, \"Transition to post-quantum cryptography standards,\" NIST IR 8547 (initial public draft), Nov. 2024.",
  "M. Herrero-Collantes and J. C. Garcia-Escartin, \"Quantum random number generators,\" *Rev. Mod. Phys.*, vol. 89, no. 1, Art. 015004, 2017.",
  "C. Paquin, D. Stebila, and G. Tamvada, \"Benchmarking post-quantum cryptography in TLS,\" in *Post-Quantum Cryptography (PQCrypto 2020)*, LNCS vol. 12100, Springer, 2020, pp. 72–91.",
  "J. Bos *et al.*, \"CRYSTALS-Kyber: A CCA-secure module-lattice-based KEM,\" in *Proc. IEEE Eur. Symp. Security and Privacy (EuroS&P)*, 2018, pp. 353–367.",
  "L. Ducas *et al.*, \"CRYSTALS-Dilithium: A lattice-based digital signature scheme,\" *IACR Trans. Cryptogr. Hardw. Embed. Syst.*, vol. 2018, no. 1, pp. 238–268, 2018.",
  "Z. Yu, Y. Qin, X. Li, C. Zhao, Z. Lei, and G. Zhao, \"Deep learning for face anti-spoofing: A survey,\" *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. 45, no. 5, pp. 5609–5631, 2023.",
  "D. Wen, H. Han, and A. K. Jain, \"Face spoof detection with image distortion analysis,\" *IEEE Trans. Inf. Forensics Security*, vol. 10, no. 4, pp. 746–761, 2015.",
  "L. M. K. Vandersypen, M. Steffen, G. Breyta, C. S. Yannoni, M. H. Sherwood, and I. L. Chuang, \"Experimental realization of Shor's quantum factoring algorithm using nuclear magnetic resonance,\" *Nature*, vol. 414, pp. 883–887, 2001.",
];

const refParas = REFS.map((r, i) => new Paragraph({
  alignment: AlignmentType.LEFT,
  spacing: { after: 80, line: 260 },
  indent: { left: 560, hanging: 560 },
  children: [new TextRun({ text: `[${i + 1}]\t`, font: FONT, size: 20 }), ...runs(r, { size: 20 })],
  tabStops: [{ type: "left", position: 560 }],
}));

module.exports = (N) => [
  H2("Code Availability"),
  P("The complete implementation, tests, experiment scripts, result files (evaluation/results, report/data) and figure scripts are available at [[https://github.com/Devangi777/AI-Based-Biometric-Evidence-Authentication-for-Post-Quantum-Digital-Identity-Systems|github.com/Devangi777/AI-Based-Biometric-Evidence-Authentication-for-Post-Quantum-Digital-Identity-Systems]]. Datasets and model weights are not redistributed. evaluation/fetch_data.py downloads the exact public subsets used here; the MiniFASNet and RetinaFace files come from the Silent-Face-Anti-Spoofing release [14]; evaluation/finetune_pad.py, train_deepfake.py and train_injection.py regenerate the trained weights; and pytest tests/ runs the 117 unit and API tests."),
  H2("Declaration of Interests"),
  P("The authors declare no competing financial interests or personal relationships that could have influenced the work reported in this paper. This work was carried out as a course project for Quantum Computing (BITE407L) and received no external funding."),
  H2("Declaration of AI Usage"),
  P("An AI coding assistant (Claude Code) was used during this revision to help implement code changes (device-side signing, the ordered binding gate, model integration and the training and evaluation scripts), to run experiments, and to draft parts of this text from the code and result files. The authors specified the changes in response to the review, checked the code and its tests, verified every reported number against the generated result files, and take responsibility for the content. No AI tool was used to generate or alter data: all reported measurements come from executing the code described on the public datasets listed in Table 5."),
  H2("Originality Statement"),
  P("We, the members of this project team, declare that this report and the accompanying implementation are our original work. All external ideas, algorithms, models, datasets, software libraries and standards are acknowledged and cited. No part of this work has been submitted elsewhere for academic credit."),
  H1("References"),
  ...refParas,
];
module.exports.REF_COUNT = REFS.length;
