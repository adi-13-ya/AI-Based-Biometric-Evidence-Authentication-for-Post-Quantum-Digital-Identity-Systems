"""Section 8 evaluation metrics: standard classification metrics plus the
ISO/IEC 30107-3 PAD metrics APCER/BPCER, computed from (score, label) pairs.

Score convention matches detection/base.py: higher = more likely
bona-fide/genuine. `label=True` means the sample actually is bona-fide;
`label=False` means it's an attack.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ClassificationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    apcer: float  # fraction of ATTACK samples wrongly classified as genuine (false accepts)
    bpcer: float  # fraction of GENUINE samples wrongly classified as attack (false rejects)
    n_genuine: int
    n_attack: int
    threshold: float

    def as_dict(self) -> dict:
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "apcer": self.apcer,
            "bpcer": self.bpcer,
            "n_genuine": self.n_genuine,
            "n_attack": self.n_attack,
            "threshold": self.threshold,
        }


def evaluate(scores: list[float], labels: list[bool], threshold: float = 0.5) -> ClassificationMetrics:
    if len(scores) != len(labels):
        raise ValueError("scores and labels must be the same length")
    if not scores:
        raise ValueError("evaluate() requires at least one (score, label) sample")

    predictions = [s >= threshold for s in scores]  # True = predicted genuine

    tp = sum(1 for p, y in zip(predictions, labels) if p and y)
    tn = sum(1 for p, y in zip(predictions, labels) if not p and not y)
    fp = sum(1 for p, y in zip(predictions, labels) if p and not y)  # predicted genuine, actually attack
    fn = sum(1 for p, y in zip(predictions, labels) if not p and y)  # predicted attack, actually genuine

    n = len(labels)
    n_genuine = sum(1 for y in labels if y)
    n_attack = n - n_genuine

    accuracy = (tp + tn) / n
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    apcer = fp / n_attack if n_attack else 0.0
    bpcer = fn / n_genuine if n_genuine else 0.0

    return ClassificationMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        apcer=apcer,
        bpcer=bpcer,
        n_genuine=n_genuine,
        n_attack=n_attack,
        threshold=threshold,
    )


# ---------------------------------------------------------------------------
# ISO/IEC 30107-3 reporting helpers
#
# APCER is computed per presentation-attack-instrument (PAI) species and the
# headline APCER is the worst species (ISO/IEC 30107-3:2017 clause 13.2:
# APCER_PAIS per species, overall APCER = max over species). BPCER is over
# all bona fide presentations. Thresholds are always chosen on a separate
# development set and then frozen for the test set; picking them on the test
# scores would make the reported error rates optimistic.
# ---------------------------------------------------------------------------
import numpy as _np


def _arrays(scores, labels):
    s = _np.asarray(scores, dtype=_np.float64)
    y = _np.asarray(labels, dtype=bool)
    if s.shape != y.shape or s.size == 0:
        raise ValueError("scores and labels must be non-empty and the same length")
    return s, y


def apcer_bpcer(scores, labels, threshold: float) -> tuple[float, float]:
    """(APCER, BPCER) at `threshold`; score >= threshold is accepted as bona fide."""
    s, y = _arrays(scores, labels)
    accept = s >= threshold
    apcer = float(accept[~y].mean()) if (~y).any() else 0.0
    bpcer = float((~accept[y]).mean()) if y.any() else 0.0
    return apcer, bpcer


def apcer_per_species(scores, labels, species, threshold: float) -> dict[str, float]:
    """APCER for each PAI species (attack samples only); `species` gives the
    species name of every sample (ignored for bona fide samples)."""
    s, y = _arrays(scores, labels)
    sp = _np.asarray(species)
    out = {}
    for name in sorted(set(sp[~y])):
        mask = (~y) & (sp == name)
        out[str(name)] = float((s[mask] >= threshold).mean())
    return out


def eer(scores, labels) -> tuple[float, float]:
    """Equal error rate and the threshold where APCER and BPCER cross
    (evaluated at every distinct score; the EER is the mean of the two rates
    at the closest crossing)."""
    s, y = _arrays(scores, labels)
    best = (2.0, 0.5, 0.5)
    for t in _np.unique(s):
        a, b = apcer_bpcer(s, y, t)
        if abs(a - b) < best[0]:
            best = (abs(a - b), (a + b) / 2, float(t))
    return best[1], best[2]


def threshold_at_apcer(scores, labels, target_apcer: float) -> float:
    """Lowest threshold whose APCER on these (development) scores is <=
    target_apcer - i.e. the most bona-fide-friendly operating point that
    still meets the attack budget."""
    s, y = _arrays(scores, labels)
    attack = _np.sort(s[~y])
    if attack.size == 0:
        return 0.0
    k = int(_np.floor(target_apcer * attack.size))  # attacks allowed through
    if k >= attack.size:
        return float(attack[0])
    # accept iff score >= t: choose t just above the (size-k)-th highest attack score
    return float(_np.nextafter(attack[attack.size - k - 1], _np.inf))


def threshold_at_bpcer(scores, labels, target_bpcer: float) -> float:
    """Highest threshold whose BPCER on these (development) scores is <= target."""
    s, y = _arrays(scores, labels)
    bona = _np.sort(s[y])
    if bona.size == 0:
        return 1.0
    k = int(_np.floor(target_bpcer * bona.size))  # bona fide allowed to be rejected
    return float(bona[min(k, bona.size - 1)])


def bpcer_at_apcer(scores, labels, target_apcer: float) -> float:
    """BPCER at the operating point where APCER <= target (computed on the
    same scores; report it as a curve point, not as a deployed error rate)."""
    t = threshold_at_apcer(scores, labels, target_apcer)
    return apcer_bpcer(scores, labels, t)[1]


def auc(scores, labels) -> float:
    """ROC AUC (probability a random bona fide outscores a random attack)."""
    s, y = _arrays(scores, labels)
    pos, neg = s[y], s[~y]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    order = _np.argsort(_np.concatenate([pos, neg]), kind="mergesort")
    ranks = _np.empty(order.size)
    allv = _np.concatenate([pos, neg])[order]
    # average ranks for ties
    i = 0
    r = _np.arange(1, order.size + 1, dtype=_np.float64)
    while i < order.size:
        j = i
        while j + 1 < order.size and allv[j + 1] == allv[i]:
            j += 1
        r[i: j + 1] = (i + j + 2) / 2
        i = j + 1
    ranks[order] = r
    return float((ranks[: pos.size].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def wilson_interval(errors: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for an error rate errors/n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = errors / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))
