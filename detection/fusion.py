"""Fusion / Decision Engine - spec Section 6.6.

Weighted-average fusion of the five detector module scores, thresholded for
a final ACCEPT/REJECT decision. Section 6.6 names a learned fusion (small
logistic regression over the 5 scores) as a stretch goal; not attempted here
given the offline/no-labeled-data constraints of this build - the weighted
average is what spec Section 10's own risk mitigation recommends starting
with ("Keep fusion simple (weighted average) initially; only attempt learned
fusion if ahead of schedule").

A submission is only ACCEPTed if BOTH:
  - the weighted-average fused score clears `accept_threshold`, AND
  - no individual module fell below its own per-module threshold
    (a single strong "this is an attack" signal from one module can veto an
    otherwise-good weighted average - e.g. a confident replay-hash match
    shouldn't be washed out by four unrelated modules scoring well).

Per-module thresholds are calibrated on development data and differ between
modules (a model's operating point is rarely 0.5). Averaging raw scores
would then mix scales: a PAD score of 0.4 above a 0.3 threshold is a pass,
yet would pull the average down as if it were a weak result. Each score is
therefore first mapped piecewise-linearly so that its own threshold lands
at 0.5 (0 -> 0, threshold -> 0.5, 1 -> 1). The mapping is monotonic, so a
module is below its threshold exactly when its normalised score is below
0.5, and with accept_threshold = 0.5 the fused check can never accept what
the veto rejects or vice versa.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from detection.base import DetectionResult, clip01

DEFAULT_WEIGHTS = {
    "presentation_attack": 0.25,
    "morph": 0.20,
    "deepfake": 0.25,
    "replay": 0.15,
    "injection": 0.15,
}
DEFAULT_ACCEPT_THRESHOLD = 0.5


def normalise(score: float, threshold: float) -> float:
    """Monotonic map of [0, 1] onto [0, 1] that sends `threshold` to 0.5."""
    t = min(max(threshold, 1e-6), 1 - 1e-6)
    return clip01(0.5 * score / t if score < t else 0.5 + 0.5 * (score - t) / (1 - t))


@dataclass
class FusionResult:
    accepted: bool
    fused_score: float
    threshold: float
    per_module_scores: dict[str, float]
    per_module_is_mock: dict[str, bool]
    below_threshold_modules: list[str] = field(default_factory=list)


def fuse(
    results: list[DetectionResult],
    weights: dict[str, float] | None = None,
    per_module_thresholds: dict[str, float] | None = None,
    accept_threshold: float = DEFAULT_ACCEPT_THRESHOLD,
) -> FusionResult:
    weights = weights or DEFAULT_WEIGHTS
    per_module_thresholds = per_module_thresholds or {}

    if not results:
        raise ValueError("fuse() requires at least one DetectionResult")

    scores = {r.module: r.score for r in results}
    is_mock = {r.module: r.is_mock for r in results}

    normalised = {m: normalise(s, per_module_thresholds.get(m, 0.5)) for m, s in scores.items()}
    weight_sum = sum(weights.get(m, 0.0) for m in scores) or 1.0
    fused_score = clip01(sum(normalised[m] * weights.get(m, 0.0) for m in scores) / weight_sum)

    below_threshold = sorted(
        m for m, s in scores.items() if m in per_module_thresholds and s < per_module_thresholds[m]
    )

    accepted = (fused_score >= accept_threshold) and not below_threshold

    return FusionResult(
        accepted=accepted,
        fused_score=fused_score,
        threshold=accept_threshold,
        per_module_scores=scores,
        per_module_is_mock=is_mock,
        below_threshold_modules=below_threshold,
    )
