"""Fusion / Decision Engine - spec Section 6.6.

Weighted-average fusion of the five detector module scores, thresholded for
a final ACCEPT/REJECT decision. Section 6.6 names a learned fusion (small
logistic regression over the 5 scores) as a stretch goal; not attempted here
given the offline/no-labeled-data constraints of this build - the weighted
average is what spec Section 10's own risk mitigation recommends starting
with ("Keep fusion simple (weighted average) initially; only attempt learned
fusion if ahead of schedule").

Decision rule: every module is a separate veto. A submission is ACCEPTed
only if each module's score is at or above that module's own threshold;
one module below its threshold rejects, however well the others score (a
confident replay-hash match must not be washed out by four unrelated
modules scoring well). The weighted mean is NOT part of the decision: it is
computed and reported only (dashboard, explanations, logs).

Why the mean is not used to decide: a weighted mean can clear any fixed
threshold while one module is far below its own (four modules at 0.95 and
one at 0.0 average above 0.5). The first version (v0.2) used "mean >= 0.6
AND no veto"; v1.0 used "normalised mean >= 0.5 AND no veto". In both, the
veto is what decided every rejection that the mean alone would have missed,
and in v1.0 the mean condition could never reject anything the veto passed
(no module below threshold implies every normalised score >= 0.5, so the
mean is >= 0.5). Veto-only therefore makes exactly the same decisions as
v1.0 on every input where all five thresholds are set, which is how the
server always calls fuse(); see tests/test_fusion.py.

Per-module thresholds are calibrated on development data and differ between
modules (a model's operating point is rarely 0.5). For the reported mean,
each score is first mapped piecewise-linearly so that its own threshold
lands at 0.5 (0 -> 0, threshold -> 0.5, 1 -> 1), so the mean does not mix
scales. A module with no threshold given uses 0.5.
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
DEFAULT_MODULE_THRESHOLD = 0.5


def normalise(score: float, threshold: float) -> float:
    """Monotonic map of [0, 1] onto [0, 1] that sends `threshold` to 0.5."""
    t = min(max(threshold, 1e-6), 1 - 1e-6)
    return clip01(0.5 * score / t if score < t else 0.5 + 0.5 * (score - t) / (1 - t))


@dataclass
class FusionResult:
    accepted: bool
    fused_score: float  # reported only; not used for the decision
    per_module_scores: dict[str, float]
    per_module_is_mock: dict[str, bool]
    below_threshold_modules: list[str] = field(default_factory=list)


def fuse(
    results: list[DetectionResult],
    weights: dict[str, float] | None = None,
    per_module_thresholds: dict[str, float] | None = None,
) -> FusionResult:
    weights = weights or DEFAULT_WEIGHTS
    per_module_thresholds = per_module_thresholds or {}

    if not results:
        raise ValueError("fuse() requires at least one DetectionResult")

    scores = {r.module: r.score for r in results}
    is_mock = {r.module: r.is_mock for r in results}

    thresholds = {m: per_module_thresholds.get(m, DEFAULT_MODULE_THRESHOLD) for m in scores}
    normalised = {m: normalise(s, thresholds[m]) for m, s in scores.items()}
    weight_sum = sum(weights.get(m, 0.0) for m in scores) or 1.0
    fused_score = clip01(sum(normalised[m] * weights.get(m, 0.0) for m in scores) / weight_sum)

    below_threshold = sorted(m for m, s in scores.items() if s < thresholds[m])
    accepted = not below_threshold

    return FusionResult(
        accepted=accepted,
        fused_score=fused_score,
        per_module_scores=scores,
        per_module_is_mock=is_mock,
        below_threshold_modules=below_threshold,
    )
