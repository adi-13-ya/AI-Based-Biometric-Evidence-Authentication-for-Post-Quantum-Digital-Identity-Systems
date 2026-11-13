"""Unit tests for detection/fusion.py using controlled DetectionResult
inputs (not real detector output), so the ACCEPT/REJECT boundary logic is
tested deterministically regardless of mock-heuristic calibration.
"""
from __future__ import annotations

import pytest

from detection.base import DetectionResult
from detection.fusion import DEFAULT_WEIGHTS, fuse, normalise

ALL_GENUINE = [
    DetectionResult(module="presentation_attack", score=0.9, is_mock=True),
    DetectionResult(module="morph", score=0.9, is_mock=True),
    DetectionResult(module="deepfake", score=0.9, is_mock=True),
    DetectionResult(module="replay", score=1.0, is_mock=False),
    DetectionResult(module="injection", score=0.9, is_mock=True),
]


def test_all_high_scores_are_accepted():
    result = fuse(ALL_GENUINE)
    assert result.accepted is True
    assert result.below_threshold_modules == []


def test_all_low_scores_are_rejected():
    results = [DetectionResult(module=r.module, score=0.1, is_mock=r.is_mock) for r in ALL_GENUINE]
    result = fuse(results)
    assert result.accepted is False
    assert result.below_threshold_modules == sorted(r.module for r in ALL_GENUINE)


def test_single_module_veto_below_per_module_threshold():
    """A confident single-module attack signal (e.g. an exact replay hash
    match) must reject even if the weighted average alone would pass."""
    results = [
        DetectionResult(module="presentation_attack", score=0.95, is_mock=True),
        DetectionResult(module="morph", score=0.95, is_mock=True),
        DetectionResult(module="deepfake", score=0.95, is_mock=True),
        DetectionResult(module="replay", score=0.0, is_mock=False),  # exact replay match
        DetectionResult(module="injection", score=0.95, is_mock=True),
    ]
    thresholds = {m: 0.5 for m in DEFAULT_WEIGHTS}
    result = fuse(results, per_module_thresholds=thresholds)
    assert result.fused_score >= 0.5  # the weighted mean alone would pass
    assert result.accepted is False  # but the veto still rejects
    assert result.below_threshold_modules == ["replay"]


def test_fused_score_is_weighted_average():
    weights = {"presentation_attack": 1.0}
    results = [DetectionResult(module="presentation_attack", score=0.42, is_mock=True)]
    result = fuse(results, weights=weights)
    assert result.fused_score == pytest.approx(0.42)


def test_score_at_threshold_passes_and_just_below_vetoes():
    weights = {"presentation_attack": 1.0}
    thresholds = {"presentation_attack": 0.57}
    at = [DetectionResult(module="presentation_attack", score=0.57, is_mock=False)]
    below = [DetectionResult(module="presentation_attack", score=0.5699, is_mock=False)]
    assert fuse(at, weights=weights, per_module_thresholds=thresholds).accepted is True
    assert fuse(below, weights=weights, per_module_thresholds=thresholds).accepted is False


def test_decision_ignores_the_mean():
    """The mean is reported only. A high mean never rescues a vetoed
    submission, and nothing but a veto can reject."""
    thresholds = {"presentation_attack": 0.57, "morph": 0.5, "deepfake": 0.48, "replay": 0.5, "injection": 0.35}
    one_fails = [DetectionResult(module=m, score=1.0, is_mock=False) for m in thresholds]
    one_fails[4] = DetectionResult(module="injection", score=0.34, is_mock=False)
    result = fuse(one_fails, per_module_thresholds=thresholds)
    assert result.fused_score > 0.9
    assert result.accepted is False and result.below_threshold_modules == ["injection"]

    all_just_pass = [DetectionResult(module=m, score=t, is_mock=False) for m, t in thresholds.items()]
    result = fuse(all_just_pass, per_module_thresholds=thresholds)
    assert result.accepted is True
    assert result.fused_score == pytest.approx(0.5)


def test_veto_only_matches_v1_rule_when_all_thresholds_set():
    """v1.0 decided 'normalised mean >= 0.5 AND no veto'. With every module's
    threshold set (as the server always does), that is identical to veto-only,
    so experiments run under v1.0 keep their decisions."""
    import random

    rng = random.Random(0)
    thresholds = {"presentation_attack": 0.57, "morph": 0.5, "deepfake": 0.48, "replay": 0.5, "injection": 0.35}
    for _ in range(5000):
        results = [DetectionResult(module=m, score=rng.random(), is_mock=False) for m in thresholds]
        new = fuse(results, per_module_thresholds=thresholds)
        norm = {r.module: normalise(r.score, thresholds[r.module]) for r in results}
        mean = sum(norm[m] * DEFAULT_WEIGHTS[m] for m in norm) / sum(DEFAULT_WEIGHTS.values())
        old = mean >= 0.5 and not any(r.score < thresholds[r.module] for r in results)
        assert new.accepted == old


def test_empty_results_raises():
    with pytest.raises(ValueError):
        fuse([])


def test_per_module_scores_and_is_mock_are_reported():
    result = fuse(ALL_GENUINE)
    assert result.per_module_scores["replay"] == 1.0
    assert result.per_module_is_mock["replay"] is False
    assert result.per_module_is_mock["presentation_attack"] is True


def test_deployed_injection_threshold_matches_the_trained_model():
    """The veto threshold must be the operating point stored with the trained
    classifier (BPCER = 5%), not a stale hand-copied value."""
    import json

    from detection.injection_detector import WEIGHTS_PATH
    from server.config import Settings

    if not WEIGHTS_PATH.exists():
        pytest.skip("injection weights not present (gitignored; run evaluation/train_injection.py)")
    stored = json.loads(WEIGHTS_PATH.read_text())["threshold_bpcer5"]
    assert Settings.model_fields["threshold_injection"].default == pytest.approx(round(stored, 2))
