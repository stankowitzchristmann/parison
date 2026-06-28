from __future__ import annotations

import numpy as np

from parison.calipers.calibration import (
    brier,
    calibration_slope,
    expected_calibration_error,
)
from parison.calipers.decision import (
    decision_curve,
    integrated_discrimination,
    net_benefit,
    net_reclassification,
)
from parison.calipers.discrimination import (
    auprc,
    auroc,
    auroc_ci,
    delong_test,
    recall_at_specificity,
)
from parison.calipers.resampling import (
    benjamini_hochberg,
    bootstrap_ci,
    holm_bonferroni,
)


def _ref_auc(scores: np.ndarray, label: np.ndarray) -> float:
    pos = scores[label == 1]
    neg = scores[label == 0]
    wins = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(wins) / (len(pos) * len(neg))


def test_auroc_matches_mann_whitney() -> None:
    rng = np.random.default_rng(0)
    scores = rng.normal(size=400)
    label = (rng.random(400) < 1.0 / (1.0 + np.exp(-scores))).astype(int)
    assert abs(auroc(scores, label) - _ref_auc(scores, label)) < 1e-9


def test_delong_identical_predictors_high_p() -> None:
    rng = np.random.default_rng(1)
    scores = rng.normal(size=200)
    label = (rng.random(200) < 0.4).astype(int)
    _, _, p = delong_test(scores, scores.copy(), label)
    assert p > 0.99


def test_auroc_ci_brackets_point() -> None:
    rng = np.random.default_rng(2)
    scores = rng.normal(size=300)
    label = (rng.random(300) < 1.0 / (1.0 + np.exp(-scores))).astype(int)
    point, lo, hi = auroc_ci(scores, label)
    assert lo <= point <= hi


def test_auprc_perfect_ranking() -> None:
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    label = np.array([1, 1, 0, 0])
    assert auprc(scores, label) > 0.99


def test_recall_at_specificity_threshold() -> None:
    scores = np.concatenate([np.linspace(0, 0.4, 50), np.linspace(0.6, 1.0, 50)])
    label = np.concatenate([np.zeros(50), np.ones(50)]).astype(int)
    recall, _ = recall_at_specificity(scores, label, 0.9)
    assert recall == 1.0


def test_calibration_slope_near_one_when_calibrated() -> None:
    rng = np.random.default_rng(3)
    prob = rng.uniform(0.05, 0.95, size=4000)
    label = (rng.random(4000) < prob).astype(float)
    slope, _ = calibration_slope(prob, label)
    assert 0.8 < slope < 1.2
    assert expected_calibration_error(prob, label) < 0.05
    assert brier(prob, label) < 0.25


def test_holm_dominates_bh() -> None:
    p = np.array([0.001, 0.02, 0.2, 0.5])
    assert np.all(holm_bonferroni(p) >= benjamini_hochberg(p) - 1e-12)


def test_decision_curve_treat_all_formula() -> None:
    rng = np.random.default_rng(4)
    label = (rng.random(500) < 0.2).astype(int)
    prob = rng.uniform(size=500)
    curve = decision_curve(prob, label, np.array([0.1, 0.2]))
    prevalence = label.mean()
    expected = prevalence - (1 - prevalence) * (0.1 / 0.9)
    assert abs(curve["treat_all"][0] - expected) < 1e-9


def test_idi_and_nri_reward_better_model() -> None:
    label = np.array([1, 1, 1, 0, 0, 0])
    old = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    new = np.array([0.9, 0.8, 0.7, 0.1, 0.2, 0.3])
    assert integrated_discrimination(old, new, label) > 0
    assert net_reclassification(old, new, label) > 0


def test_bootstrap_contains_point() -> None:
    rng = np.random.default_rng(5)
    scores = rng.normal(size=300)
    label = (rng.random(300) < 1.0 / (1.0 + np.exp(-scores))).astype(int)
    point, lo, hi = bootstrap_ci(scores, label, auroc, rounds=300, seed=5)
    assert lo <= point <= hi


def test_net_benefit_non_positive_for_random_high_threshold() -> None:
    rng = np.random.default_rng(6)
    label = (rng.random(400) < 0.1).astype(int)
    prob = rng.uniform(size=400)
    assert net_benefit(prob, label, 0.5) <= 0.05
