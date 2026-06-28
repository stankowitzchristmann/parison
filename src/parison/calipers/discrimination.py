from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    fn: Any = getattr(np, "trapezoid", getattr(np, "trapz", None))
    return float(fn(y, x))


def _midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    sorted_x = x[order]
    n = len(x)
    ranks = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n)
    out[order] = ranks
    return out


def _delong(predictions: np.ndarray, label: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = (-label).argsort(kind="mergesort")
    m = int(label.sum())
    n = predictions.shape[1] - m
    if m == 0 or n == 0:
        raise ValueError("both classes required for DeLong")
    ordered = predictions[:, order]
    k = predictions.shape[0]
    pos = ordered[:, :m]
    neg = ordered[:, m:]
    tx = np.stack([_midrank(pos[r]) for r in range(k)])
    ty = np.stack([_midrank(neg[r]) for r in range(k)])
    tz = np.stack([_midrank(ordered[r]) for r in range(k)])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01).reshape(k, k)
    sy = np.cov(v10).reshape(k, k)
    cov = sx / m + sy / n
    return aucs, cov


def auroc(scores: np.ndarray, label: np.ndarray) -> float:
    aucs, _ = _delong(scores[None, :], label.astype(float))
    return float(aucs[0])


def auroc_ci(
    scores: np.ndarray, label: np.ndarray, alpha: float = 0.05
) -> tuple[float, float, float]:
    aucs, cov = _delong(scores[None, :], label.astype(float))
    se = float(np.sqrt(max(cov[0, 0], 0.0)))
    z = stats.norm.ppf(1.0 - alpha / 2.0)
    point = float(aucs[0])
    return point, max(0.0, point - z * se), min(1.0, point + z * se)


def delong_test(
    scores_a: np.ndarray, scores_b: np.ndarray, label: np.ndarray
) -> tuple[float, float, float]:
    aucs, cov = _delong(np.stack([scores_a, scores_b]), label.astype(float))
    var = cov[0, 0] + cov[1, 1] - 2.0 * cov[0, 1]
    if var <= 0:
        return float(aucs[0]), float(aucs[1]), 1.0
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    p = float(2.0 * stats.norm.sf(abs(z)))
    return float(aucs[0]), float(aucs[1]), p


def auprc(scores: np.ndarray, label: np.ndarray) -> float:
    order = np.argsort(-scores)
    y = label[order].astype(float)
    tp = np.cumsum(y)
    fp = np.cumsum(1.0 - y)
    total_pos = y.sum()
    precision = tp / np.maximum(tp + fp, 1.0)
    recall = tp / max(total_pos, 1.0)
    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[1.0], precision])
    return _trapz(precision, recall)


def recall_at_specificity(
    scores: np.ndarray, label: np.ndarray, specificity: float
) -> tuple[float, float]:
    neg = scores[label == 0]
    pos = scores[label == 1]
    if len(neg) == 0 or len(pos) == 0:
        return 0.0, 0.0
    threshold = float(np.quantile(neg, specificity))
    recall = float((pos >= threshold).mean())
    return recall, threshold
