from __future__ import annotations

import numpy as np


def brier(prob: np.ndarray, label: np.ndarray) -> float:
    return float(np.mean((prob - label) ** 2))


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def calibration_slope(prob: np.ndarray, label: np.ndarray, iters: int = 100) -> tuple[float, float]:
    x = _logit(prob)
    beta = np.array([0.0, 1.0])
    design = np.stack([np.ones_like(x), x], axis=1)
    for _ in range(iters):
        eta = design @ beta
        mu = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(mu * (1.0 - mu), 1e-6, None)
        grad = design.T @ (label - mu)
        hess = design.T @ (design * w[:, None])
        step = np.linalg.solve(hess + 1e-8 * np.eye(2), grad)
        beta = beta + step
        if np.max(np.abs(step)) < 1e-8:
            break
    return float(beta[1]), float(beta[0])


def expected_calibration_error(prob: np.ndarray, label: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(prob, edges[1:-1]), 0, bins - 1)
    total = len(prob)
    error = 0.0
    for b in range(bins):
        mask = idx == b
        count = int(mask.sum())
        if count == 0:
            continue
        conf = float(prob[mask].mean())
        acc = float(label[mask].mean())
        error += count / total * abs(conf - acc)
    return float(error)
