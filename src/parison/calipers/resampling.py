from __future__ import annotations

from typing import Callable

import numpy as np


def bootstrap_ci(
    scores: np.ndarray,
    label: np.ndarray,
    statistic: Callable[[np.ndarray, np.ndarray], float],
    rounds: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(label)
    point = statistic(scores, label)
    draws: list[float] = []
    for _ in range(rounds):
        idx = rng.integers(0, n, size=n)
        if label[idx].sum() in (0, label[idx].size):
            continue
        draws.append(statistic(scores[idx], label[idx]))
    lo = float(np.quantile(draws, alpha / 2.0))
    hi = float(np.quantile(draws, 1.0 - alpha / 2.0))
    return point, lo, hi


def holm_bonferroni(pvalues: np.ndarray) -> np.ndarray:
    order = np.argsort(pvalues)
    m = len(pvalues)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvalues[idx]
        running = max(running, val)
        adjusted[idx] = min(running, 1.0)
    return adjusted


def benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    order = np.argsort(pvalues)
    m = len(pvalues)
    adjusted = np.empty(m)
    running = 1.0
    for rank in range(m - 1, -1, -1):
        idx = order[rank]
        val = m / (rank + 1) * pvalues[idx]
        running = min(running, val)
        adjusted[idx] = min(running, 1.0)
    return adjusted
