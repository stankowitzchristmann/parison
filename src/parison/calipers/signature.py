from __future__ import annotations

import numpy as np
from scipy import stats


def _smooth(x: np.ndarray, span: int) -> np.ndarray:
    if span <= 1:
        return x
    kernel = np.ones(span) / span
    return np.convolve(x, kernel, mode="same")


def prx(icp: np.ndarray, abp: np.ndarray, span: int = 10) -> float:
    si = _smooth(icp, span)
    sa = _smooth(abp, span)
    r, _ = stats.pearsonr(sa, si)
    return float(r)


def rap(amp: np.ndarray, icp: np.ndarray, span: int = 10) -> float:
    mean_icp = _smooth(icp, span)
    r, _ = stats.pearsonr(amp, mean_icp)
    return float(r)


def two_axis_embedding(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = coords - coords.mean(axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(centered, full_matrices=False)
    scores = u[:, :2] * s[:2]
    explained = (s**2) / np.sum(s**2)
    return scores, explained[:2]


def axis_alignment(axis: np.ndarray, target: np.ndarray) -> float:
    r, _ = stats.spearmanr(axis, target)
    return float(abs(r))


def r_squared(pred: np.ndarray, actual: np.ndarray) -> float:
    ss_res = float(np.sum((actual - pred) ** 2))
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))
    return 1.0 - ss_res / max(ss_tot, 1e-12)
