from __future__ import annotations

import numpy as np


def net_benefit(prob: np.ndarray, label: np.ndarray, threshold: float) -> float:
    n = len(label)
    flag = prob >= threshold
    tp = float(np.sum(flag & (label == 1)))
    fp = float(np.sum(flag & (label == 0)))
    odds = threshold / max(1.0 - threshold, 1e-9)
    return tp / n - fp / n * odds


def decision_curve(
    prob: np.ndarray, label: np.ndarray, thresholds: np.ndarray
) -> dict[str, np.ndarray]:
    prevalence = float(label.mean())
    treat_all = np.array(
        [prevalence - (1.0 - prevalence) * (t / max(1.0 - t, 1e-9)) for t in thresholds]
    )
    model = np.array([net_benefit(prob, label, float(t)) for t in thresholds])
    return {
        "threshold": thresholds,
        "model": model,
        "treat_all": treat_all,
        "treat_none": np.zeros_like(thresholds),
    }


def integrated_discrimination(old: np.ndarray, new: np.ndarray, label: np.ndarray) -> float:
    ev = label == 1
    ne = label == 0
    return float((new[ev].mean() - old[ev].mean()) - (new[ne].mean() - old[ne].mean()))


def net_reclassification(old: np.ndarray, new: np.ndarray, label: np.ndarray) -> float:
    ev = label == 1
    ne = label == 0
    up_ev = float(np.mean(new[ev] > old[ev]))
    down_ev = float(np.mean(new[ev] < old[ev]))
    up_ne = float(np.mean(new[ne] > old[ne]))
    down_ne = float(np.mean(new[ne] < old[ne]))
    return (up_ev - down_ev) + (down_ne - up_ne)
