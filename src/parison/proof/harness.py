from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from parison.calipers.calibration import (
    brier,
    calibration_slope,
    expected_calibration_error,
)
from parison.calipers.decision import decision_curve
from parison.calipers.discrimination import (
    auprc,
    auroc,
    auroc_ci,
    recall_at_specificity,
)
from parison.gather.streams import Parcel, collate
from parison.gather.windows import Foundry, Norm


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@torch.no_grad()
def gather_predictions(
    model: torch.nn.Module, dataset: Foundry, device: torch.device
) -> dict[str, np.ndarray]:
    model.eval()
    loader: DataLoader[Parcel] = DataLoader(dataset, batch_size=16, collate_fn=collate)
    warn_logits: list[np.ndarray] = []
    warn_labels: list[np.ndarray] = []
    out_logits: list[np.ndarray] = []
    out_labels: list[np.ndarray] = []
    fc_abs: list[np.ndarray] = []
    norm: Norm = dataset.norm
    for batch in loader:
        batch = batch.to(device)
        outputs, _ = model(batch)
        valid = batch.valid.bool().reshape(-1).cpu().numpy()
        warn_logits.append(outputs.warning.reshape(-1).cpu().numpy()[valid])
        warn_labels.append(batch.ih_label.reshape(-1).cpu().numpy()[valid])
        out_logits.append(outputs.outcome.cpu().numpy())
        out_labels.append(batch.outcome_label.cpu().numpy())
        pred = norm.invert_icp(outputs.forecast.cpu()).numpy()
        target = norm.invert_icp(batch.future_icp.cpu()).numpy()
        vmask = batch.valid.cpu().numpy().astype(bool)
        fc_abs.append(np.abs(pred[vmask] - target[vmask]).reshape(-1))
    return {
        "warn_prob": _sigmoid(np.concatenate(warn_logits)),
        "warn_label": np.concatenate(warn_labels),
        "outcome_prob": _sigmoid(np.concatenate(out_logits)),
        "outcome_label": np.concatenate(out_labels),
        "forecast_abs": np.concatenate(fc_abs),
    }


def evaluate(model: torch.nn.Module, dataset: Foundry, device: torch.device) -> dict[str, float]:
    pred = gather_predictions(model, dataset, device)
    report: dict[str, float] = {"forecast_mae": float(pred["forecast_abs"].mean())}
    wl, wy = pred["warn_prob"], pred["warn_label"]
    if 0 < wy.sum() < len(wy):
        point, lo, hi = auroc_ci(wl, wy)
        recall, _ = recall_at_specificity(wl, wy, 0.90)
        report.update(
            ih_auroc=point,
            ih_auroc_lo=lo,
            ih_auroc_hi=hi,
            ih_auprc=auprc(wl, wy),
            ih_recall_at_90=recall,
        )
    ol, oy = pred["outcome_prob"], pred["outcome_label"]
    if 0 < oy.sum() < len(oy):
        report.update(
            outcome_auroc=auroc(ol, oy),
            outcome_brier=brier(ol, oy),
            outcome_ece=expected_calibration_error(ol, oy),
        )
        slope, intercept = calibration_slope(ol, oy)
        report["outcome_calibration_slope"] = slope
        report["outcome_calibration_intercept"] = intercept
        curve = decision_curve(ol, oy, np.linspace(0.05, 0.30, 6))
        report["net_benefit_at_0.15"] = float(
            curve["model"][np.argmin(np.abs(curve["threshold"] - 0.15))]
        )
    return report
