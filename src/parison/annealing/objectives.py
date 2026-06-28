from __future__ import annotations

from torch import Tensor, nn

from parison.bench.settings import OvenCfg
from parison.furnace.heads import TwinOutputs
from parison.gather.streams import Parcel


def _masked_huber(pred: Tensor, target: Tensor, valid: Tensor, delta: float) -> Tensor:
    per = nn.functional.huber_loss(pred, target, delta=delta, reduction="none")
    weight = valid.unsqueeze(-1)
    return (per * weight).sum() / weight.sum().clamp_min(1.0) / pred.shape[-1]


def _masked_bce(logits: Tensor, target: Tensor, valid: Tensor) -> Tensor:
    per = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (per * valid).sum() / valid.sum().clamp_min(1.0)


def kiln_loss(outputs: TwinOutputs, batch: Parcel, cfg: OvenCfg) -> tuple[Tensor, dict[str, float]]:
    forecast = _masked_huber(outputs.forecast, batch.future_icp, batch.valid, cfg.huber_delta)
    warning = _masked_bce(outputs.warning, batch.ih_label, batch.valid)
    outcome = nn.functional.binary_cross_entropy_with_logits(outputs.outcome, batch.outcome_label)
    total = (
        cfg.forecast_weight * forecast + cfg.warning_weight * warning + cfg.outcome_weight * outcome
    )
    report = {
        "forecast": float(forecast.detach()),
        "warning": float(warning.detach()),
        "outcome": float(outcome.detach()),
        "total": float(total.detach()),
    }
    return total, report
