from __future__ import annotations

import torch
from torch import Tensor, nn

from parison.bench.settings import BenchCfg, OvenCfg
from parison.furnace.twin import TherapyTwin
from parison.gather.streams import Parcel


def _snapshot(params: list[Tensor]) -> list[Tensor]:
    return [p.detach().clone() for p in params]


def _restore(params: list[Tensor], saved: list[Tensor]) -> None:
    with torch.no_grad():
        for p, s in zip(params, saved):
            p.copy_(s)


def online_recalibrate(
    model: TherapyTwin,
    parcel: Parcel,
    bench: BenchCfg,
    oven: OvenCfg,
) -> dict[str, Tensor]:
    adapt = model.adapt_parameters()
    saved = _snapshot(adapt)
    frozen = [p for p in model.parameters() if all(p is not a for a in adapt)]
    prior = [p.requires_grad for p in frozen]
    for p in frozen:
        p.requires_grad_(False)

    opt = torch.optim.SGD(adapt, lr=bench.adapt_lr)
    window = min(bench.adapt_window, parcel.valid.shape[1])
    sliced = _tail(parcel, window)
    last_loss = torch.zeros(())
    for _ in range(bench.adapt_steps):
        opt.zero_grad()
        outputs, _ = model(sliced)
        one_step = outputs.forecast[..., 0]
        target = sliced.future_icp[..., 0]
        per = nn.functional.huber_loss(one_step, target, delta=oven.huber_delta, reduction="none")
        loss = (per * sliced.valid).sum() / sliced.valid.sum().clamp_min(1.0)
        loss.backward()
        opt.step()
        last_loss = loss.detach()

    coord = torch.cat([p.detach().reshape(-1) for p in adapt])
    _restore(adapt, saved)
    for p, flag in zip(frozen, prior):
        p.requires_grad_(flag)
    return {"coordinate": coord, "residual": last_loss}


def _tail(parcel: Parcel, window: int) -> Parcel:
    return Parcel(
        waveforms={k: v[:, -window:] for k, v in parcel.waveforms.items()},
        presence=parcel.presence[:, -window:],
        actions=parcel.actions[:, -window:],
        future_icp=parcel.future_icp[:, -window:],
        ih_label=parcel.ih_label[:, -window:],
        outcome_label=parcel.outcome_label,
        valid=parcel.valid[:, -window:],
    )
