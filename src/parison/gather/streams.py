from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor


class Parcel(NamedTuple):
    waveforms: dict[str, Tensor]
    presence: Tensor
    actions: Tensor
    future_icp: Tensor
    ih_label: Tensor
    outcome_label: Tensor
    valid: Tensor

    def to(self, device: torch.device) -> Parcel:
        return Parcel(
            waveforms={k: v.to(device) for k, v in self.waveforms.items()},
            presence=self.presence.to(device),
            actions=self.actions.to(device),
            future_icp=self.future_icp.to(device),
            ih_label=self.ih_label.to(device),
            outcome_label=self.outcome_label.to(device),
            valid=self.valid.to(device),
        )


def collate(items: list[Parcel]) -> Parcel:
    keys = list(items[0].waveforms.keys())
    return Parcel(
        waveforms={k: torch.stack([it.waveforms[k] for it in items]) for k in keys},
        presence=torch.stack([it.presence for it in items]),
        actions=torch.stack([it.actions for it in items]),
        future_icp=torch.stack([it.future_icp for it in items]),
        ih_label=torch.stack([it.ih_label for it in items]),
        outcome_label=torch.stack([it.outcome_label for it in items]),
        valid=torch.stack([it.valid for it in items]),
    )
