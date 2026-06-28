from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor, nn


class TwinOutputs(NamedTuple):
    forecast: Tensor
    warning: Tensor
    outcome: Tensor


class ForecastHead(nn.Module):
    def __init__(self, d_model: int, horizon: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, horizon),
        )

    def forward(self, latent: Tensor) -> Tensor:
        return self.net(latent)


class WarningHead(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.SiLU(), nn.Linear(d_model // 2, 1)
        )

    def forward(self, latent: Tensor) -> Tensor:
        return self.net(latent).squeeze(-1)


class OutcomeHead(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.pool = nn.Linear(d_model, d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, sequence: Tensor) -> Tensor:
        summary = torch.tanh(self.pool(sequence.mean(dim=1)))
        return self.head(summary).squeeze(-1)
