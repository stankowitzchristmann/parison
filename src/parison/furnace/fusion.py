from __future__ import annotations

import torch
from torch import Tensor, nn


class CoincidenceGate(nn.Module):
    def __init__(self, n_modalities: int, d_model: int, hidden: int, gated: bool = True) -> None:
        super().__init__()
        self.gated = gated
        self.scorers = nn.ModuleList(
            nn.Sequential(nn.Linear(d_model, hidden), nn.Tanh(), nn.Linear(hidden, 1))
            for _ in range(n_modalities)
        )

    def forward(self, streams: Tensor, presence: Tensor) -> tuple[Tensor, Tensor]:
        mask = presence.to(streams.dtype)
        if self.gated:
            scores = torch.stack(
                [self.scorers[m](streams[:, :, m]).squeeze(-1) for m in range(len(self.scorers))],
                dim=-1,
            )
            shifted = scores - scores.amax(dim=-1, keepdim=True)
            weights = torch.exp(shifted) * mask
        else:
            weights = mask
        denom = weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        weights = weights / denom
        fused = (weights.unsqueeze(-1) * streams).sum(dim=2)
        return fused, weights
