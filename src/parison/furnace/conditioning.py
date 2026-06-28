from __future__ import annotations

import torch
from torch import Tensor, nn


class ActionTint(nn.Module):
    def __init__(self, action_dim: int, width: int) -> None:
        super().__init__()
        self.lift = nn.Sequential(
            nn.Linear(action_dim, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )

    def forward(self, actions: Tensor) -> Tensor:
        return self.lift(actions)


class ConditionedSelection(nn.Module):
    def __init__(self, d_inner: int, d_state: int, dt_rank: int, action_width: int) -> None:
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        self.dt_rank = dt_rank
        self.project = nn.Linear(d_inner + action_width, dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

    def forward(self, u: Tensor, tint: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        joined = torch.cat([u, tint], dim=-1)
        parts = self.project(joined)
        dt_low, b_sel, c_sel = torch.split(
            parts, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        delta = nn.functional.softplus(self.dt_proj(dt_low))
        return delta, b_sel, c_sel
