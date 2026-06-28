from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor

from parison.furnace.twin import TherapyTwin
from parison.gather.streams import Parcel
from parison.soffietta.rollout import encode_level, roll_forward


class Verdict(NamedTuple):
    level: Tensor
    scores: Tensor
    trajectories: Tensor


@torch.no_grad()
def best_intervention(
    model: TherapyTwin, parcel: Parcel, action_levels: int, horizon: int
) -> Verdict:
    device = parcel.actions.device
    bsz = parcel.actions.shape[0]
    action_dim = model.cfg.action_dim
    per_level: list[Tensor] = []
    for level in range(action_levels):
        vec = encode_level(level, action_dim, action_levels, bsz, device)
        future = vec.unsqueeze(1).expand(bsz, horizon, action_dim).contiguous()
        per_level.append(roll_forward(model, parcel, future))
    trajectories = torch.stack(per_level, dim=1)
    scores = trajectories.mean(dim=2)
    chosen = torch.argmin(scores, dim=1)
    return Verdict(level=chosen, scores=scores, trajectories=trajectories)
