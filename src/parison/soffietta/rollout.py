from __future__ import annotations

import torch
from torch import Tensor

from parison.furnace.twin import TherapyTwin
from parison.gather.streams import Parcel


def encode_level(
    level: int, action_dim: int, action_levels: int, batch: int, device: torch.device
) -> Tensor:
    vec = torch.zeros(batch, action_dim, device=device)
    vec[:, 0] = level / max(1, action_levels - 1)
    slot = 1 + (level % (action_dim - 1))
    vec[:, slot] = 1.0
    return vec


@torch.no_grad()
def roll_forward(model: TherapyTwin, parcel: Parcel, actions_future: Tensor) -> Tensor:
    model.eval()
    fused, states = model.prime(parcel)
    bsz = fused.shape[0]
    presence = torch.ones(bsz, len(model.modalities), device=fused.device)
    last: dict[str, Tensor] = {m: parcel.waveforms[m][:, -1].clone() for m in model.modalities}
    horizon = actions_future.shape[1]
    traj = []
    for h in range(horizon):
        icp_next = model.forecast_head(fused)[:, 0]
        traj.append(icp_next)
        tokens = dict(last)
        tokens["icp"] = icp_next.unsqueeze(-1)
        fused, states = model.advance(tokens, actions_future[:, h], presence, states)
        last["icp"] = icp_next.unsqueeze(-1)
    return torch.stack(traj, dim=1)
