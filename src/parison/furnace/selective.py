from __future__ import annotations

import math
from typing import NamedTuple

import torch
from torch import Tensor, nn

from parison.furnace.conditioning import ActionTint, ConditionedSelection


class StreamState(NamedTuple):
    ssm: Tensor
    conv: Tensor


def _zoh(delta: Tensor, amat: Tensor, b_sel: Tensor) -> tuple[Tensor, Tensor]:
    abar = torch.exp(delta.unsqueeze(-1) * amat)
    bbar = delta.unsqueeze(-1) * b_sel.unsqueeze(-2)
    return abar, bbar


class SelectiveStream(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int,
        d_conv: int,
        expand: int,
        dt_rank: int,
        action_dim: int,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = expand * d_model
        action_width = max(8, d_state)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=True,
        )
        self.tint = ActionTint(action_dim, action_width)
        self.select = ConditionedSelection(self.d_inner, d_state, dt_rank, action_width)
        self.skip = nn.Parameter(torch.ones(self.d_inner))
        amp = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.a_log = nn.Parameter(torch.log(amp))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.dt_bias = nn.Parameter(torch.full((self.d_inner,), math.log(math.expm1(1.0))))

    def _amat(self) -> Tensor:
        return -torch.exp(self.a_log)

    def forward(self, seq: Tensor, actions: Tensor) -> tuple[Tensor, StreamState]:
        bsz, length, _ = seq.shape
        xz = self.in_proj(seq)
        x_in, gate = xz.chunk(2, dim=-1)
        conv_in = x_in.transpose(1, 2)
        conv_out = self.conv(conv_in)[..., :length]
        u = nn.functional.silu(conv_out.transpose(1, 2))
        tint = self.tint(actions)
        delta, b_sel, c_sel = self.select(u, tint)
        amat = self._amat()
        abar, bbar = _zoh(delta, amat, b_sel)
        carry = seq.new_zeros(bsz, self.d_inner, self.d_state)
        outs = []
        for t in range(length):
            carry = abar[:, t] * carry + bbar[:, t] * u[:, t].unsqueeze(-1)
            y_t = torch.einsum("bdn,bn->bd", carry, c_sel[:, t]) + self.skip * u[:, t]
            outs.append(y_t)
        stacked = torch.stack(outs, dim=1)
        gated = stacked * nn.functional.silu(gate)
        out = self.out_proj(gated)
        conv_tail = conv_in[..., max(0, length - (self.d_conv - 1)) :]
        return out, StreamState(ssm=carry, conv=conv_tail)

    def prime(self, seq: Tensor, actions: Tensor) -> StreamState:
        _, state = self.forward(seq, actions)
        return state

    def step(self, token: Tensor, action: Tensor, state: StreamState) -> tuple[Tensor, StreamState]:
        xz = self.in_proj(token)
        x_in, gate = xz.chunk(2, dim=-1)
        window = torch.cat([state.conv, x_in.unsqueeze(-1)], dim=-1)
        if window.shape[-1] < self.d_conv:
            window = nn.functional.pad(window, (self.d_conv - window.shape[-1], 0))
        window = window[..., -self.d_conv :]
        weight = self.conv.weight[:, 0, :]
        bias = self.conv.bias
        assert bias is not None
        conv_out = (window * weight.unsqueeze(0)).sum(-1) + bias
        u = nn.functional.silu(conv_out)
        tint = self.tint(action)
        delta, b_sel, c_sel = self.select(u, tint)
        amat = self._amat()
        abar, bbar = _zoh(delta, amat, b_sel)
        carry = abar * state.ssm + bbar * u.unsqueeze(-1)
        y = torch.einsum("bdn,bn->bd", carry, c_sel) + self.skip * u
        gated = y * nn.functional.silu(gate)
        out = self.out_proj(gated)
        new_conv = window[..., -(self.d_conv - 1) :] if self.d_conv > 1 else state.conv
        return out, StreamState(ssm=carry, conv=new_conv)
