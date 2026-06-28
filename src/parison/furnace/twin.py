from __future__ import annotations

from typing import Iterable, cast

import torch
from torch import Tensor, nn
from torch.nn import ModuleList

from parison.bench.settings import FeedCfg, GlassCfg
from parison.furnace.fusion import CoincidenceGate
from parison.furnace.heads import ForecastHead, OutcomeHead, TwinOutputs, WarningHead
from parison.furnace.selective import SelectiveStream, StreamState
from parison.gather.streams import Parcel


def _input_dims(cfg: GlassCfg, feed: FeedCfg) -> dict[str, int]:
    return {m: feed.ehr_dim if m == "ehr" else feed.waveform_dim for m in cfg.modalities}


class TherapyTwin(nn.Module):
    def __init__(self, cfg: GlassCfg, feed: FeedCfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.modalities = list(cfg.modalities)
        dims = _input_dims(cfg, feed)
        self.embed = nn.ModuleDict({m: nn.Linear(dims[m], cfg.d_model) for m in self.modalities})
        self.norms = nn.ModuleDict(
            {
                m: nn.ModuleList(nn.LayerNorm(cfg.d_model) for _ in range(cfg.depth))
                for m in self.modalities
            }
        )
        self.streams = nn.ModuleDict(
            {
                m: nn.ModuleList(
                    SelectiveStream(
                        cfg.d_model,
                        cfg.d_state,
                        cfg.d_conv,
                        cfg.expand,
                        cfg.dt_rank,
                        cfg.action_dim,
                    )
                    for _ in range(cfg.depth)
                )
                for m in self.modalities
            }
        )
        self.gate = CoincidenceGate(
            len(self.modalities), cfg.d_model, cfg.fuse_hidden, cfg.gated_fusion
        )
        self.condition = cfg.condition
        self.patient_bias = nn.Parameter(torch.zeros(cfg.d_model))
        self.forecast_head = ForecastHead(cfg.d_model, cfg.horizon)
        self.warning_head = WarningHead(cfg.d_model)
        self.outcome_head = OutcomeHead(cfg.d_model)

    def _dose(self, actions: Tensor) -> Tensor:
        return actions if self.condition else torch.zeros_like(actions)

    def _layers(self, m: str) -> list[tuple[SelectiveStream, nn.LayerNorm]]:
        blocks = cast(Iterable[SelectiveStream], cast(ModuleList, self.streams[m]))
        norms = cast(Iterable[nn.LayerNorm], cast(ModuleList, self.norms[m]))
        return list(zip(blocks, norms))

    def _encode(self, waveforms: dict[str, Tensor], actions: Tensor) -> Tensor:
        dose = self._dose(actions)
        latents = []
        for m in self.modalities:
            h = self.embed[m](waveforms[m])
            for block, norm in self._layers(m):
                out, _ = block(norm(h), dose)
                h = h + out
            latents.append(h)
        return torch.stack(latents, dim=2)

    def fuse(
        self, waveforms: dict[str, Tensor], presence: Tensor, actions: Tensor
    ) -> tuple[Tensor, Tensor]:
        stacked = self._encode(waveforms, actions)
        fused, weights = self.gate(stacked, presence)
        return fused + self.patient_bias, weights

    def forward(self, batch: Parcel) -> tuple[TwinOutputs, Tensor]:
        fused, weights = self.fuse(batch.waveforms, batch.presence, batch.actions)
        outputs = TwinOutputs(
            forecast=self.forecast_head(fused),
            warning=self.warning_head(fused),
            outcome=self.outcome_head(fused),
        )
        return outputs, weights

    def prime(self, batch: Parcel) -> tuple[Tensor, dict[str, list[StreamState]]]:
        states: dict[str, list[StreamState]] = {}
        dose = self._dose(batch.actions)
        latents = []
        for m in self.modalities:
            h = self.embed[m](batch.waveforms[m])
            block_states: list[StreamState] = []
            for block, norm in self._layers(m):
                out, st = block(norm(h), dose)
                h = h + out
                block_states.append(st)
            states[m] = block_states
            latents.append(h[:, -1])
        stacked = torch.stack(latents, dim=1).unsqueeze(1)
        presence_last = batch.presence[:, -1:].clone()
        fused, _ = self.gate(stacked, presence_last)
        return fused.squeeze(1) + self.patient_bias, states

    def advance(
        self,
        tokens: dict[str, Tensor],
        action: Tensor,
        presence: Tensor,
        states: dict[str, list[StreamState]],
    ) -> tuple[Tensor, dict[str, list[StreamState]]]:
        dose = self._dose(action)
        latents = []
        new_states: dict[str, list[StreamState]] = {}
        for m in self.modalities:
            h = self.embed[m](tokens[m])
            block_states: list[StreamState] = []
            for (block, norm), prev in zip(self._layers(m), states[m]):
                out, st = block.step(norm(h), dose, prev)
                h = h + out
                block_states.append(st)
            new_states[m] = block_states
            latents.append(h)
        stacked = torch.stack(latents, dim=1).unsqueeze(1)
        fused, _ = self.gate(stacked, presence.unsqueeze(1))
        return fused.squeeze(1) + self.patient_bias, new_states

    def adapt_parameters(self) -> list[Tensor]:
        params: list[Tensor] = [self.patient_bias]
        params.extend(self.forecast_head.parameters())
        return params
