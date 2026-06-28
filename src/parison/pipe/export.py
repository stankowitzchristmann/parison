from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from parison.bench.logbook import scribe
from parison.bench.settings import Recipe, load_recipe
from parison.furnace.twin import TherapyTwin

_LOG = scribe("parison.export")


class GlassPort(nn.Module):
    def __init__(self, twin: TherapyTwin) -> None:
        super().__init__()
        self.twin = twin
        self.modalities = list(twin.modalities)

    def forward(
        self, waves: Tensor, ehr: Tensor, presence: Tensor, actions: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        bag: dict[str, Tensor] = {}
        idx = 0
        for m in self.modalities:
            if m == "ehr":
                bag[m] = ehr
            else:
                bag[m] = waves[..., idx : idx + 1]
                idx += 1
        fused, _ = self.twin.fuse(bag, presence, actions)
        return (
            self.twin.forecast_head(fused),
            self.twin.warning_head(fused),
            self.twin.outcome_head(fused),
        )


def _dummy(recipe: Recipe) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    g = recipe.glass
    f = recipe.feed
    steps = 32
    n_wave = sum(1 for m in g.modalities if m != "ehr")
    waves = torch.randn(1, steps, n_wave)
    ehr = torch.randn(1, steps, f.ehr_dim)
    presence = torch.ones(1, steps, len(g.modalities))
    actions = torch.randn(1, steps, g.action_dim)
    return waves, ehr, presence, actions


def run(recipe_path: str, out: str, overrides: list[str]) -> None:
    recipe = load_recipe(recipe_path, overrides)
    twin = TherapyTwin(recipe.glass, recipe.feed).eval()
    port = GlassPort(twin).eval()
    args = _dummy(recipe)
    kwargs: dict[str, Any] = {
        "input_names": ["waves", "ehr", "presence", "actions"],
        "output_names": ["forecast", "warning", "outcome"],
        "opset_version": 13,
        "dynamic_axes": {"waves": {1: "time"}, "ehr": {1: "time"}},
    }
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        kwargs["dynamo"] = False
    torch.onnx.export(port, args, out, **kwargs)
    _LOG.info("exported ONNX graph -> %s", Path(out))
