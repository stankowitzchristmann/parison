from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from parison.bench.logbook import scribe
from parison.bench.settings import load_recipe
from parison.furnace.twin import TherapyTwin
from parison.gather.streams import collate
from parison.gather.windows import Foundry, Norm
from parison.lehr.ledger import revive
from parison.soffietta.optimizer import best_intervention

_LOG = scribe("parison.infer")


def run(recipe_path: str, ckpt: str, overrides: list[str]) -> None:
    recipe = load_recipe(recipe_path, overrides)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TherapyTwin(recipe.glass, recipe.feed).to(device)
    payload = revive(Path(ckpt), map_location=str(device))
    model.load_state_dict(payload["model"])
    val = Foundry(recipe.feed, recipe.glass, recipe.bench.seed, "val")
    if "norm" in payload.get("extra", {}):
        val.attach_norm(Norm(payload["extra"]["norm"]))
    batch = next(iter(DataLoader(val, batch_size=8, collate_fn=collate))).to(device)
    verdict = best_intervention(model, batch, recipe.glass.action_levels, recipe.glass.horizon)
    for i, lvl in enumerate(verdict.level.tolist()):
        _LOG.info(
            "stay %d chosen_level=%d predicted_icp=%.3f",
            i,
            lvl,
            float(verdict.scores[i, lvl]),
        )
