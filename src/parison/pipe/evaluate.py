from __future__ import annotations

from pathlib import Path

import torch

from parison.bench.logbook import scribe
from parison.bench.settings import load_recipe
from parison.furnace.twin import TherapyTwin
from parison.gather.windows import Foundry, Norm
from parison.lehr.ledger import revive
from parison.proof.harness import evaluate

_LOG = scribe("parison.evaluate")


def run(recipe_path: str, ckpt: str, overrides: list[str]) -> None:
    recipe = load_recipe(recipe_path, overrides)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TherapyTwin(recipe.glass, recipe.feed).to(device)
    payload = revive(Path(ckpt), map_location=str(device))
    model.load_state_dict(payload["model"])
    val = Foundry(recipe.feed, recipe.glass, recipe.bench.seed, "val")
    if "norm" in payload.get("extra", {}):
        val.attach_norm(Norm(payload["extra"]["norm"]))
    report = evaluate(model, val, device)
    for key in sorted(report):
        _LOG.info("%-28s %.5f", key, report[key])
