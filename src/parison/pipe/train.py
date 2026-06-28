from __future__ import annotations

from pathlib import Path

import torch

from parison.bench.logbook import scribe
from parison.bench.settings import load_recipe
from parison.lehr.trainer import Glassworks

_LOG = scribe("parison.train")


def run(recipe_path: str, out: str, overrides: list[str], epochs: int) -> None:
    recipe = load_recipe(recipe_path, overrides)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    workdir = Path(out)
    works = Glassworks(recipe, device, workdir)
    _LOG.info(
        "recipe=%s effective_batch=%d device=%s",
        recipe.name,
        recipe.bench.effective_batch,
        device,
    )
    best = works.fit(None if epochs <= 0 else epochs)
    _LOG.info("best validation objective %.5f -> %s", best, workdir / "best.ckpt")
