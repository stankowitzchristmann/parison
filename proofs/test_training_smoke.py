from __future__ import annotations

from pathlib import Path

import torch

from parison.gather.streams import collate
from parison.lehr.trainer import Glassworks


def test_two_steps_reduce_loss(check_recipe, tmp_path: Path) -> None:
    works = Glassworks(check_recipe, torch.device("cpu"), tmp_path)
    batch = collate([works.train_set[i] for i in range(8)])
    first = works.run_step(batch)["total"]
    last = first
    for _ in range(6):
        last = works.run_step(batch)["total"]
    assert last < first


def test_fit_writes_checkpoint(check_recipe, tmp_path: Path) -> None:
    works = Glassworks(check_recipe, torch.device("cpu"), tmp_path)
    best = works.fit(max_epochs=2)
    assert (tmp_path / "best.ckpt").exists()
    assert best < float("inf")
