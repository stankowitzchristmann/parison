from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import torch
from torch import nn


def stamp(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    seed: int,
    extra: dict[str, Any],
) -> None:
    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "seed": seed,
        "extra": extra,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".part")
    os.close(fd)
    torch.save(payload, tmp)
    os.replace(tmp, str(path))


def revive(path: Path, map_location: str = "cpu") -> dict[str, Any]:
    return torch.load(str(path), map_location=map_location, weights_only=False)
