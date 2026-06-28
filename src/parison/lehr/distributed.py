from __future__ import annotations

import os

import torch
import torch.distributed as dist


def afoot() -> bool:
    return dist.is_available() and dist.is_initialized()


def ignite() -> tuple[int, int, torch.device]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        rank = dist.get_rank()
        world = dist.get_world_size()
        if torch.cuda.is_available():
            local = int(os.environ.get("LOCAL_RANK", rank))
            torch.cuda.set_device(local)
            return rank, world, torch.device(f"cuda:{local}")
        return rank, world, torch.device("cpu")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return 0, 1, device


def principal() -> bool:
    return not afoot() or dist.get_rank() == 0


def douse() -> None:
    if afoot():
        dist.destroy_process_group()
