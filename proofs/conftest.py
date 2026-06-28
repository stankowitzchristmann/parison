from __future__ import annotations

import numpy as np
import pytest
import torch

from parison.bench.settings import FeedCfg, GlassCfg, load_recipe
from parison.gather.streams import Parcel, collate
from parison.gather.windows import Foundry


@pytest.fixture(autouse=True)
def _determinism() -> None:
    torch.manual_seed(1234)
    np.random.seed(1234)


@pytest.fixture()
def tiny_glass() -> GlassCfg:
    return GlassCfg(
        d_model=24,
        d_state=8,
        d_conv=4,
        expand=2,
        depth=2,
        dt_rank=4,
        modalities=["icp", "abp", "ehr"],
        action_dim=5,
        action_levels=5,
        horizon=6,
        lead_window=6,
        fuse_hidden=16,
    )


@pytest.fixture()
def tiny_feed() -> FeedCfg:
    return FeedCfg(
        window_steps=32,
        stride_steps=8,
        sustained_steps=4,
        ehr_dim=6,
        cohort=16,
        drop_rate=0.1,
    )


@pytest.fixture()
def tiny_batch(tiny_glass: GlassCfg, tiny_feed: FeedCfg) -> Parcel:
    ds = Foundry(tiny_feed, tiny_glass, 0, "train")
    return collate([ds[i] for i in range(4)])


@pytest.fixture()
def check_recipe():
    return load_recipe("batches/bench_check.toml")
