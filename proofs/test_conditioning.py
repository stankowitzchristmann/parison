from __future__ import annotations

import torch

from parison.bench.settings import FeedCfg, GlassCfg
from parison.furnace.twin import TherapyTwin
from parison.gather.streams import Parcel, collate
from parison.gather.windows import Foundry


def _batch(glass: GlassCfg, feed: FeedCfg) -> Parcel:
    ds = Foundry(feed, glass, 3, "train")
    return collate([ds[i] for i in range(4)])


def test_action_changes_selection(tiny_glass: GlassCfg, tiny_feed: FeedCfg) -> None:
    model = TherapyTwin(tiny_glass, tiny_feed).eval()
    batch = _batch(tiny_glass, tiny_feed)
    with torch.no_grad():
        base, _ = model(batch)
        shifted = batch._replace(actions=torch.zeros_like(batch.actions))
        other, _ = model(shifted)
    assert not torch.allclose(base.forecast, other.forecast, atol=1e-4)


def test_disabled_conditioning_ignores_action(tiny_feed: FeedCfg) -> None:
    glass = GlassCfg(
        d_model=24,
        d_state=8,
        depth=2,
        dt_rank=4,
        modalities=["icp", "abp", "ehr"],
        action_dim=5,
        horizon=6,
        lead_window=6,
        condition=False,
    )
    model = TherapyTwin(glass, tiny_feed).eval()
    batch = _batch(glass, tiny_feed)
    with torch.no_grad():
        base, _ = model(batch)
        shifted = batch._replace(actions=batch.actions + 3.0)
        other, _ = model(shifted)
    assert torch.allclose(base.forecast, other.forecast, atol=1e-6)
