from __future__ import annotations

import torch

from parison.annealing.recalibrate import online_recalibrate
from parison.bench.settings import BenchCfg, FeedCfg, GlassCfg, OvenCfg
from parison.furnace.twin import TherapyTwin
from parison.gather.streams import collate
from parison.gather.windows import Foundry


def _setup(tiny_glass: GlassCfg, tiny_feed: FeedCfg):
    ds = Foundry(tiny_feed, tiny_glass, 5, "train")
    batch = collate([ds[i] for i in range(4)])
    model = TherapyTwin(tiny_glass, tiny_feed)
    return model, batch


def test_recalibration_reduces_one_step_error(tiny_glass: GlassCfg, tiny_feed: FeedCfg) -> None:
    model, batch = _setup(tiny_glass, tiny_feed)
    oven = OvenCfg()
    with torch.no_grad():
        outputs, _ = model(batch)
        pre = torch.nn.functional.huber_loss(
            outputs.forecast[..., 0], batch.future_icp[..., 0], delta=oven.huber_delta
        )
    bench = BenchCfg(adapt_steps=30, adapt_lr=5e-2, adapt_window=24)
    result = online_recalibrate(model, batch, bench, oven)
    assert torch.isfinite(result["residual"])
    assert float(result["residual"]) < float(pre)


def test_backbone_unchanged_after_recalibration(tiny_glass: GlassCfg, tiny_feed: FeedCfg) -> None:
    model, batch = _setup(tiny_glass, tiny_feed)
    backbone = model.streams["icp"][0].out_proj.weight.detach().clone()
    head_before = next(iter(model.forecast_head.parameters())).detach().clone()
    bench = BenchCfg(adapt_steps=10, adapt_lr=1e-2, adapt_window=24)
    online_recalibrate(model, batch, bench, OvenCfg())
    after = model.streams["icp"][0].out_proj.weight.detach()
    head_after = next(iter(model.forecast_head.parameters())).detach()
    assert torch.allclose(backbone, after, atol=1e-7)
    assert torch.allclose(head_before, head_after, atol=1e-7)


def test_coordinate_is_low_dimensional(tiny_glass: GlassCfg, tiny_feed: FeedCfg) -> None:
    model, batch = _setup(tiny_glass, tiny_feed)
    bench = BenchCfg(adapt_steps=5, adapt_lr=1e-2, adapt_window=24)
    result = online_recalibrate(model, batch, bench, OvenCfg())
    assert result["coordinate"].ndim == 1
    assert result["coordinate"].numel() > 0
