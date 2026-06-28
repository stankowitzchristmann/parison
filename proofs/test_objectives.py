from __future__ import annotations

import torch

from parison.annealing.objectives import kiln_loss
from parison.bench.settings import FeedCfg, GlassCfg, OvenCfg
from parison.furnace.heads import TwinOutputs
from parison.furnace.twin import TherapyTwin
from parison.gather.streams import Parcel, collate
from parison.gather.windows import Foundry


def test_masked_huber_ignores_invalid() -> None:
    cfg = OvenCfg(warning_weight=0.0, outcome_weight=0.0, forecast_weight=1.0)
    forecast = torch.zeros(1, 3, 2)
    target = torch.zeros(1, 3, 2)
    target[0, 2] = 10.0
    valid = torch.tensor([[1.0, 1.0, 0.0]])
    outputs = TwinOutputs(forecast=forecast, warning=torch.zeros(1, 3), outcome=torch.zeros(1))
    batch = Parcel(
        waveforms={},
        presence=torch.empty(0),
        actions=torch.empty(0),
        future_icp=target,
        ih_label=torch.zeros(1, 3),
        outcome_label=torch.zeros(1),
        valid=valid,
    )
    total, report = kiln_loss(outputs, batch, cfg)
    assert report["forecast"] == 0.0
    assert float(total) == 0.0


def test_overfit_single_batch(tiny_glass: GlassCfg, tiny_feed: FeedCfg) -> None:
    ds = Foundry(tiny_feed, tiny_glass, 1, "train")
    batch = collate([ds[i] for i in range(4)])
    model = TherapyTwin(tiny_glass, tiny_feed)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    cfg = OvenCfg()
    first = None
    for _ in range(40):
        opt.zero_grad()
        outputs, _ = model(batch)
        loss, _ = kiln_loss(outputs, batch, cfg)
        loss.backward()
        opt.step()
        if first is None:
            first = float(loss.detach())
    assert float(loss.detach()) < 0.6 * first
