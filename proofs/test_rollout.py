from __future__ import annotations

import torch

from parison.bench.settings import FeedCfg, GlassCfg
from parison.furnace.twin import TherapyTwin
from parison.gather.streams import collate
from parison.gather.windows import Foundry
from parison.soffietta.optimizer import best_intervention
from parison.soffietta.rollout import encode_level, roll_forward


def _model_batch(tiny_glass: GlassCfg, tiny_feed: FeedCfg):
    ds = Foundry(tiny_feed, tiny_glass, 8, "train")
    batch = collate([ds[i] for i in range(4)])
    return TherapyTwin(tiny_glass, tiny_feed).eval(), batch


def test_rollout_shape(tiny_glass: GlassCfg, tiny_feed: FeedCfg) -> None:
    model, batch = _model_batch(tiny_glass, tiny_feed)
    actions = encode_level(
        2, tiny_glass.action_dim, tiny_glass.action_levels, 4, batch.actions.device
    )
    future = actions.unsqueeze(1).expand(4, tiny_glass.horizon, tiny_glass.action_dim).contiguous()
    traj = roll_forward(model, batch, future)
    assert traj.shape == (4, tiny_glass.horizon)
    assert torch.isfinite(traj).all()


def test_distinct_actions_yield_distinct_trajectories(
    tiny_glass: GlassCfg, tiny_feed: FeedCfg
) -> None:
    model, batch = _model_batch(tiny_glass, tiny_feed)
    h, ad, al = tiny_glass.horizon, tiny_glass.action_dim, tiny_glass.action_levels
    low = (
        encode_level(0, ad, al, 4, batch.actions.device).unsqueeze(1).expand(4, h, ad).contiguous()
    )
    high = (
        encode_level(4, ad, al, 4, batch.actions.device).unsqueeze(1).expand(4, h, ad).contiguous()
    )
    diff = (roll_forward(model, batch, low) - roll_forward(model, batch, high)).abs().max()
    assert float(diff) > 1e-6


def test_optimizer_picks_argmin(tiny_glass: GlassCfg, tiny_feed: FeedCfg) -> None:
    model, batch = _model_batch(tiny_glass, tiny_feed)
    verdict = best_intervention(model, batch, tiny_glass.action_levels, tiny_glass.horizon)
    assert verdict.scores.shape == (4, tiny_glass.action_levels)
    assert verdict.trajectories.shape == (
        4,
        tiny_glass.action_levels,
        tiny_glass.horizon,
    )
    assert torch.equal(verdict.level, verdict.scores.argmin(dim=1))
    assert int(verdict.level.min()) >= 0
    assert int(verdict.level.max()) < tiny_glass.action_levels
