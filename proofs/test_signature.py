from __future__ import annotations

import numpy as np

from parison.bench.settings import FeedCfg, GlassCfg
from parison.calipers.signature import (
    axis_alignment,
    prx,
    r_squared,
    rap,
    two_axis_embedding,
)
from parison.gather.synthetic import bake_patient


def test_prx_tracks_reactivity() -> None:
    rng = np.random.default_rng(0)
    feed = FeedCfg(ehr_dim=8)
    glass = GlassCfg()
    high = [bake_patient(feed, glass, 400, rng) for _ in range(12)]
    values = [(p.reactivity, prx(p.icp, p.abp)) for p in high]
    react = np.array([v[0] for v in values])
    measured = np.array([v[1] for v in values])
    assert axis_alignment(measured, react) > 0.4


def test_rap_tracks_reserve() -> None:
    rng = np.random.default_rng(1)
    feed = FeedCfg(ehr_dim=8)
    glass = GlassCfg()
    cohort = [bake_patient(feed, glass, 400, rng) for _ in range(12)]
    reserve = np.array([p.reserve for p in cohort])
    measured = np.array([rap(p.amp, p.icp) for p in cohort])
    assert axis_alignment(measured, reserve) > 0.4


def test_two_axis_embedding_orders_variance() -> None:
    rng = np.random.default_rng(2)
    coords = rng.normal(size=(40, 12)) @ np.diag([5.0, 3.0] + [0.2] * 10)
    scores, explained = two_axis_embedding(coords)
    assert scores.shape == (40, 2)
    assert explained[0] >= explained[1]
    assert explained.sum() > 0.5


def test_r_squared_perfect() -> None:
    y = np.linspace(0, 1, 20)
    assert abs(r_squared(y, y) - 1.0) < 1e-9
