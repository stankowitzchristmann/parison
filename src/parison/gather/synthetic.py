from __future__ import annotations

from typing import NamedTuple

import numpy as np

from parison.bench.settings import FeedCfg, GlassCfg


class Patient(NamedTuple):
    icp: np.ndarray
    abp: np.ndarray
    ecg: np.ndarray
    amp: np.ndarray
    ehr: np.ndarray
    actions: np.ndarray
    reserve: float
    reactivity: float


def _slow(length: int, period: float, rng: np.random.Generator) -> np.ndarray:
    phase = rng.uniform(0.0, 2.0 * np.pi)
    base = np.sin(2.0 * np.pi * np.arange(length) / period + phase)
    drift = np.cumsum(rng.normal(0.0, 0.03, size=length))
    return base + 0.4 * (drift - drift.mean())


def bake_patient(feed: FeedCfg, glass: GlassCfg, length: int, rng: np.random.Generator) -> Patient:
    reserve = float(rng.uniform(0.15, 0.95))
    reactivity = float(rng.uniform(-0.4, 0.9))
    icp0 = float(rng.uniform(9.0, 15.0))
    abp0 = float(rng.uniform(80.0, 100.0))

    slow_abp = _slow(length, period=70.0, rng=rng)
    slow_icp_drive = reactivity * slow_abp
    surge = np.clip(np.cumsum(rng.normal(reserve * 0.05, 0.2, size=length)), 0.0, None)
    surge = surge / (surge.max() + 1e-6) * (8.0 + 18.0 * reserve)

    actions = rng.integers(0, glass.action_levels, size=length).astype(np.float32)
    therapy = np.zeros(length, dtype=np.float32)
    decay = 0.85
    carry = 0.0
    for t in range(length):
        carry = decay * carry + (1.0 - decay) * actions[t]
        therapy[t] = carry
    therapy_effect = 2.4 * therapy

    pulse = 1.2 + 0.6 * np.sin(2.0 * np.pi * np.arange(length) / 5.0)
    noise = rng.normal(0.0, 0.8, size=length)
    icp = icp0 + surge + 2.5 * slow_icp_drive - therapy_effect + pulse + noise
    icp = np.clip(icp, 0.0, 80.0)

    amp = (
        0.8
        + 2.6 * reserve * (icp - icp.mean()) / (icp.std() + 1e-6)
        + rng.normal(0.0, 0.3, size=length)
    )
    abp = abp0 + 6.0 * slow_abp + rng.normal(0.0, 1.5, size=length)
    ecg = np.sin(2.0 * np.pi * np.arange(length) / 4.0) + rng.normal(0.0, 0.1, size=length)

    ehr = np.stack(
        [
            np.full(length, reserve),
            np.full(length, reactivity),
            np.full(length, abp0 / 100.0),
            np.convolve(icp, np.ones(8) / 8.0, mode="same") / 30.0,
        ]
        + [rng.normal(0.0, 0.4, size=length) for _ in range(feed.ehr_dim - 4)],
        axis=-1,
    ).astype(np.float32)

    one = np.zeros((length, glass.action_dim), dtype=np.float32)
    one[:, 0] = actions / max(1, glass.action_levels - 1)
    one[np.arange(length), 1 + (actions.astype(int) % (glass.action_dim - 1))] = 1.0

    return Patient(
        icp=icp.astype(np.float32),
        abp=abp.astype(np.float32),
        ecg=ecg.astype(np.float32),
        amp=amp.astype(np.float32),
        ehr=ehr,
        actions=one,
        reserve=reserve,
        reactivity=reactivity,
    )


def sustained_onset(icp: np.ndarray, threshold: float, run: int, lead: int) -> np.ndarray:
    over = icp >= threshold
    streak = 0
    starts = []
    for t in range(len(icp)):
        streak = streak + 1 if over[t] else 0
        if streak == run:
            starts.append(t - run + 1)
    label = np.zeros(len(icp), dtype=np.float32)
    for s in starts:
        lo = max(0, s - lead)
        label[lo:s] = 1.0
    return label
