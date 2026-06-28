from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from parison.bench.settings import FeedCfg, GlassCfg
from parison.gather.streams import Parcel
from parison.gather.synthetic import Patient, bake_patient, sustained_onset

_WAVE = ("icp", "abp", "ecg")


class Norm:
    def __init__(self, stats: dict[str, tuple[float, float]]) -> None:
        self.stats = stats

    def apply(self, key: str, arr: np.ndarray) -> np.ndarray:
        mean, std = self.stats[key]
        return (arr - mean) / std

    def invert_icp(self, arr: torch.Tensor) -> torch.Tensor:
        mean, std = self.stats["icp"]
        return arr * std + mean


def _cumulative_dose(icp: np.ndarray, threshold: float) -> float:
    return float(np.clip(icp - threshold, 0.0, None).sum())


class Foundry(Dataset[Parcel]):
    def __init__(
        self,
        feed: FeedCfg,
        glass: GlassCfg,
        seed: int,
        split: str = "train",
    ) -> None:
        self.feed = feed
        self.glass = glass
        self.split = split
        rng = np.random.default_rng(seed if split == "train" else seed + 9973)
        length = feed.window_steps + glass.horizon
        n = feed.cohort if split == "train" else max(16, feed.cohort // 4)
        self.patients: list[Patient] = [bake_patient(feed, glass, length, rng) for _ in range(n)]
        doses = np.array([_cumulative_dose(p.icp, feed.icp_threshold) for p in self.patients])
        cutoff = np.quantile(doses, 0.6)
        self.outcomes = (doses > cutoff).astype(np.float32)
        self.presence_rng = np.random.default_rng(seed + 17)
        self.norm = self._fit_norm()

    def _fit_norm(self) -> Norm:
        stats: dict[str, tuple[float, float]] = {}
        for key in _WAVE:
            pool = np.concatenate([getattr(p, key) for p in self.patients])
            stats[key] = (float(pool.mean()), float(pool.std() + 1e-6))
        ehr_pool = np.concatenate([p.ehr for p in self.patients], axis=0)
        stats["ehr"] = (float(ehr_pool.mean()), float(ehr_pool.std() + 1e-6))
        return Norm(stats)

    def attach_norm(self, norm: Norm) -> None:
        self.norm = norm

    def __len__(self) -> int:
        return len(self.patients)

    def raw_signals(self, idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
        p = self.patients[idx]
        return p.icp, p.abp, p.amp, p.reserve, p.reactivity

    def __getitem__(self, idx: int) -> Parcel:
        p = self.patients[idx]
        w = self.feed.window_steps
        h = self.glass.horizon
        waves: dict[str, torch.Tensor] = {}
        for key in _WAVE:
            arr = self.norm.apply(key, getattr(p, key)[:w])
            waves[key] = torch.from_numpy(arr[:, None].astype(np.float32))
        ehr = self.norm.apply("ehr", p.ehr[:w])
        waves["ehr"] = torch.from_numpy(ehr.astype(np.float32))

        m = len(self.glass.modalities)
        presence = np.ones((w, m), dtype=np.float32)
        if self.split == "train":
            drop = self.presence_rng.random((w, m)) < self.feed.drop_rate
            drop[:, 0] = False
            presence[drop] = 0.0

        icp_mean, icp_std = self.norm.stats["icp"]
        future = np.stack([p.icp[t + 1 : t + 1 + h] for t in range(w)], axis=0)
        future = (future - icp_mean) / icp_std

        onset = sustained_onset(
            p.icp,
            self.feed.icp_threshold,
            self.feed.sustained_steps,
            self.glass.lead_window,
        )

        return Parcel(
            waveforms=waves,
            presence=torch.from_numpy(presence),
            actions=torch.from_numpy(p.actions[:w]),
            future_icp=torch.from_numpy(future.astype(np.float32)),
            ih_label=torch.from_numpy(onset[:w]),
            outcome_label=torch.tensor(self.outcomes[idx]),
            valid=torch.ones(w, dtype=torch.float32),
        )
