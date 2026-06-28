from __future__ import annotations

from pathlib import Path
from typing import Any

import tomli
from pydantic import BaseModel, Field, model_validator


class GlassCfg(BaseModel):
    d_model: int = 96
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    depth: int = 4
    dt_rank: int = 8
    modalities: list[str] = Field(default_factory=lambda: ["icp", "abp", "ecg", "ehr"])
    action_dim: int = 6
    action_levels: int = 5
    horizon: int = 30
    lead_window: int = 30
    fuse_hidden: int = 64
    condition: bool = True
    gated_fusion: bool = True

    @model_validator(mode="after")
    def _check(self) -> GlassCfg:
        if not self.modalities:
            raise ValueError("at least one modality required")
        if "icp" not in self.modalities:
            raise ValueError("icp modality is mandatory")
        return self

    @property
    def d_inner(self) -> int:
        return self.expand * self.d_model


class FeedCfg(BaseModel):
    sample_hz: int = 1
    window_steps: int = 360
    stride_steps: int = 30
    icp_threshold: float = 22.0
    sustained_steps: int = 10
    ehr_dim: int = 24
    waveform_dim: int = 1
    cohort: int = 256
    val_fraction: float = 0.2
    drop_rate: float = 0.15


class OvenCfg(BaseModel):
    lr: float = 3.0e-4
    weight_decay: float = 0.05
    betas: tuple[float, float] = (0.9, 0.999)
    warmup_steps: int = 2000
    max_epochs: int = 100
    grad_clip: float = 1.0
    scheduler: str = "cosine"
    early_stop_patience: int = 12
    forecast_weight: float = 1.0
    warning_weight: float = 1.0
    outcome_weight: float = 0.5
    huber_delta: float = 1.0


class BenchCfg(BaseModel):
    per_device_batch: int = 32
    grad_accum: int = 16
    world_size: int = 8
    precision: str = "fp32"
    seed: int = 0
    adapt_steps: int = 24
    adapt_lr: float = 5.0e-4
    adapt_window: int = 120
    epochs_between_anneal: int = 1

    @property
    def effective_batch(self) -> int:
        return self.per_device_batch * self.grad_accum * self.world_size


class Recipe(BaseModel):
    name: str = "main"
    glass: GlassCfg = Field(default_factory=GlassCfg)
    feed: FeedCfg = Field(default_factory=FeedCfg)
    oven: OvenCfg = Field(default_factory=OvenCfg)
    bench: BenchCfg = Field(default_factory=BenchCfg)


def _coerce(text: str) -> Any:
    low = text.lower()
    if low in {"true", "false"}:
        return low == "true"
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    if "," in text:
        return [_coerce(part.strip()) for part in text.split(",")]
    return text


def _graft(tree: dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    node = tree
    for key in keys[:-1]:
        nxt = node.setdefault(key, {})
        if not isinstance(nxt, dict):
            raise ValueError(f"override path {dotted} collides with scalar")
        node = nxt
    node[keys[-1]] = value


def load_recipe(path: str | Path | None, overrides: list[str] | None = None) -> Recipe:
    raw: dict[str, Any] = {}
    if path is not None:
        with open(path, "rb") as handle:
            raw = tomli.load(handle)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must be key=value, got {item!r}")
        key, value = item.split("=", 1)
        _graft(raw, key.strip(), _coerce(value.strip()))
    return Recipe.model_validate(raw)
