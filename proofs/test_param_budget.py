from __future__ import annotations

from parison.bench.settings import load_recipe
from parison.furnace.twin import TherapyTwin


def test_main_recipe_param_budget() -> None:
    recipe = load_recipe("batches/main.toml")
    model = TherapyTwin(recipe.glass, recipe.feed)
    params = sum(p.numel() for p in model.parameters())
    assert 3.0e6 < params < 3.8e6


def test_main_effective_batch() -> None:
    recipe = load_recipe("batches/main.toml")
    assert recipe.bench.effective_batch == 4096
    assert recipe.oven.lr == 3.0e-4
    assert recipe.bench.precision == "fp32"


def test_world_size_is_demanding() -> None:
    recipe = load_recipe("batches/main.toml")
    assert recipe.bench.world_size == 8
