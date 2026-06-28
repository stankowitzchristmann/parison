from __future__ import annotations

import torch

from parison.furnace.fusion import CoincidenceGate


def test_weights_sum_to_one_over_present() -> None:
    gate = CoincidenceGate(4, 12, 8, gated=True).eval()
    streams = torch.randn(2, 5, 4, 12)
    presence = torch.ones(2, 5, 4)
    presence[:, :, 3] = 0.0
    fused, weights = gate(streams, presence)
    assert fused.shape == (2, 5, 12)
    assert torch.allclose(weights.sum(-1), torch.ones(2, 5), atol=1e-5)
    assert torch.all(weights[:, :, 3] == 0.0)


def test_absent_modality_does_not_leak() -> None:
    gate = CoincidenceGate(3, 12, 8, gated=True).eval()
    streams = torch.randn(1, 4, 3, 12)
    presence = torch.tensor([[[1.0, 1.0, 0.0]] * 4])
    fused_a, _ = gate(streams, presence)
    poisoned = streams.clone()
    poisoned[:, :, 2] = 99.0
    fused_b, _ = gate(poisoned, presence)
    assert torch.allclose(fused_a, fused_b, atol=1e-5)


def test_ungated_is_uniform_over_present() -> None:
    gate = CoincidenceGate(3, 12, 8, gated=False).eval()
    streams = torch.randn(2, 3, 3, 12)
    presence = torch.tensor([[[1.0, 1.0, 0.0]] * 3] * 2)
    _, weights = gate(streams, presence)
    expected = torch.tensor([0.5, 0.5, 0.0])
    assert torch.allclose(weights[0, 0], expected, atol=1e-6)
