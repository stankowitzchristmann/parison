from __future__ import annotations

import torch

from parison.furnace.selective import SelectiveStream


def _block() -> SelectiveStream:
    torch.manual_seed(7)
    return SelectiveStream(
        d_model=16, d_state=8, d_conv=4, expand=2, dt_rank=4, action_dim=5
    ).eval()


def test_output_shape_matches_input() -> None:
    block = _block()
    seq = torch.randn(3, 20, 16)
    act = torch.randn(3, 20, 5)
    out, state = block(seq, act)
    assert out.shape == seq.shape
    assert state.ssm.shape == (3, 32, 8)


def test_step_matches_scan() -> None:
    block = _block()
    seq = torch.randn(1, 12, 16)
    act = torch.randn(1, 12, 5)
    full, _ = block(seq, act)
    from parison.furnace.selective import StreamState

    state = StreamState(
        ssm=torch.zeros(1, block.d_inner, block.d_state),
        conv=torch.zeros(1, block.d_inner, block.d_conv - 1),
    )
    streamed = []
    for t in range(seq.shape[1]):
        y, state = block.step(seq[:, t], act[:, t], state)
        streamed.append(y)
    online = torch.stack(streamed, dim=1)
    assert torch.allclose(full, online, atol=1e-5)


def test_gradients_flow_through_scan() -> None:
    block = _block()
    seq = torch.randn(2, 10, 16, requires_grad=True)
    act = torch.randn(2, 10, 5, requires_grad=True)
    out, _ = block(seq, act)
    out.pow(2).mean().backward()
    assert seq.grad is not None and torch.isfinite(seq.grad).all()
    assert act.grad is not None and torch.isfinite(act.grad).all()
    assert block.a_log.grad is not None and torch.isfinite(block.a_log.grad).all()
