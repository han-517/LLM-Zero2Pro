from __future__ import annotations

import torch
from student_lm.model import (
    GPTConfig,
    RMSNorm,
    SwiGLU,
    TransformerLM,
    apply_rope,
    causal_attention,
)
from torch.nn import functional as F


def test_rmsnorm_and_swiglu_match_small_oracles() -> None:
    x = torch.tensor([[1.0, 2.0, -3.0]], dtype=torch.float64)
    norm = RMSNorm(3, eps=1e-8).double()
    expected = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-8)
    torch.testing.assert_close(norm(x), expected)

    layer = SwiGLU(3, 5).double()
    expected_ffn = layer.down(F.silu(layer.gate(x)) * layer.up(x))
    torch.testing.assert_close(layer(x), expected_ffn)


def test_rope_preserves_pair_norms_and_position_zero() -> None:
    torch.manual_seed(1)
    x = torch.randn(2, 3, 4, 8)
    rotated = apply_rope(x, torch.arange(4))
    torch.testing.assert_close(rotated[..., 0, :], x[..., 0, :])
    before = x.reshape(2, 3, 4, 4, 2).square().sum(-1)
    after = rotated.reshape(2, 3, 4, 4, 2).square().sum(-1)
    torch.testing.assert_close(before, after, atol=1e-5, rtol=1e-5)


def test_causal_attention_blocks_future_values() -> None:
    query = torch.tensor([[[[1.0], [1.0]]]])
    key = torch.tensor([[[[1.0], [1.0]]]])
    value = torch.tensor([[[[2.0], [10_000.0]]]])
    output = causal_attention(query, key, value)
    torch.testing.assert_close(output[..., 0, :], torch.tensor([[[2.0]]]))
    assert output.shape == value.shape


def test_full_transformer_is_causal_and_has_finite_gradients() -> None:
    torch.manual_seed(2)
    config = GPTConfig(vocab_size=37, d_model=24, num_heads=4, num_layers=2, d_ff=48)
    model = TransformerLM(config)
    tokens = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
    logits = model(tokens)
    assert logits.shape == (2, 4, 37)

    perturbed = tokens.clone()
    perturbed[:, -1] = 9
    changed = model(perturbed)
    torch.testing.assert_close(logits[:, :-1], changed[:, :-1], atol=1e-6, rtol=1e-6)

    F.cross_entropy(logits[:, :-1].reshape(-1, 37), tokens[:, 1:].reshape(-1)).backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
