import torch
from torch.nn import functional as F

from llm_from_scratch.attention import (
    GroupedQueryAttention,
    MultiHeadAttention,
    SimpleMLA,
    apply_rope,
    causal_linear_attention,
    gated_delta_rule,
    grouped_query_attention,
    scaled_dot_product_attention,
    sliding_window_mask,
)


def test_attention_matches_pytorch_sdpa() -> None:
    torch.manual_seed(1)
    q = torch.randn(2, 3, 5, 4, requires_grad=True)
    k = torch.randn(2, 3, 5, 4, requires_grad=True)
    v = torch.randn(2, 3, 5, 4, requires_grad=True)
    actual, _ = scaled_dot_product_attention(q, k, v, causal=True)
    expected = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    actual.sum().backward()
    assert all(tensor.grad is not None for tensor in (q, k, v))


def test_causal_attention_does_not_read_future() -> None:
    torch.manual_seed(2)
    q = torch.randn(1, 1, 4, 8)
    k = torch.randn(1, 1, 4, 8)
    v = torch.randn(1, 1, 4, 8)
    before, _ = scaled_dot_product_attention(q, k, v, causal=True)
    k[:, :, 3] += 1000
    v[:, :, 3] -= 1000
    after, _ = scaled_dot_product_attention(q, k, v, causal=True)
    torch.testing.assert_close(before[:, :, :3], after[:, :, :3])


def test_multi_head_cache_matches_full_forward() -> None:
    torch.manual_seed(3)
    module = MultiHeadAttention(16, 4).eval()
    x = torch.randn(2, 5, 16)
    full, _ = module(x)
    cache = None
    pieces = []
    for index in range(x.shape[1]):
        current, cache = module(x[:, index : index + 1], cache)
        pieces.append(current)
    cached = torch.cat(pieces, dim=1)
    torch.testing.assert_close(full, cached, atol=1e-6, rtol=1e-5)


def test_grouped_attention_matches_explicit_kv_expansion() -> None:
    torch.manual_seed(31)
    query = torch.randn(2, 4, 5, 3)
    key = torch.randn(2, 2, 5, 3)
    value = torch.randn(2, 2, 5, 6)
    actual, weights = grouped_query_attention(query, key, value, causal=True)
    expected, _ = scaled_dot_product_attention(
        query,
        key.repeat_interleave(2, dim=1),
        value.repeat_interleave(2, dim=1),
        causal=True,
    )
    torch.testing.assert_close(actual, expected)
    assert weights.shape == (2, 2, 2, 5, 5)


def test_gqa_cache_uses_fewer_kv_heads() -> None:
    module = GroupedQueryAttention(16, num_query_heads=4, num_kv_heads=2)
    output, cache = module(torch.randn(2, 3, 16))
    assert output.shape == (2, 3, 16)
    assert cache[0].shape == (2, 2, 3, 4)
    assert cache[1].shape == (2, 2, 3, 4)


def test_rope_preserves_norm() -> None:
    x = torch.randn(2, 3, 7, 8)
    rotated = apply_rope(x)
    torch.testing.assert_close(rotated.norm(dim=-1), x.norm(dim=-1))


def test_mla_cache_is_latent_and_incremental_output_matches() -> None:
    torch.manual_seed(4)
    module = SimpleMLA(d_model=16, num_heads=4, latent_dim=3).eval()
    x = torch.randn(1, 5, 16)
    full, latent = module(x)
    assert latent.shape == (1, 5, 3)
    cache = None
    pieces = []
    for index in range(5):
        current, cache = module(x[:, index : index + 1], cache)
        pieces.append(current)
    torch.testing.assert_close(full, torch.cat(pieces, dim=1), atol=1e-6, rtol=1e-5)


def test_linear_attention_is_causal() -> None:
    torch.manual_seed(5)
    q = torch.randn(1, 2, 6, 4)
    k = torch.randn(1, 2, 6, 4)
    v = torch.randn(1, 2, 6, 3)
    before = causal_linear_attention(q, k, v)
    v[:, :, -1] += 100
    after = causal_linear_attention(q, k, v)
    torch.testing.assert_close(before[:, :, :-1], after[:, :, :-1])


def test_gated_delta_zero_beta_writes_nothing() -> None:
    q = torch.randn(1, 1, 3, 2)
    k = torch.randn(1, 1, 3, 2)
    v = torch.randn(1, 1, 3, 2)
    beta = torch.zeros(1, 1, 3, 1)
    decay = torch.ones_like(beta)
    output, state = gated_delta_rule(q, k, v, beta, decay)
    assert torch.count_nonzero(output) == 0
    assert torch.count_nonzero(state) == 0


def test_sliding_window_mask() -> None:
    expected = torch.tensor(
        [
            [True, False, False, False],
            [True, True, False, False],
            [False, True, True, False],
            [False, False, True, True],
        ]
    )
    assert torch.equal(sliding_window_mask(4, 2), expected)

