import pytest
import torch
from torch.nn import functional as F

from llm_from_scratch.attention import (
    GroupedQueryAttention,
    LatentCacheMLABaseline,
    MultiHeadAttention,
    apply_rope,
    causal_linear_attention,
    causal_linear_attention_parallel,
    gated_delta_rule,
    grouped_query_attention,
    mla_cache_cost,
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


def test_fully_masked_attention_rows_are_zero() -> None:
    q = torch.tensor([[[[1.0], [2.0]]]])
    k = torch.tensor([[[[1.0], [2.0]]]])
    v = torch.tensor([[[[3.0], [7.0]]]])
    mask = torch.tensor([[False, False], [True, False]])
    output, weights = scaled_dot_product_attention(q, k, v, attention_mask=mask)
    assert torch.equal(weights[0, 0, 0], torch.zeros(2))
    assert output[0, 0, 0, 0] == 0
    assert torch.equal(weights[0, 0, 1], torch.tensor([1.0, 0.0]))


def test_causal_and_padding_masks_compose() -> None:
    q = torch.ones(1, 1, 3, 2)
    k = torch.ones_like(q)
    v = torch.arange(6, dtype=torch.float32).view(1, 1, 3, 2)
    padding_mask = torch.tensor([[[[False, True, True]]]])
    output, weights = scaled_dot_product_attention(
        q, k, v, causal=True, attention_mask=padding_mask
    )
    assert torch.count_nonzero(weights[..., 0, :]) == 0
    assert torch.count_nonzero(output[..., 0, :]) == 0
    assert weights[..., 1, 2] == 0


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


def test_multi_head_cache_matches_full_forward_with_rope() -> None:
    torch.manual_seed(3)
    module = MultiHeadAttention(16, 4, rope_base=10_000).eval()
    x = torch.randn(2, 5, 16)
    full, _ = module(x)
    cache = None
    pieces = []
    for index in range(x.shape[1]):
        current, cache = module(x[:, index : index + 1], cache)
        assert cache is not None
        pieces.append(current)
    cached = torch.cat(pieces, dim=1)
    torch.testing.assert_close(full, cached, atol=1e-6, rtol=1e-5)


def test_attention_can_skip_cache_materialization() -> None:
    output, cache = MultiHeadAttention(8, 2)(torch.randn(1, 3, 8), return_cache=False)
    assert output.shape == (1, 3, 8)
    assert cache is None


def test_grouped_attention_matches_explicit_kv_expansion_and_mqa() -> None:
    torch.manual_seed(31)
    for kv_heads in (1, 2):
        query = torch.randn(2, 4, 5, 3)
        key = torch.randn(2, kv_heads, 5, 3)
        value = torch.randn(2, kv_heads, 5, 6)
        actual, weights = grouped_query_attention(query, key, value, causal=True)
        repeats = 4 // kv_heads
        expected, _ = scaled_dot_product_attention(
            query,
            key.repeat_interleave(repeats, dim=1),
            value.repeat_interleave(repeats, dim=1),
            causal=True,
        )
        torch.testing.assert_close(actual, expected)
        assert weights.shape == (2, kv_heads, repeats, 5, 5)


def test_gqa_cache_uses_fewer_kv_heads_and_composes_mask() -> None:
    module = GroupedQueryAttention(16, num_query_heads=4, num_kv_heads=2)
    mask = torch.ones(2, 1, 3, 3, dtype=torch.bool)
    mask[0, :, 0] = False
    output, cache = module(torch.randn(2, 3, 16), attention_mask=mask)
    assert output.shape == (2, 3, 16)
    assert cache is not None
    assert cache[0].shape == (2, 2, 3, 4)
    assert cache[1].shape == (2, 2, 3, 4)


def test_head_parameter_validation_rejects_zero() -> None:
    with pytest.raises(ValueError):
        GroupedQueryAttention(16, num_query_heads=4, num_kv_heads=0)


def test_rope_preserves_norm_relative_position_and_partial_nope() -> None:
    torch.manual_seed(7)
    x = torch.randn(2, 3, 1, 8)
    y = torch.randn(2, 3, 1, 8)
    first = (apply_rope(x, torch.tensor([2])) * apply_rope(y, torch.tensor([5]))).sum(-1)
    shifted = (apply_rope(x, torch.tensor([9])) * apply_rope(y, torch.tensor([12]))).sum(-1)
    torch.testing.assert_close(first, shifted, atol=1e-6, rtol=1e-5)
    batched = apply_rope(x, torch.tensor([[2], [9]]), rotary_dim=4)
    torch.testing.assert_close(batched.norm(dim=-1), x.norm(dim=-1))
    torch.testing.assert_close(batched[..., 4:], x[..., 4:])


def test_mla_baseline_cache_and_cost_are_explicit() -> None:
    torch.manual_seed(4)
    module = LatentCacheMLABaseline(d_model=16, num_heads=4, latent_dim=3).eval()
    x = torch.randn(1, 5, 16)
    full, latent = module(x)
    assert latent.shape == (1, 5, 3)
    cache = None
    pieces = []
    for index in range(5):
        current, cache = module(x[:, index : index + 1], cache)
        pieces.append(current)
    torch.testing.assert_close(full, torch.cat(pieces, dim=1), atol=1e-6, rtol=1e-5)
    cost = mla_cache_cost(batch_size=1, layers=2, sequence_length=5, d_model=16, latent_dim=4)
    assert cost.dense_cache_bytes == 640
    assert cost.latent_cache_bytes == 80
    assert cost.compression_ratio == 8
    assert cost.reconstruction_macs_per_decode_step == 1280


def test_linear_attention_parallel_matches_recurrent_and_is_causal() -> None:
    torch.manual_seed(5)
    q = torch.randn(1, 2, 6, 4)
    k = torch.randn(1, 2, 6, 4)
    v = torch.randn(1, 2, 6, 3)
    recurrent = causal_linear_attention(q, k, v)
    parallel = causal_linear_attention_parallel(q, k, v)
    torch.testing.assert_close(recurrent, parallel, atol=1e-6, rtol=1e-5)
    v[:, :, -1] += 100
    changed = causal_linear_attention(q, k, v)
    torch.testing.assert_close(recurrent[:, :, :-1], changed[:, :, :-1])


def test_gated_delta_zero_beta_and_state_continuation() -> None:
    torch.manual_seed(8)
    q = torch.randn(1, 1, 4, 2)
    k = torch.randn(1, 1, 4, 2)
    v = torch.randn(1, 1, 4, 2)
    beta = torch.zeros(1, 1, 4, 1)
    decay = torch.ones_like(beta)
    output, state = gated_delta_rule(q, k, v, beta, decay)
    assert torch.count_nonzero(output) == 0
    assert torch.count_nonzero(state) == 0

    beta.fill_(0.7)
    decay.fill_(0.9)
    full, full_state = gated_delta_rule(q, k, v, beta, decay)
    first, split_state = gated_delta_rule(
        q[:, :, :2], k[:, :, :2], v[:, :, :2], beta[:, :, :2], decay[:, :, :2]
    )
    second, split_state = gated_delta_rule(
        q[:, :, 2:],
        k[:, :, 2:],
        v[:, :, 2:],
        beta[:, :, 2:],
        decay[:, :, 2:],
        split_state,
    )
    torch.testing.assert_close(full, torch.cat((first, second), dim=2))
    torch.testing.assert_close(full_state, split_state)


def test_sliding_window_mask_full_and_cached() -> None:
    expected = torch.tensor(
        [
            [True, False, False, False],
            [True, True, False, False],
            [False, True, True, False],
            [False, False, True, True],
        ]
    )
    assert torch.equal(sliding_window_mask(4, 2), expected)
    cached = torch.tensor([[False, False, False, True, True]])
    assert torch.equal(sliding_window_mask(1, 2, key_length=5), cached)
