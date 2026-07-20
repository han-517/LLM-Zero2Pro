import math

import pytest
import torch

from checks.exercises._loader import load_starter

student = load_starter("08_grouped_query_attention.py")


def explicit_gqa(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    groups = query.shape[1] // key.shape[1]
    expanded_key = key.repeat_interleave(groups, dim=1)
    expanded_value = value.repeat_interleave(groups, dim=1)
    scores = query @ expanded_key.transpose(-2, -1) / math.sqrt(query.shape[-1])
    query_length, key_length = query.shape[-2], key.shape[-2]
    query_positions = torch.arange(key_length - query_length, key_length).unsqueeze(-1)
    key_positions = torch.arange(key_length).unsqueeze(0)
    scores = scores.masked_fill(~(key_positions <= query_positions), -torch.inf)
    return torch.softmax(scores, dim=-1) @ expanded_value


def test_gqa_matches_explicit_kv_expansion() -> None:
    torch.manual_seed(108)
    query = torch.randn(2, 4, 5, 3, requires_grad=True)
    key = torch.randn(2, 2, 5, 3, requires_grad=True)
    value = torch.randn(2, 2, 5, 6, requires_grad=True)
    actual = student.grouped_query_attention(query, key, value)
    expected = explicit_gqa(query, key, value)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    actual.mean().backward()
    assert all(tensor.grad is not None for tensor in (query, key, value))


def test_cache_decode_uses_bottom_right_causal_alignment() -> None:
    torch.manual_seed(109)
    query = torch.randn(1, 4, 2, 4)
    key = torch.randn(1, 2, 5, 4)
    value = torch.randn(1, 2, 5, 3)
    actual = student.grouped_query_attention(query, key, value)
    torch.testing.assert_close(actual, explicit_gqa(query, key, value), atol=1e-6, rtol=1e-5)


def test_incompatible_head_counts_are_rejected() -> None:
    with pytest.raises(ValueError):
        student.grouped_query_attention(
            torch.randn(1, 3, 2, 4), torch.randn(1, 2, 2, 4), torch.randn(1, 2, 2, 4)
        )
