import torch

from checks.exercises._loader import load_starter

student = load_starter("16_attention_frontiers.py")


def _dense_causal_attention(query, key, value):
    scale = query.shape[-1] ** -0.5
    scores = query @ key.transpose(-2, -1) * scale
    mask = torch.ones(scores.shape[-2:], dtype=torch.bool).tril()
    probabilities = scores.masked_fill(~mask, float("-inf")).softmax(dim=-1)
    return probabilities @ value


def test_tiled_online_softmax_matches_dense_for_partial_blocks() -> None:
    torch.manual_seed(0)
    query = torch.randn(2, 3, 7, 5, dtype=torch.float64)
    key = torch.randn(2, 3, 7, 5, dtype=torch.float64)
    value = torch.randn(2, 3, 7, 4, dtype=torch.float64)
    actual = student.tiled_causal_attention(query, key, value, block_size=3)
    expected = _dense_causal_attention(query, key, value)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)


def test_tiled_attention_is_causal() -> None:
    torch.manual_seed(1)
    query = torch.randn(1, 1, 6, 4)
    key = torch.randn(1, 1, 6, 4)
    value = torch.randn(1, 1, 6, 4)
    baseline = student.tiled_causal_attention(query, key, value, block_size=4)
    value[:, :, 4:] += 10_000
    changed = student.tiled_causal_attention(query, key, value, block_size=4)
    torch.testing.assert_close(baseline[:, :, :4], changed[:, :, :4])


def test_sliding_window_cache_alignment() -> None:
    expected = torch.tensor([[0, 1, 1, 0], [0, 0, 1, 1]], dtype=torch.bool)
    torch.testing.assert_close(student.sliding_window_mask(2, 4, 2), expected)


def test_positive_feature_is_strictly_positive() -> None:
    values = student.positive_feature(torch.tensor([-100.0, 0.0, 2.0]))
    assert torch.all(values > 0)
