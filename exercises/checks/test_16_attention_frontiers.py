import torch

from exercises.checks._loader import load_starter

student = load_starter("16_attention_frontiers.py")


def test_sliding_window_cache_alignment() -> None:
    expected = torch.tensor([[0, 1, 1, 0], [0, 0, 1, 1]], dtype=torch.bool)
    torch.testing.assert_close(student.sliding_window_mask(2, 4, 2), expected)


def test_positive_feature_is_strictly_positive() -> None:
    values = student.positive_feature(torch.tensor([-100.0, 0.0, 2.0]))
    assert torch.all(values > 0)
