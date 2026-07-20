import math

import pytest
import torch

from checks.exercises._loader import load_starter

student = load_starter("07_rope.py")


def test_rope_preserves_shape_dtype_and_norm() -> None:
    x = torch.randn(2, 3, 7, 8, dtype=torch.float64)
    rotated = student.apply_rope(x)
    assert rotated.shape == x.shape
    assert rotated.dtype == x.dtype
    torch.testing.assert_close(rotated.norm(dim=-1), x.norm(dim=-1))


def test_position_zero_is_identity_and_position_one_rotates() -> None:
    x = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
    rotated = student.apply_rope(x, positions=torch.tensor([0, 1]))
    torch.testing.assert_close(rotated[0, 0, 0], torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(rotated[0, 0, 1], torch.tensor([math.cos(1.0), math.sin(1.0)]))


def test_odd_head_dimension_is_rejected() -> None:
    with pytest.raises(ValueError):
        student.apply_rope(torch.randn(1, 1, 3, 5))
