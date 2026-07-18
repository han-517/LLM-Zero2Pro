import torch
from torch import nn

from exercises.checks._loader import load_starter

student = load_starter("13_tiny_gpt.py")


def test_rectangular_causal_mask_uses_decode_offset() -> None:
    expected = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]], dtype=torch.bool)
    torch.testing.assert_close(student.causal_mask(2, 4, offset=2), expected)


def test_prenorm_residual_keeps_skip_path() -> None:
    x = torch.tensor([[1.0, 3.0]])
    out = student.prenorm_residual(x, nn.Identity(), nn.Identity())
    torch.testing.assert_close(out, 2 * x)
