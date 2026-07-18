import torch
from torch.nn import functional as F

from exercises.checks._loader import load_starter

student = load_starter("09_modern_decoder.py")


def test_rms_norm_matches_equation_without_mean_centering() -> None:
    x = torch.tensor([[[1.0, 2.0, 3.0, 4.0], [-2.0, 1.0, 0.0, 3.0]]])
    weight = torch.tensor([1.0, 0.5, 2.0, -1.0])
    expected = x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + 1e-6) * weight
    actual = student.rms_norm(x, weight)
    torch.testing.assert_close(actual, expected)
    assert not torch.allclose(actual.mean(dim=-1), torch.zeros(1, 2))


def test_swiglu_matches_explicit_formula_and_backpropagates() -> None:
    torch.manual_seed(110)
    x = torch.randn(2, 3, 4, requires_grad=True)
    gate = torch.randn(7, 4, requires_grad=True)
    up = torch.randn(7, 4, requires_grad=True)
    down = torch.randn(4, 7, requires_grad=True)
    expected = F.linear(F.silu(F.linear(x, gate)) * F.linear(x, up), down)
    actual = student.swiglu(x, gate, up, down)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    assert all(tensor.grad is not None for tensor in (x, gate, up, down))
