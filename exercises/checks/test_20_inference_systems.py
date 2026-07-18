import pytest
import torch

from exercises.checks._loader import load_starter

student = load_starter("20_inference_systems.py")


def test_quantization_bounds_and_zero_tensor() -> None:
    q, scale = student.symmetric_quantize(torch.tensor([-2.0, 0.0, 2.0]), bits=8)
    assert q.dtype in (torch.int8, torch.int16, torch.int32, torch.int64)
    assert q.min() >= -127 and q.max() <= 127
    zero_q, zero_scale = student.symmetric_quantize(torch.zeros(3), bits=8)
    assert torch.equal(zero_q, torch.zeros_like(zero_q))
    assert torch.isfinite(zero_scale)


def test_speculative_acceptance_ratio() -> None:
    out = student.acceptance_probability(torch.tensor([0.5, 0.2]), torch.tensor([0.25, 0.4]))
    torch.testing.assert_close(out, torch.tensor([0.5, 1.0]))
    with pytest.raises(ValueError):
        student.acceptance_probability(torch.tensor([-0.1]), torch.tensor([0.2]))
