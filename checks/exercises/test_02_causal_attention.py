import torch
from torch.nn import functional as F

from checks.exercises._loader import load_starter

student = load_starter("02_causal_attention.py")


def test_attention_matches_pytorch_oracle() -> None:
    torch.manual_seed(101)
    query = torch.randn(6, 4, requires_grad=True)
    key = torch.randn(6, 4, requires_grad=True)
    value = torch.randn(6, 3, requires_grad=True)
    actual = student.causal_attention(query, key, value)
    expected = F.scaled_dot_product_attention(
        query[None, None], key[None, None], value[None, None], is_causal=True
    )[0, 0]
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    actual.square().mean().backward()
    assert all(tensor.grad is not None for tensor in (query, key, value))


def test_future_value_cannot_change_past_outputs() -> None:
    torch.manual_seed(102)
    query, key, value = (torch.randn(5, 4) for _ in range(3))
    before = student.causal_attention(query, key, value)
    changed = value.clone()
    changed[-1] += 10_000
    after = student.causal_attention(query, key, changed)
    torch.testing.assert_close(before[:-1], after[:-1])
