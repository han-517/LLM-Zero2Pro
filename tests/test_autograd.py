import math

from llm_from_scratch.autograd import Value


def test_scalar_autograd_chain_rule() -> None:
    x = Value(2.0)
    y = x * x + 3 * x - 1
    y.backward()
    assert math.isclose(y.data, 9.0)
    assert math.isclose(x.grad, 7.0)


def test_tanh_gradient_matches_formula() -> None:
    x = Value(0.4)
    y = x.tanh()
    y.backward()
    assert math.isclose(x.grad, 1 - math.tanh(0.4) ** 2, rel_tol=1e-7)

