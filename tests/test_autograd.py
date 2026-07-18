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


def test_shared_subexpression_and_branch_gradients_are_summed() -> None:
    x = Value(2.0)
    shared = x * x
    y = shared + 3 * shared
    y.backward()
    assert math.isclose(x.grad, 16.0)


def test_repeated_backward_accumulates_only_leaf_gradients() -> None:
    x = Value(2.0)
    y = x * x + 3 * x - 1
    y.backward()
    y.backward()
    assert math.isclose(x.grad, 14.0)

    y.zero_grad()
    assert x.grad == 0.0
    assert y.grad == 0.0
    y.backward()
    assert math.isclose(x.grad, 7.0)


def test_autograd_matches_central_finite_difference() -> None:
    point = 0.4
    epsilon = 1e-5

    def function(value: float) -> float:
        return math.tanh(value * value + 3 * value)

    numerical = (function(point + epsilon) - function(point - epsilon)) / (2 * epsilon)
    x = Value(point)
    y = (x * x + 3 * x).tanh()
    y.backward()
    assert math.isclose(x.grad, numerical, rel_tol=1e-7, abs_tol=1e-7)
