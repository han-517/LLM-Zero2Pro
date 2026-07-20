import pytest

from checks.exercises._loader import load_starter

student = load_starter("11_autograd.py")


def test_shared_branch_gradients_match_finite_difference() -> None:
    dx, dy = student.branch_gradients(2.0, -0.5)
    assert dx == pytest.approx(1.0)
    assert dy == pytest.approx(4.0)


def test_zero_intermediate_still_has_well_defined_gradients() -> None:
    assert student.branch_gradients(2.0, -1.0) == pytest.approx((0.0, 0.0))
