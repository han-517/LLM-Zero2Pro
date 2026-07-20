import math

from checks.exercises._loader import load_starter

student = load_starter("05_moe_capacity.py")


def test_capacity_uses_all_topk_assignments_and_rounds_up() -> None:
    assert student.expert_capacity(100, 8, 2, 1.25) == 32
    assert student.expert_capacity(7, 4, 1, 1.0) == math.ceil(7 / 4)


def test_overflow_is_capped_and_counted() -> None:
    loads = [50, 30, 20, 20, 20, 20, 20, 20]
    accepted, dropped = student.accepted_and_dropped(loads, 32)
    assert accepted == [32, 30, 20, 20, 20, 20, 20, 20]
    assert dropped == 18
    assert sum(accepted) + dropped == sum(loads)


def test_no_overflow_drops_nothing() -> None:
    accepted, dropped = student.accepted_and_dropped([0, 2, 4], 4)
    assert accepted == [0, 2, 4]
    assert dropped == 0
