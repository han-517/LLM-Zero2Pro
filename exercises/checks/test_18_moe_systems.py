import pytest
import torch

from exercises.checks._loader import load_starter

student = load_starter("18_moe_systems.py")


def test_capacity_rounds_up() -> None:
    assert student.expert_capacity(5, 4, 2, 1.0) == 3
    with pytest.raises(ValueError):
        student.expert_capacity(5, 0, 2, 1.0)


def test_dispatch_counts_all_selected_slots() -> None:
    routed = torch.tensor([[0, 1], [1, 2], [2, 2]])
    torch.testing.assert_close(student.dispatch_counts(routed, 3), torch.tensor([1, 2, 3]))
