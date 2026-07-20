import pytest
import torch

from checks.exercises._loader import load_starter

student = load_starter("15_adamw_schedule.py")


def test_first_adamw_step_matches_scalar_oracle() -> None:
    p, m, v = student.adamw_step(
        torch.tensor([1.0]),
        torch.tensor([2.0]),
        torch.zeros(1),
        torch.zeros(1),
        step=1,
        learning_rate=0.1,
        weight_decay=0.1,
    )
    torch.testing.assert_close(m, torch.tensor([0.2]))
    torch.testing.assert_close(v, torch.tensor([0.004]))
    assert p.item() == pytest.approx(0.89, abs=1e-6)


def test_schedule_boundaries() -> None:
    assert student.warmup_cosine_lr(0, 2, 10, 1.0) == 0.0
    assert student.warmup_cosine_lr(2, 2, 10, 1.0) == pytest.approx(1.0)
    assert student.warmup_cosine_lr(10, 2, 10, 1.0) == pytest.approx(0.0)
