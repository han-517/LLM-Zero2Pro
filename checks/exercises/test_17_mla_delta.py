import pytest
import torch

from checks.exercises._loader import load_starter

student = load_starter("17_mla_delta.py")


def test_latent_cache_accounting() -> None:
    assert student.latent_cache_bytes(2, 8, 16, bytes_per_element=2) == 512
    with pytest.raises(ValueError):
        student.latent_cache_bytes(0, 8, 16)


def test_delta_update_corrects_previous_prediction() -> None:
    state = torch.zeros(2, 2)
    out = student.delta_update(
        state, torch.tensor([1.0, 0.0]), torch.tensor([2.0, 3.0]), torch.tensor(1.0)
    )
    torch.testing.assert_close(out, torch.tensor([[2.0, 0.0], [3.0, 0.0]]))
    torch.testing.assert_close(state, torch.zeros_like(state))
