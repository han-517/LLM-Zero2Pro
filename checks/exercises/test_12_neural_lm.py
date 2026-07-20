import pytest
import torch

from checks.exercises._loader import load_starter

student = load_starter("12_neural_lm.py")


def test_bigram_counts_do_not_wrap_last_to_first() -> None:
    counts = student.bigram_counts(torch.tensor([0, 1, 0]), vocab_size=2)
    torch.testing.assert_close(counts, torch.tensor([[0, 1], [1, 0]]))


def test_rnn_step_shape_and_value() -> None:
    x = torch.tensor([[1.0, 2.0]])
    h = torch.tensor([[0.5]])
    out = student.rnn_step(x, h, torch.ones(2, 1), torch.ones(1, 1), torch.zeros(1))
    torch.testing.assert_close(out, torch.tanh(torch.tensor([[3.5]])))


def test_invalid_vocab_is_rejected() -> None:
    with pytest.raises(ValueError):
        student.bigram_counts(torch.tensor([0, 1]), vocab_size=0)
