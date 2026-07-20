import torch

from checks.exercises._loader import load_starter

student = load_starter("04_sft_shift.py")


def test_labels_and_answer_mask_shift_together() -> None:
    tokens = torch.tensor([[10, 11, 12, 13, 14], [20, 21, 22, 23, 24]])
    mask = torch.tensor([[0, 0, 0, 1, 1], [0, 1, 1, 0, 0]])
    labels, shifted_mask = student.causal_sft_targets(tokens, mask)
    assert labels.tolist() == [[11, 12, 13, 14], [21, 22, 23, 24]]
    assert shifted_mask.tolist() == [[0, 0, 1, 1], [1, 1, 0, 0]]
    assert labels.shape == shifted_mask.shape == (2, 4)


def test_inputs_are_not_modified() -> None:
    tokens = torch.tensor([[1, 2, 3]])
    mask = torch.tensor([[0, 1, 1]])
    original_tokens, original_mask = tokens.clone(), mask.clone()
    student.causal_sft_targets(tokens, mask)
    assert torch.equal(tokens, original_tokens)
    assert torch.equal(mask, original_mask)
