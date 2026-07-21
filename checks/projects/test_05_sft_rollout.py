from __future__ import annotations

import torch
from student_alignment import (
    masked_token_log_probs,
    sample_grouped,
    sft_loss,
    sft_train_step,
    verifiable_answer_reward,
)
from torch import nn
from torch.nn import functional as F


class TinyLM(nn.Module):
    def __init__(self, vocab_size: int = 7) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 8)
        self.output = nn.Linear(8, vocab_size)

    def forward(self, tokens):
        return self.output(self.embedding(tokens))


def test_response_only_shift_matches_manual_log_probs() -> None:
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 7, dtype=torch.float64)
    tokens = torch.randint(0, 7, (2, 5))
    mask = torch.tensor([[0, 0, 0, 1, 1], [0, 0, 1, 1, 0]], dtype=torch.bool)
    actual = masked_token_log_probs(logits, tokens, mask)
    expected = (
        F.log_softmax(logits[:, :-1], dim=-1).gather(-1, tokens[:, 1:].unsqueeze(-1)).squeeze(-1)
    )
    expected = expected * mask[:, 1:]
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(sft_loss(logits, tokens, mask), -expected.sum() / mask.sum())


def test_sft_step_changes_policy_with_finite_detached_loss() -> None:
    torch.manual_seed(1)
    model = TinyLM()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.05)
    tokens = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]])
    mask = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]], dtype=torch.bool)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    loss = sft_train_step(model, tokens, mask, optimizer)
    assert torch.isfinite(loss) and not loss.requires_grad
    assert any(
        not torch.equal(old, new) for old, new in zip(before, model.parameters(), strict=True)
    )


def test_grouped_rollout_is_seeded_and_keeps_behavior_log_probs() -> None:
    torch.manual_seed(2)
    model = TinyLM()
    prompts = torch.tensor([[0, 1], [2, 3]])
    first = sample_grouped(
        model,
        prompts,
        group_size=3,
        max_new_tokens=2,
        temperature=0.8,
        generator=torch.Generator().manual_seed(9),
    )
    second = sample_grouped(
        model,
        prompts,
        group_size=3,
        max_new_tokens=2,
        temperature=0.8,
        generator=torch.Generator().manual_seed(9),
    )
    torch.testing.assert_close(first.sequences, second.sequences)
    torch.testing.assert_close(first.old_token_log_probs, second.old_token_log_probs)
    assert first.sequences.shape == first.response_mask.shape == (2, 3, 4)
    assert first.old_token_log_probs.shape == (2, 3, 3)
    assert not first.response_mask[:, :, :2].any() and first.response_mask[:, :, 2:].all()
    assert torch.isfinite(first.old_token_log_probs[first.response_mask[:, :, 1:]]).all()


def test_verifiable_reward_scores_first_response_token_only() -> None:
    sequences = torch.tensor([[[9, 8, 4, 0], [9, 8, 3, 4], [9, 8, 4, 7]]])
    mask = torch.tensor([[[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]]], dtype=torch.bool)
    rewards = verifiable_answer_reward(sequences, mask, torch.tensor([4]))
    torch.testing.assert_close(rewards, torch.tensor([[1.0, 0.0, 1.0]]))
