from __future__ import annotations

import copy
import math

import pytest
import torch
from student_alignment import group_advantages, grpo_loss, rlvr_train_step, sample_grouped
from torch import nn


class TinyLM(nn.Module):
    def __init__(self, vocab_size: int = 6) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 8)
        self.output = nn.Linear(8, vocab_size)

    def forward(self, tokens):
        return self.output(self.embedding(tokens))


def test_group_advantages_handle_zero_variance_without_nan() -> None:
    rewards = torch.tensor([[1.0, 2.0, 3.0], [4.0, 4.0, 4.0]])
    advantages = group_advantages(rewards)
    torch.testing.assert_close(advantages[0].mean(), torch.tensor(0.0), atol=1e-6, rtol=0)
    torch.testing.assert_close(advantages[0].std(unbiased=False), torch.tensor(1.0))
    torch.testing.assert_close(advantages[1], torch.zeros(3))


def test_grpo_clipping_is_sign_aware_and_response_masked() -> None:
    current = torch.tensor([[[math.log(1.1)], [math.log(2.0)]]])
    old = torch.zeros_like(current)
    reference = torch.zeros_like(current)
    advantages = torch.tensor([[1.0, -1.0]])
    mask = torch.ones_like(current, dtype=torch.bool)
    loss = grpo_loss(current, old, reference, advantages, mask, clip_epsilon=0.2, beta=0.0)
    torch.testing.assert_close(loss, torch.tensor(0.45), atol=1e-6, rtol=0)
    with pytest.raises(ValueError):
        grpo_loss(current, old, reference, advantages, torch.zeros_like(mask))


def test_rlvr_step_updates_policy_but_not_reference() -> None:
    torch.manual_seed(3)
    policy = TinyLM()
    reference = copy.deepcopy(policy)
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=0.02)
    rollout = sample_grouped(
        policy,
        torch.tensor([[0, 1], [2, 3]]),
        group_size=3,
        max_new_tokens=2,
        temperature=1.0,
        generator=torch.Generator().manual_seed(4),
    )
    rewards = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    policy_before = [parameter.detach().clone() for parameter in policy.parameters()]
    reference_before = [parameter.detach().clone() for parameter in reference.parameters()]
    metrics = rlvr_train_step(policy, reference, rollout, rewards, optimizer, beta=0.01)
    assert math.isfinite(metrics.loss) and math.isfinite(metrics.gradient_norm)
    assert any(
        not torch.equal(old, new)
        for old, new in zip(policy_before, policy.parameters(), strict=True)
    )
    assert all(
        torch.equal(old, new)
        for old, new in zip(reference_before, reference.parameters(), strict=True)
    )
