import torch
from torch import nn

from llm_from_scratch.post_training import (
    LoRALinear,
    causal_sft_loss,
    dpo_loss,
    group_relative_advantages,
    masked_sft_loss,
    pairwise_reward_loss,
)


def test_lora_initially_matches_base_and_only_adapters_train() -> None:
    torch.manual_seed(30)
    base = nn.Linear(5, 3)
    module = LoRALinear(base, rank=2, alpha=4)
    x = torch.randn(4, 5)
    torch.testing.assert_close(module(x), base(x))
    module(x).sum().backward()
    assert module.a.grad is not None and module.b.grad is not None
    assert base.weight.grad is None


def test_masked_sft_ignores_prompt_tokens() -> None:
    logits = torch.tensor([[[9.0, -9.0], [-9.0, 9.0], [-9.0, 9.0]]])
    labels = torch.tensor([[1, 1, 1]])
    mask = torch.tensor([[0, 1, 1]])
    good = masked_sft_loss(logits, labels, mask)
    logits[:, 0] = torch.tensor([-100.0, 100.0])
    unchanged = masked_sft_loss(logits, labels, mask)
    torch.testing.assert_close(good, unchanged)


def test_causal_sft_shifts_logits_labels_and_answer_mask_together() -> None:
    token_ids = torch.tensor([[0, 1, 2]])
    assistant_mask = torch.tensor([[0, 0, 1]])
    logits = torch.zeros(1, 3, 3)
    logits[:, 1, 2] = 8.0
    good = causal_sft_loss(logits, token_ids, assistant_mask)

    logits[:, 0, 1] = -100.0
    prompt_changed = causal_sft_loss(logits, token_ids, assistant_mask)
    torch.testing.assert_close(good, prompt_changed)
    assert good < 0.001


def test_pairwise_reward_improves_when_margin_grows() -> None:
    small = pairwise_reward_loss(torch.tensor([1.0]), torch.tensor([0.9]))
    large = pairwise_reward_loss(torch.tensor([3.0]), torch.tensor([0.0]))
    assert large < small


def test_dpo_prefers_better_policy_margin() -> None:
    loss_good, chosen, rejected = dpo_loss(
        torch.tensor([2.0]), torch.tensor([0.0]), torch.tensor([0.5]), torch.tensor([0.5]), beta=1
    )
    loss_bad, _, _ = dpo_loss(
        torch.tensor([0.0]), torch.tensor([2.0]), torch.tensor([0.5]), torch.tensor([0.5]), beta=1
    )
    assert loss_good < loss_bad
    assert chosen > rejected


def test_group_relative_advantages_center_and_constant_case() -> None:
    rewards = torch.tensor([[1.0, 2.0, 3.0], [4.0, 4.0, 4.0]])
    advantages = group_relative_advantages(rewards)
    torch.testing.assert_close(advantages[0].mean(), torch.tensor(0.0), atol=1e-6, rtol=0)
    assert torch.count_nonzero(advantages[1]) == 0

