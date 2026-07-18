import math

import pytest
import torch
from torch import nn

from llm_from_scratch.post_training import (
    LoRALinear,
    causal_sft_loss,
    dpo_loss,
    group_relative_advantages,
    masked_sft_loss,
    pairwise_reward_loss,
    response_only_collator,
    sequence_logprob,
    toy_grpo_clipped_loss,
    toy_ppo_clipped_loss,
)


def test_lora_initially_matches_base_and_only_adapters_train() -> None:
    torch.manual_seed(30)
    base = nn.Linear(5, 3)
    module = LoRALinear(base, rank=2, alpha=4)
    x = torch.randn(4, 5)
    torch.testing.assert_close(module(x), base(x))
    module(x).sum().backward()
    assert module.a.grad is not None and module.b.grad is not None
    assert base.weight.grad is None and base.bias.grad is None


def test_lora_merge_and_unmerge_preserve_eval_output() -> None:
    torch.manual_seed(31)
    module = LoRALinear(nn.Linear(4, 3), rank=2, alpha=4, dropout=0.2)
    torch.nn.init.normal_(module.b)
    module.eval()
    x = torch.randn(5, 4)
    expected = module(x)
    base_weight = module.base.weight.detach().clone()
    module.merge_()
    assert module.is_merged
    torch.testing.assert_close(module(x), expected)
    module.unmerge_()
    assert not module.is_merged
    torch.testing.assert_close(module(x), expected)
    torch.testing.assert_close(module.base.weight, base_weight)


def test_lora_rejects_merging_training_dropout_semantics() -> None:
    module = LoRALinear(nn.Linear(3, 2), dropout=0.1)
    with pytest.raises(RuntimeError, match="eval"):
        module.merge_()


def test_response_only_collator_pads_and_masks_prompt() -> None:
    batch = response_only_collator([[1, 2, 3], [4, 5]], [[0, 0, 1], [0, 1]], pad_token_id=0)
    assert batch["input_ids"].tolist() == [[1, 2, 3], [4, 5, 0]]
    assert batch["attention_mask"].tolist() == [[True, True, True], [True, True, False]]
    assert batch["assistant_mask"].tolist() == [[False, False, True], [False, True, False]]
    assert batch["labels"].tolist() == [[-100, -100, 3], [-100, 5, -100]]


def test_sequence_logprob_shifts_and_scores_only_response() -> None:
    token_ids = torch.tensor([[0, 1, 2]])
    response_mask = torch.tensor([[0, 0, 1]])
    logits = torch.zeros(1, 3, 3)
    logits[:, 1, 2] = 2.0
    actual = sequence_logprob(logits, token_ids, response_mask)
    expected = torch.log_softmax(logits[:, 1], dim=-1)[:, 2]
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        sequence_logprob(logits, token_ids, response_mask, reduction="mean"), expected
    )


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
    with pytest.raises(ValueError, match="不能为空"):
        pairwise_reward_loss(torch.tensor([]), torch.tensor([]))


def test_dpo_prefers_better_policy_margin() -> None:
    loss_good, chosen, rejected = dpo_loss(
        torch.tensor([2.0]), torch.tensor([0.0]), torch.tensor([0.5]), torch.tensor([0.5]), beta=1
    )
    loss_bad, _, _ = dpo_loss(
        torch.tensor([0.0]), torch.tensor([2.0]), torch.tensor([0.5]), torch.tensor([0.5]), beta=1
    )
    assert loss_good < loss_bad
    assert chosen > rejected
    with pytest.raises(ValueError, match="beta"):
        dpo_loss(*(torch.tensor([0.0]) for _ in range(4)), beta=0)


def test_group_relative_advantages_center_and_constant_case() -> None:
    rewards = torch.tensor([[1.0, 2.0, 3.0], [4.0, 4.0, 4.0]])
    advantages = group_relative_advantages(rewards)
    torch.testing.assert_close(advantages[0].mean(), torch.tensor(0.0), atol=1e-6, rtol=0)
    assert torch.count_nonzero(advantages[1]) == 0
    with pytest.raises(ValueError, match="group"):
        group_relative_advantages(torch.ones(2, 1))


def test_toy_ppo_objective_exposes_clipping_without_claiming_full_ppo() -> None:
    new_logp = torch.tensor([math.log(1.5)])
    loss = toy_ppo_clipped_loss(new_logp, torch.zeros(1), torch.ones(1), clip_epsilon=0.2)
    torch.testing.assert_close(loss, torch.tensor(-1.2))


def test_toy_grpo_constant_group_reward_has_zero_update() -> None:
    new_logp = torch.zeros(1, 3, 2, requires_grad=True)
    old_logp = torch.zeros_like(new_logp)
    rewards = torch.ones(1, 3)
    response_mask = torch.ones_like(new_logp, dtype=torch.bool)
    loss = toy_grpo_clipped_loss(new_logp, old_logp, rewards, response_mask)
    torch.testing.assert_close(loss, torch.tensor(0.0))
    loss.backward()
    torch.testing.assert_close(new_logp.grad, torch.zeros_like(new_logp))


def test_strict_validation_rejects_nonfinite_values_and_empty_response() -> None:
    with pytest.raises(ValueError, match="有限"):
        sequence_logprob(
            torch.full((1, 2, 2), float("nan")),
            torch.tensor([[0, 1]]),
            torch.tensor([[0, 1]]),
        )
    with pytest.raises(ValueError, match="assistant token"):
        response_only_collator([[1, 2]], [[0, 0]], pad_token_id=0)
    with pytest.raises(ValueError, match="response_mask"):
        sequence_logprob(
            torch.zeros(1, 2, 2),
            torch.tensor([[0, 1]]),
            torch.tensor([[0, 2]]),
        )
    with pytest.raises(ValueError, match="loss_mask"):
        masked_sft_loss(
            torch.zeros(1, 1, 2),
            torch.tensor([[1]]),
            torch.tensor([[2]]),
        )
    with pytest.raises(ValueError, match="不能是 -100"):
        masked_sft_loss(
            torch.zeros(1, 1, 2),
            torch.tensor([[-100]]),
            torch.tensor([[1]]),
        )
    with pytest.raises(ValueError, match="二值"):
        toy_ppo_clipped_loss(torch.zeros(1), torch.zeros(1), torch.ones(1), mask=torch.tensor([2]))
