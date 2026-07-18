import math

import pytest
import torch

from llm_from_scratch.training import (
    adamw_step_,
    detect_ngram_contamination,
    deterministic_mixture_sample,
    evaluate_token_logits,
    exact_deduplicate,
    pack_documents,
    training_memory_ledger,
    warmup_cosine_lr,
)


def test_exact_deduplicate_preserves_first_occurrence_without_normalizing() -> None:
    unique, duplicate_indices = exact_deduplicate(["A  B", "A B", "A  B", ""])
    assert unique == ["A  B", "A B", ""]
    assert duplicate_indices == [2]


def test_ngram_contamination_is_casefolded_and_marks_only_overlap() -> None:
    training = ["The QUICK brown fox jumps", "unrelated short text"]
    evaluation = ["quick brown fox sleeps"]
    assert detect_ngram_contamination(training, evaluation, ngram_size=3) == [True, False]
    with pytest.raises(ValueError, match="ngram_size"):
        detect_ngram_contamination(training, evaluation, ngram_size=0)


def test_deterministic_mixture_keeps_source_provenance() -> None:
    sources = {"web": [1, 2], "code": [3]}
    weights = {"web": 0.7, "code": 0.3}
    first = deterministic_mixture_sample(sources, weights, 20, seed=7)
    second = deterministic_mixture_sample(sources, weights, 20, seed=7)
    assert first == second
    assert all(name in sources and value in sources[name] for name, value in first)
    with pytest.raises(ValueError, match="相同的来源"):
        deterministic_mixture_sample(sources, {"web": 1.0}, 2, seed=0)


def test_pack_documents_inserts_eos_masks_boundaries_and_padding() -> None:
    input_ids, loss_mask = pack_documents(
        [[10, 11], torch.tensor([20])], block_size=4, eos_token_id=2, pad_token_id=0
    )
    assert input_ids.tolist() == [[10, 11, 2, 20], [2, 0, 0, 0]]
    assert loss_mask.tolist() == [
        [False, True, True, False],
        [True, False, False, False],
    ]
    empty_ids, empty_mask = pack_documents([], block_size=4, eos_token_id=2, pad_token_id=0)
    assert empty_ids.shape == empty_mask.shape == (0, 4)


def test_adamw_single_step_matches_pytorch() -> None:
    reference = torch.nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.float64))
    actual = reference.detach().clone()
    gradient = torch.tensor([0.25, -0.5], dtype=torch.float64)
    optimizer = torch.optim.AdamW(
        [reference], lr=0.01, betas=(0.8, 0.95), eps=1e-9, weight_decay=0.1
    )
    reference.grad = gradient.clone()
    optimizer.step()

    exp_avg = torch.zeros_like(actual)
    exp_avg_sq = torch.zeros_like(actual)
    adamw_step_(
        actual,
        gradient,
        exp_avg,
        exp_avg_sq,
        step=1,
        learning_rate=0.01,
        beta1=0.8,
        beta2=0.95,
        eps=1e-9,
        weight_decay=0.1,
    )
    torch.testing.assert_close(actual, reference.detach(), atol=1e-12, rtol=1e-12)


def test_warmup_cosine_schedule_has_named_endpoints() -> None:
    assert warmup_cosine_lr(0, total_steps=100, max_lr=1e-3, warmup_steps=10) == 0
    assert warmup_cosine_lr(10, total_steps=100, max_lr=1e-3, warmup_steps=10) == 1e-3
    assert warmup_cosine_lr(
        100, total_steps=100, max_lr=1e-3, warmup_steps=10, min_lr=1e-5
    ) == pytest.approx(1e-5)


def test_training_memory_ledger_shards_only_requested_components() -> None:
    ledger = training_memory_ledger(
        10,
        activation_bytes=7,
        world_size=2,
        shard_parameters=True,
        shard_gradients=True,
        shard_optimizer=True,
    )
    assert ledger == {
        "parameters": 10,
        "gradients": 10,
        "optimizer_states": 40,
        "master_weights": 20,
        "activations": 7,
        "total": 87,
    }


def test_evaluate_token_logits_respects_mask() -> None:
    logits = torch.tensor([[[8.0, -8.0], [-8.0, 8.0], [8.0, -8.0]]])
    labels = torch.tensor([[0, 1, 1]])
    result = evaluate_token_logits(logits, labels, torch.tensor([[1, 1, 0]]))
    assert result["token_count"] == 2
    assert result["accuracy"] == 1.0
    assert result["loss"] < 1e-5
    assert result["perplexity"] == pytest.approx(math.exp(result["loss"]))


def test_evaluate_rejects_empty_mask() -> None:
    with pytest.raises(ValueError, match="有效评测 token"):
        evaluate_token_logits(
            torch.zeros(1, 2, 3), torch.zeros(1, 2, dtype=torch.long), torch.zeros(1, 2)
        )
