from __future__ import annotations

import torch
from student_lm.model import GPTConfig, TransformerLM
from student_lm.training import (
    AdamW,
    clip_grad_norm_,
    cross_entropy,
    load_checkpoint,
    next_token_batch,
    save_checkpoint,
    train_steps,
)
from torch import nn
from torch.nn import functional as F


def test_batch_and_cross_entropy_contracts() -> None:
    generator = torch.Generator().manual_seed(7)
    inputs, targets = next_token_batch(torch.arange(30), 4, 6, generator)
    assert inputs.shape == targets.shape == (4, 6)
    torch.testing.assert_close(targets, inputs + 1)

    torch.manual_seed(8)
    logits = torch.randn(2, 3, 11, dtype=torch.float64)
    labels = torch.randint(0, 11, (2, 3))
    expected = F.cross_entropy(logits.reshape(-1, 11), labels.reshape(-1))
    torch.testing.assert_close(cross_entropy(logits, labels), expected)


def test_adamw_and_gradient_clipping_match_pytorch() -> None:
    ours = nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.float64))
    reference = nn.Parameter(ours.detach().clone())
    gradient = torch.tensor([3.0, 4.0], dtype=torch.float64)
    ours.grad = gradient.clone()
    reference.grad = gradient.clone()
    norm = clip_grad_norm_([ours], 2.5)
    assert float(norm) == 5.0
    torch.testing.assert_close(ours.grad, gradient * 0.5)
    reference.grad = ours.grad.clone()

    ours_optimizer = AdamW([ours], lr=1e-2, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    reference_optimizer = torch.optim.AdamW(
        [reference], lr=1e-2, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1
    )
    ours_optimizer.step()
    reference_optimizer.step()
    torch.testing.assert_close(ours, reference)


def test_checkpoint_restores_model_optimizer_step_and_rng(tmp_path) -> None:
    torch.manual_seed(9)
    model = nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    generator = torch.Generator().manual_seed(10)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        path,
        model,
        optimizer,
        step=12,
        generator=generator,
        metadata={"tag": "public-test"},
    )
    expected_random = torch.rand(4, generator=generator)
    saved_weights = {name: value.detach().clone() for name, value in model.state_dict().items()}
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(100)
    generator.manual_seed(999)
    step, metadata = load_checkpoint(path, model, optimizer, generator=generator)
    assert step == 12 and metadata["tag"] == "public-test"
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, saved_weights[name])
    torch.testing.assert_close(torch.rand(4, generator=generator), expected_random)


def test_integrated_training_reduces_loss() -> None:
    torch.manual_seed(11)
    config = GPTConfig(vocab_size=8, d_model=16, num_heads=4, num_layers=1, d_ff=32)
    model = TransformerLM(config)
    optimizer = AdamW(model.parameters(), lr=1e-2, weight_decay=0.0)
    tokens = torch.tensor(([0, 1, 2, 3, 4, 5, 6, 7] * 40), dtype=torch.long)
    losses = train_steps(
        model,
        tokens,
        optimizer,
        steps=30,
        batch_size=8,
        sequence_length=16,
        generator=torch.Generator().manual_seed(12),
    )
    assert len(losses) == 30
    assert all(torch.isfinite(torch.tensor(losses)))
    assert sum(losses[-5:]) / 5 < sum(losses[:5]) / 5
