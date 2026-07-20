from pathlib import Path

import nbformat
import pytest
import torch
from nbclient import NotebookClient

from llm_from_scratch.neural_lm import (
    BigramLanguageModel,
    ElmanRNNLanguageModel,
    FixedWindowMLP,
    make_document_windows,
    make_next_token_windows,
    split_documents,
)

ROOT = Path(__file__).resolve().parents[1]


def test_document_split_is_deterministic_disjoint_and_complete() -> None:
    documents = [f"document-{index}" for index in range(10)]
    first = split_documents(documents, seed=7)
    second = split_documents(documents, seed=7)
    assert first == second
    groups = [set(first.train), set(first.validation), set(first.test)]
    assert all(groups[left].isdisjoint(groups[right]) for left in range(3) for right in range(left))
    assert set.union(*groups) == set(documents)
    assert all(groups)


@pytest.mark.parametrize(
    ("train_fraction", "validation_fraction"),
    [(0.0, 0.1), (0.8, 0.0), (0.9, 0.2)],
)
def test_document_split_rejects_invalid_fractions(
    train_fraction: float, validation_fraction: float
) -> None:
    with pytest.raises(ValueError):
        split_documents(
            ["a", "b", "c"],
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
        )


def test_windows_have_exact_target_shift_and_never_cross_documents() -> None:
    contexts, targets = make_next_token_windows(torch.tensor([0, 1, 2, 3]), 2)
    assert torch.equal(contexts, torch.tensor([[0, 1], [1, 2]]))
    assert torch.equal(targets, torch.tensor([2, 3]))

    contexts, targets = make_document_windows(
        [torch.tensor([0, 1, 2]), torch.tensor([10, 11, 12])], 2
    )
    assert torch.equal(contexts, torch.tensor([[0, 1], [10, 11]]))
    assert torch.equal(targets, torch.tensor([2, 12]))


def test_bigram_shapes_loss_gradients_and_greedy_generation() -> None:
    model = BigramLanguageModel(vocab_size=5)
    tokens = torch.tensor([[0, 1, 2], [1, 2, 3]])
    targets = torch.tensor([[1, 2, 3], [2, 3, 4]])
    logits, loss = model(tokens, targets)
    assert logits.shape == (2, 3, 5)
    assert loss is not None
    loss.backward()
    assert model.transition_logits.weight.grad is not None
    generated = model.generate(torch.tensor([[0, 1]]), 3, temperature=0)
    assert generated.shape == (1, 5)


def test_fixed_window_mlp_can_overfit_a_tiny_mapping() -> None:
    torch.manual_seed(41)
    model = FixedWindowMLP(7, context_size=2, embedding_dim=4, hidden_dim=12)
    contexts = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 4]])
    targets = torch.tensor([2, 3, 4, 5])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.08)
    with torch.no_grad():
        _, initial = model(contexts, targets)
    for _ in range(50):
        _, loss = model(contexts, targets)
        assert loss is not None
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        logits, final = model(contexts, targets)
    assert initial is not None and final is not None
    assert final < initial * 0.1
    assert torch.equal(logits.argmax(dim=-1), targets)


def test_elman_rnn_is_causal_and_all_parameters_receive_gradients() -> None:
    torch.manual_seed(42)
    model = ElmanRNNLanguageModel(9, embedding_dim=5, hidden_dim=7)
    tokens = torch.tensor([[0, 1, 2, 3]])
    targets = torch.tensor([[1, 2, 3, 4]])
    before, loss, hidden = model(tokens, targets)
    changed = tokens.clone()
    changed[0, -1] = 8
    after, _, _ = model(changed)
    torch.testing.assert_close(before[:, :-1], after[:, :-1])
    assert hidden.shape == (1, 7)
    assert loss is not None
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_neural_lm_notebook_executes_on_cpu() -> None:
    path = ROOT / "learning" / "labs" / "02_neural_language_models.ipynb"
    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=90,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    )
    client.execute()
