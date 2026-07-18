import torch

from exercises.checks._loader import load_starter

student = load_starter("10_moe_router.py")


def test_topk_route_selects_and_renormalizes() -> None:
    logits = torch.tensor([[5.0, 1.0, 3.0, -2.0], [0.0, 4.0, 2.0, 3.0]])
    probabilities, indices, weights = student.topk_route(logits, top_k=2)
    assert probabilities.shape == (2, 4)
    assert indices.tolist() == [[0, 2], [1, 3]]
    torch.testing.assert_close(probabilities.sum(dim=-1), torch.ones(2))
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(2))
    expected = torch.gather(probabilities, 1, indices)
    expected = expected / expected.sum(dim=-1, keepdim=True)
    torch.testing.assert_close(weights, expected)


def test_balanced_uniform_router_has_unit_balance_loss() -> None:
    probabilities = torch.full((4, 4), 0.25)
    selected = torch.tensor([[0], [1], [2], [3]])
    torch.testing.assert_close(
        student.switch_balance_loss(probabilities, selected), torch.tensor(1.0)
    )


def test_collapsed_router_is_penalized_more_than_balanced_router() -> None:
    collapsed_probabilities = torch.tensor([[0.9, 0.05, 0.03, 0.02]]).repeat(4, 1)
    collapsed_indices = torch.zeros(4, 1, dtype=torch.long)
    collapsed = student.switch_balance_loss(collapsed_probabilities, collapsed_indices)
    balanced = student.switch_balance_loss(torch.full((4, 4), 0.25), torch.arange(4)[:, None])
    assert collapsed > balanced
