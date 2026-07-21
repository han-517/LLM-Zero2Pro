from __future__ import annotations

import torch
from student_systems import ddp_train_step
from torch import nn
from torch.nn import functional as F


class TinyNextTokenModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(8, 6)
        self.output = nn.Linear(6, 8)

    def forward(self, tokens):
        return self.output(self.embedding(tokens))


def test_distributed_compatible_step_matches_shifted_ce_and_updates_parameters() -> None:
    torch.manual_seed(0)
    model = TinyNextTokenModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    inputs = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])
    targets = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 0]])
    with torch.no_grad():
        expected = F.cross_entropy(model(inputs).reshape(-1, 8), targets.reshape(-1))
    before = [parameter.detach().clone() for parameter in model.parameters()]
    actual = ddp_train_step(model, inputs, targets, optimizer)
    torch.testing.assert_close(actual, expected)
    assert not actual.requires_grad
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
    assert any(
        not torch.equal(old, new) for old, new in zip(before, model.parameters(), strict=True)
    )
