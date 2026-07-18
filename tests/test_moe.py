import torch

from llm_from_scratch.moe import Expert, TopKMoE, upcycle_expert


def test_topk_moe_shapes_aux_and_gradients() -> None:
    torch.manual_seed(20)
    module = TopKMoE(8, 12, num_experts=4, top_k=2, capacity_factor=2.0)
    x = torch.randn(2, 5, 8, requires_grad=True)
    output, aux = module(x)
    assert output.shape == x.shape
    assert aux["top_indices"].shape == (10, 2)
    assert aux["accepted_load"].sum() == 20
    (output.square().mean() + 0.01 * aux["balance_loss"] + 0.001 * aux["z_loss"]).backward()
    assert x.grad is not None
    assert module.router.weight.grad is not None


def test_capacity_overflow_is_explicit() -> None:
    module = TopKMoE(4, 8, num_experts=4, top_k=2, capacity_factor=1.0)
    torch.nn.init.zeros_(module.router.weight)
    _, aux = module(torch.ones(1, 4, 4))
    assert aux["dropped"].any()
    assert torch.all(aux["accepted_load"] <= aux["capacity"])


def test_uniform_router_switch_balance_loss_is_one() -> None:
    module = TopKMoE(4, 8, num_experts=4, top_k=1, capacity_factor=10.0)
    torch.nn.init.zeros_(module.router.weight)
    _, aux = module(torch.randn(3, 4))
    torch.testing.assert_close(aux["balance_loss"], torch.tensor(1.0))


def test_upcycling_copies_dense_expert() -> None:
    torch.manual_seed(21)
    source = Expert(4, 8)
    target = TopKMoE(4, 8, num_experts=3, top_k=1)
    upcycle_expert(source, target)
    x = torch.randn(2, 4)
    expected = source(x)
    for expert in target.experts:
        torch.testing.assert_close(expert(x), expected)
    assert target.shared is None



def test_upcycling_with_shared_expert_preserves_dense_output_scale() -> None:
    torch.manual_seed(22)
    source = Expert(4, 8)
    target = TopKMoE(
        4,
        8,
        num_experts=3,
        top_k=1,
        capacity_factor=10.0,
        shared_expert=True,
    )
    upcycle_expert(source, target)
    x = torch.randn(5, 4)
    actual, stats = target(x)
    assert not stats["dropped"].any()
    torch.testing.assert_close(actual, source(x))
