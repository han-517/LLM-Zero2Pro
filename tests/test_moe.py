import pytest
import torch

from llm_from_scratch.moe import (
    Expert,
    TopKMoE,
    expert_parallel_communication_ledger,
    moe_parameter_accounting,
    upcycle_expert,
)


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


def test_capacity_overflow_is_explicit_and_conserves_assignments() -> None:
    module = TopKMoE(4, 8, num_experts=4, top_k=2, capacity_factor=1.0)
    torch.nn.init.zeros_(module.router.weight)
    _, aux = module(torch.ones(1, 4, 4))
    assert aux["dropped"].any()
    assert torch.all(aux["accepted_load"] <= aux["capacity"])
    assert aux["selected_load"].sum() == 8
    assert aux["accepted_assignments"] + aux["dropped_assignments"] == 8
    torch.testing.assert_close(aux["accepted_per_token"], (~aux["dropped"]).sum(dim=-1))


def test_uniform_router_switch_balance_loss_is_one() -> None:
    module = TopKMoE(4, 8, num_experts=4, top_k=1, capacity_factor=10.0)
    torch.nn.init.zeros_(module.router.weight)
    _, aux = module(torch.randn(3, 4))
    torch.testing.assert_close(aux["balance_loss"], torch.tensor(1.0))
    # loss 值为 1 不代表硬选择均匀；tie-breaking 仍可能集中到一个专家。
    assert torch.count_nonzero(aux["selected_load"]) == 1


def test_normalized_top1_has_no_main_task_router_gradient_but_auxiliary_does() -> None:
    torch.manual_seed(23)
    module = TopKMoE(4, 8, num_experts=4, top_k=1, capacity_factor=10.0)
    output, aux = module(torch.randn(6, 4))
    output.sum().backward()
    torch.testing.assert_close(module.router.weight.grad, torch.zeros_like(module.router.weight))
    assert aux["normalized_top1"]

    module.zero_grad(set_to_none=True)
    output, aux = module(torch.randn(6, 4))
    (output.sum() + aux["balance_loss"] + 0.01 * aux["z_loss"]).backward()
    assert torch.count_nonzero(module.router.weight.grad) > 0


def test_softmax_and_sigmoid_routing_contracts_are_observable() -> None:
    torch.manual_seed(24)
    softmax_moe = TopKMoE(4, 8, num_experts=4, top_k=2, capacity_factor=10)
    sigmoid_moe = TopKMoE(
        4,
        8,
        num_experts=4,
        top_k=2,
        capacity_factor=10,
        routing_mode="sigmoid",
        normalize_topk=False,
    )
    sigmoid_moe.load_state_dict(softmax_moe.state_dict())
    x = torch.randn(3, 4)
    _, softmax_stats = softmax_moe(x)
    _, sigmoid_stats = sigmoid_moe(x)
    torch.testing.assert_close(softmax_stats["top_weights"].sum(-1), torch.ones(3))
    assert not torch.allclose(sigmoid_stats["top_weights"].sum(-1), torch.ones(3))
    torch.testing.assert_close(softmax_stats["top_indices"], sigmoid_stats["top_indices"])


def test_router_compute_remains_float32_after_module_dtype_conversion() -> None:
    module = TopKMoE(4, 8, num_experts=4, top_k=2, capacity_factor=10).to(torch.bfloat16)
    output, aux = module(torch.randn(3, 4, dtype=torch.bfloat16))
    assert output.dtype == torch.bfloat16
    assert aux["router_logits"].dtype == torch.float32
    (output.float().sum() + aux["balance_loss"]).backward()
    assert module.router.weight.grad is not None
    assert module.router.weight.grad.dtype == torch.bfloat16


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
    with pytest.raises(ValueError, match="noise_std"):
        upcycle_expert(source, target, noise_std=-1)


def test_moe_parameter_and_communication_ledgers_state_assumptions() -> None:
    parameters = moe_parameter_accounting(4, 8, num_experts=4, top_k=2, shared_experts=1)
    assert parameters == {
        "router_parameters": 16,
        "parameters_per_expert": 96,
        "total_parameters": 496,
        "active_parameters_per_token": 304,
    }
    communication = expert_parallel_communication_ledger(
        10, d_model=4, top_k=2, element_bytes=2, world_size=2
    )
    assert communication == {
        "assignments": 20,
        "dispatch_bytes": 160,
        "combine_bytes": 160,
        "total_bytes": 320,
        "remote_bytes_uniform_assumption": 160,
    }
