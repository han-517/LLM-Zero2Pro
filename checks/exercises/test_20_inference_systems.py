from __future__ import annotations

import pytest
import torch

from checks.exercises._loader import load_starter

student = load_starter("20_inference_systems.py")


def test_quantization_bounds_and_zero_tensor() -> None:
    q, scale = student.symmetric_quantize(torch.tensor([-2.0, 0.0, 2.0]), bits=8)
    assert q.dtype in (torch.int8, torch.int16, torch.int32, torch.int64)
    assert q.min() >= -127 and q.max() <= 127
    zero_q, zero_scale = student.symmetric_quantize(torch.zeros(3), bits=8)
    assert torch.equal(zero_q, torch.zeros_like(zero_q))
    assert torch.isfinite(zero_scale)


def test_speculative_acceptance_ratio() -> None:
    out = student.acceptance_probability(torch.tensor([0.5, 0.2]), torch.tensor([0.25, 0.4]))
    torch.testing.assert_close(out, torch.tensor([0.5, 1.0]))
    with pytest.raises(ValueError):
        student.acceptance_probability(torch.tensor([-0.1]), torch.tensor([0.2]))


def test_page_table_prefix_sharing_copy_on_write_and_fragmentation() -> None:
    pages = student.PagedBlockTable(num_blocks=4, block_size=2)
    pages.create("parent")
    parent_plan = pages.append("parent", 3)
    assert parent_plan.slots == ((0, 0), (0, 1), (1, 0))
    assert parent_plan.copies == ()
    pages.fork("parent", "child")
    child_plan = pages.append("child", 1)
    assert child_plan.copies == ((1, 2, 1),)
    assert child_plan.slots == ((2, 1),)
    assert pages.block_table("parent") == (0, 1)
    assert pages.block_table("child") == (0, 2)
    assert pages.stats() == {
        "physical_blocks": 3,
        "free_blocks": 1,
        "shared_blocks": 1,
        "logical_tokens": 7,
        "internal_fragmentation_slots": 1,
    }
    pages.free("parent")
    assert pages.stats()["free_blocks"] == 2


def test_page_table_out_of_memory_is_atomic() -> None:
    pages = student.PagedBlockTable(num_blocks=2, block_size=2)
    pages.create("a")
    pages.append("a", 3)
    before = pages.block_table("a")
    with pytest.raises(MemoryError):
        pages.append("a", 2)
    assert pages.block_table("a") == before


def test_continuous_batching_admits_short_request_before_long_finishes() -> None:
    requests = [
        student.InferenceRequest("long", arrival_step=0, prompt_tokens=3, max_new_tokens=4),
        student.InferenceRequest("short", arrival_step=1, prompt_tokens=2, max_new_tokens=1),
    ]
    steps = student.simulate_continuous_batching(requests, max_batch_tokens=4)
    assert steps[0].prefill == ("long",) and steps[0].decode == ()
    assert steps[1].decode == ("long",) and steps[1].prefill == ("short",)
    assert steps[2].decode == ("long", "short")
    assert [request for step in steps for request in step.decode].count("long") == 4
    assert [request for step in steps for request in step.decode].count("short") == 1


def test_speculative_step_has_residual_and_bonus_paths() -> None:
    accepted, count = student.speculative_decode_step(
        torch.tensor([0, 1]),
        torch.tensor([[0.7, 0.3], [0.4, 0.6]]),
        torch.tensor([[0.7, 0.3], [0.4, 0.6], [0.25, 0.75]]),
        torch.tensor([0.99, 0.99]),
        torch.tensor([0.0, 0.0, 0.5]),
    )
    assert count == 2
    torch.testing.assert_close(accepted, torch.tensor([0, 1, 1]))

    rejected, count = student.speculative_decode_step(
        torch.tensor([0]),
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([[0.0, 1.0], [0.5, 0.5]]),
        torch.tensor([0.5]),
        torch.tensor([0.25, 0.25]),
    )
    assert count == 0
    torch.testing.assert_close(rejected, torch.tensor([1]))
