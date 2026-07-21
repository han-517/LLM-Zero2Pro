from __future__ import annotations

import time

import pytest
import torch
from student_systems import (
    assign_optimizer_states,
    benchmark_callable,
    bucket_tensors,
    contiguous_shards,
    gather_shards,
    profile_callable,
)


def test_benchmark_uses_warmup_sync_and_reports_robust_statistics() -> None:
    calls: list[str] = []

    def synchronize() -> None:
        calls.append("sync")

    def work() -> None:
        calls.append("work")
        time.sleep(0.0001)

    result = benchmark_callable(
        work, warmup=2, iterations=5, work_items=100, synchronize=synchronize
    )
    assert calls.count("work") == 7
    assert calls.count("sync") >= 10
    assert 0 < result.p20_seconds <= result.median_seconds <= result.p80_seconds
    assert result.iterations == 5 and result.work_items_per_second > 0


def test_profiler_reports_real_operator_self_time() -> None:
    left = torch.randn(16, 24)
    right = torch.randn(24, 8)
    events = profile_callable(lambda: left @ right, warmup=1, active_steps=2)
    assert events
    assert any("mm" in name for name in events)
    assert all(value >= 0 for value in events.values())


def test_uneven_shards_round_trip_and_preserve_gradients() -> None:
    tensor = torch.arange(35.0).reshape(5, 7).requires_grad_()
    shards = contiguous_shards(tensor, 3, dim=1)
    assert [shard.shape[1] for shard in shards] == [3, 2, 2]
    reconstructed = gather_shards(shards, dim=1)
    torch.testing.assert_close(reconstructed, tensor)
    reconstructed.sum().backward()
    torch.testing.assert_close(tensor.grad, torch.ones_like(tensor))
    with pytest.raises(ValueError):
        contiguous_shards(tensor, 0)


def test_buckets_and_optimizer_shards_are_deterministic() -> None:
    tensors = [
        ("a", torch.zeros(4, dtype=torch.float32)),
        ("b", torch.zeros(8, dtype=torch.float32)),
        ("huge", torch.zeros(40, dtype=torch.float32)),
        ("c", torch.zeros(4, dtype=torch.float32)),
    ]
    assert bucket_tensors(tensors, max_bytes=64) == (("a", "b"), ("huge",), ("c",))
    assignments = assign_optimizer_states({"large": 10, "medium": 7, "small": 3}, world_size=2)
    assert assignments == (("large",), ("medium", "small"))
