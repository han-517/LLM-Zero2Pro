"""Project 03: profiling, communication and distributed-step starter."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from torch import Tensor, nn


@dataclass(frozen=True)
class BenchmarkResult:
    median_seconds: float
    p20_seconds: float
    p80_seconds: float
    iterations: int
    work_items_per_second: float


def benchmark_callable(
    function: Callable[[], object],
    *,
    warmup: int,
    iterations: int,
    work_items: int,
    synchronize: Callable[[], None] | None = None,
) -> BenchmarkResult:
    """Benchmark with untimed warmup, device synchronization and robust quantiles."""

    # TODO: validate counts, synchronize around each timed call, and use perf_counter/median.
    raise NotImplementedError


def profile_callable(
    function: Callable[[], object],
    *,
    warmup: int = 1,
    active_steps: int = 3,
) -> dict[str, float]:
    """Profile CPU/CUDA operators and return self time in microseconds by operator name."""

    # TODO: use torch.profiler schedule/step and aggregate key_averages without hiding warmup.
    raise NotImplementedError


def contiguous_shards(tensor: Tensor, world_size: int, *, dim: int = 0) -> tuple[Tensor, ...]:
    """Split a tensor into balanced contiguous shards, including uneven dimensions."""

    # TODO: validate world_size/dim and distribute the remainder over the earliest ranks.
    raise NotImplementedError


def gather_shards(shards: Iterable[Tensor], *, dim: int = 0) -> Tensor:
    """Reconstruct a tensor from compatible contiguous shards without detaching gradients."""

    # TODO: reject empty or shape-incompatible shard lists, then concatenate.
    raise NotImplementedError


def bucket_tensors(
    named_tensors: Iterable[tuple[str, Tensor]],
    *,
    max_bytes: int,
) -> tuple[tuple[str, ...], ...]:
    """Pack tensors in input order into deterministic communication buckets."""

    # TODO: count numel*element_size; oversized tensors get a bucket of their own.
    raise NotImplementedError


def assign_optimizer_states(
    parameter_numels: dict[str, int],
    *,
    world_size: int,
) -> tuple[tuple[str, ...], ...]:
    """Greedily assign whole optimizer states to ranks by descending parameter size."""

    # TODO: deterministic longest-processing-time placement with lexical tie breaks.
    raise NotImplementedError


def ddp_train_step(
    model: nn.Module,
    inputs: Tensor,
    targets: Tensor,
    optimizer,
) -> Tensor:
    """Run one next-token step on a local or DistributedDataParallel-wrapped model."""

    # TODO: zero grads, forward, shifted CE, backward, finite check, optimizer step; return detached loss.
    raise NotImplementedError
