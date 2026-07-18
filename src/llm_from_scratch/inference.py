from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import torch
from torch import Tensor


def symmetric_quantize(x: Tensor, bits: int = 8) -> tuple[Tensor, Tensor]:
    """教学版 per-tensor 对称伪量化，返回整数张量和 scale。"""

    if not 2 <= bits <= 8:
        raise ValueError("bits 必须在 2..8")
    if x.numel() == 0 or not torch.isfinite(x).all():
        raise ValueError("x 必须是非空有限张量")
    qmax = 2 ** (bits - 1) - 1
    max_value = x.abs().max()
    scale = torch.where(max_value > 0, max_value / qmax, torch.ones_like(max_value))
    quantized = torch.round(x / scale).clamp(-qmax, qmax).to(torch.int8)
    return quantized, scale


def symmetric_dequantize(quantized: Tensor, scale: Tensor) -> Tensor:
    if quantized.numel() == 0 or scale.numel() == 0:
        raise ValueError("quantized 和 scale 不能为空")
    if not torch.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("scale 必须为有限正数")
    return quantized.float() * scale


@dataclass
class PageTable:
    """不存真实 K/V 的分页缓存语义模拟器。

    它实现逻辑页到物理页映射、引用计数、前缀共享和 partial-page
    copy-on-write。它仍然是内存管理 toy，而不是注意力 kernel。
    """

    page_size: int
    free_pages: list[int]
    sequence_pages: dict[str, list[int]] = field(default_factory=dict)
    sequence_lengths: dict[str, int] = field(default_factory=dict)
    page_refcounts: dict[int, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.page_size < 1:
            raise ValueError("page_size 必须 >= 1")
        if len(self.free_pages) != len(set(self.free_pages)):
            raise ValueError("free_pages 不能包含重复页")
        used_pages = [page for pages in self.sequence_pages.values() for page in pages]
        if set(used_pages) & set(self.free_pages):
            raise ValueError("物理页不能同时处于已分配和空闲状态")
        for sequence_id, pages in self.sequence_pages.items():
            length = self.sequence_lengths.get(sequence_id)
            if length is None or length < 0 or len(pages) != self._pages_for(length):
                raise ValueError("初始 sequence_pages 与 sequence_lengths 不一致")
            for page in pages:
                self.page_refcounts[page] = self.page_refcounts.get(page, 0) + 1
        self.free_pages.sort()

    def _pages_for(self, token_count: int) -> int:
        return (token_count + self.page_size - 1) // self.page_size

    def _allocate(self) -> int:
        page = self.free_pages.pop(0)
        self.page_refcounts[page] = 1
        return page

    def _decrement(self, page: int) -> None:
        remaining = self.page_refcounts[page] - 1
        if remaining == 0:
            del self.page_refcounts[page]
            self.free_pages.append(page)
            self.free_pages.sort()
        else:
            self.page_refcounts[page] = remaining

    def append_tokens(self, sequence_id: str, count: int) -> list[int]:
        if count < 0:
            raise ValueError("count 不能为负")
        old_length = self.sequence_lengths.get(sequence_id, 0)
        pages = self.sequence_pages.setdefault(sequence_id, [])
        new_length = old_length + count
        missing = self._pages_for(new_length) - len(pages)
        needs_cow = bool(
            count and pages and old_length % self.page_size and self.page_refcounts[pages[-1]] > 1
        )
        if missing + int(needs_cow) > len(self.free_pages):
            raise MemoryError("没有足够的 KV pages")
        if needs_cow:
            shared_page = pages[-1]
            pages[-1] = self._allocate()
            self._decrement(shared_page)
        for _ in range(missing):
            pages.append(self._allocate())
        self.sequence_lengths[sequence_id] = new_length
        return list(pages)

    def share_prefix(
        self, source_sequence_id: str, new_sequence_id: str, prefix_tokens: int | None = None
    ) -> list[int]:
        """创建共享前缀；后续写入共享 partial page 时自动 copy-on-write。"""

        if source_sequence_id not in self.sequence_pages:
            raise KeyError(f"未知 source sequence: {source_sequence_id}")
        if new_sequence_id in self.sequence_pages:
            raise ValueError("new_sequence_id 已存在")
        source_length = self.sequence_lengths[source_sequence_id]
        prefix_tokens = source_length if prefix_tokens is None else prefix_tokens
        if not 0 <= prefix_tokens <= source_length:
            raise ValueError("prefix_tokens 必须位于 source 序列范围内")
        pages = list(self.sequence_pages[source_sequence_id][: self._pages_for(prefix_tokens)])
        for page in pages:
            self.page_refcounts[page] += 1
        self.sequence_pages[new_sequence_id] = pages
        self.sequence_lengths[new_sequence_id] = prefix_tokens
        return list(pages)

    def logical_to_physical(self, sequence_id: str, token_index: int) -> tuple[int, int]:
        if sequence_id not in self.sequence_pages:
            raise KeyError(f"未知 sequence: {sequence_id}")
        if not 0 <= token_index < self.sequence_lengths[sequence_id]:
            raise IndexError("token_index 超出序列范围")
        logical_page, offset = divmod(token_index, self.page_size)
        return self.sequence_pages[sequence_id][logical_page], offset

    def release(self, sequence_id: str) -> None:
        pages = self.sequence_pages.pop(sequence_id, [])
        self.sequence_lengths.pop(sequence_id, None)
        for page in pages:
            self._decrement(page)

    @property
    def internal_fragmentation_tokens(self) -> int:
        """按唯一物理页计算未使用槽位；共享页只计算一次。"""

        used_per_page: dict[int, int] = {}
        for sequence_id, pages in self.sequence_pages.items():
            length = self.sequence_lengths[sequence_id]
            for logical_page, physical_page in enumerate(pages):
                used = min(self.page_size, max(0, length - logical_page * self.page_size))
                used_per_page[physical_page] = max(used_per_page.get(physical_page, 0), used)
        return sum(self.page_size - used for used in used_per_page.values())

    @property
    def allocated_pages(self) -> int:
        return len(self.page_refcounts)

    @property
    def physical_utilization(self) -> float:
        if not self.page_refcounts:
            return 1.0
        capacity = self.allocated_pages * self.page_size
        return (capacity - self.internal_fragmentation_tokens) / capacity


@dataclass(frozen=True)
class SpeculativeStats:
    proposed: int
    accepted: int
    target_calls: int


def greedy_speculative_decode(
    draft_next: Callable[[list[int]], int],
    target_verify: Callable[[list[int], list[int]], list[int]],
    prefix: list[int],
    max_new_tokens: int,
    draft_steps: int = 4,
) -> tuple[list[int], SpeculativeStats]:
    """教学用块式贪心推测解码，不保证随机采样分布。"""

    if max_new_tokens < 0 or draft_steps < 1:
        raise ValueError("max_new_tokens 必须非负且 draft_steps >= 1")
    output = list(prefix)
    generated = proposed = accepted = target_calls = 0
    while generated < max_new_tokens:
        draft_context = list(output)
        candidates: list[int] = []
        for _ in range(min(draft_steps, max_new_tokens - generated)):
            candidate = int(draft_next(draft_context))
            candidates.append(candidate)
            draft_context.append(candidate)
        proposed += len(candidates)
        target_tokens = [int(token) for token in target_verify(list(output), list(candidates))]
        target_calls += 1
        if len(target_tokens) != len(candidates):
            raise ValueError("target_verify 必须为每个 candidate 返回一个 token")
        for candidate, target_token in zip(candidates, target_tokens, strict=True):
            if candidate == target_token:
                output.append(candidate)
                accepted += 1
            else:
                output.append(target_token)
            generated += 1
            if candidate != target_token or generated >= max_new_tokens:
                break
    return output, SpeculativeStats(proposed, accepted, target_calls)


def _probabilities(values: Tensor) -> Tensor:
    if values.ndim != 1 or values.numel() < 2:
        raise ValueError("分布必须是一维且至少包含两个 token")
    values = values.float()
    if not torch.isfinite(values).all() or (values < 0).any() or values.sum() <= 0:
        raise ValueError("分布必须是非负、有限且总和大于零")
    return values / values.sum()


def _sample(probabilities: Tensor, generator: torch.Generator | None) -> int:
    return int(torch.multinomial(probabilities, 1, generator=generator).item())


def stochastic_speculative_decode(
    draft_distribution: Callable[[list[int]], Tensor],
    target_distributions: Callable[[list[int], list[int]], Sequence[Tensor]],
    prefix: list[int],
    max_new_tokens: int,
    draft_steps: int = 4,
    *,
    generator: torch.Generator | None = None,
) -> tuple[list[int], SpeculativeStats]:
    """分布保持的标准 stochastic speculative decoding 教学参考。

    ``target_distributions(prefix, candidates)`` 用一次目标调用返回 ``K+1`` 个
    分布：K 个候选位置和“全部接受”后的 bonus 位置。拒绝时从归一化的
    ``(p-q)_+`` 校正分布采样。
    """

    if max_new_tokens < 0 or draft_steps < 1:
        raise ValueError("max_new_tokens 必须非负且 draft_steps >= 1")
    output = list(prefix)
    generated = proposed = accepted = target_calls = 0
    while generated < max_new_tokens:
        steps = min(draft_steps, max_new_tokens - generated)
        draft_context = list(output)
        candidates: list[int] = []
        draft_probabilities: list[Tensor] = []
        for _ in range(steps):
            probabilities = _probabilities(draft_distribution(draft_context))
            candidate = _sample(probabilities, generator)
            candidates.append(candidate)
            draft_probabilities.append(probabilities)
            draft_context.append(candidate)
        proposed += len(candidates)
        target_values = list(target_distributions(list(output), list(candidates)))
        target_calls += 1
        if len(target_values) != len(candidates) + 1:
            raise ValueError("target_distributions 必须返回 K+1 个分布（含 bonus）")
        target_probabilities = [_probabilities(values) for values in target_values]
        if any(values.numel() != draft_probabilities[0].numel() for values in target_probabilities):
            raise ValueError("draft 与 target 的词表大小必须相同")

        rejected = False
        for index, candidate in enumerate(candidates):
            p = target_probabilities[index]
            q = draft_probabilities[index]
            acceptance = min(1.0, float((p[candidate] / q[candidate]).item()))
            uniform = float(torch.rand((), generator=generator).item())
            if uniform <= acceptance:
                output.append(candidate)
                accepted += 1
            else:
                residual = (p - q).clamp_min(0)
                # 数值舍入可能让 residual 为零；此时直接回退到 target 分布。
                corrected = p if residual.sum() <= 0 else residual / residual.sum()
                output.append(_sample(corrected, generator))
                rejected = True
            generated += 1
            if rejected or generated >= max_new_tokens:
                break
        if not rejected and generated < max_new_tokens:
            output.append(_sample(target_probabilities[-1], generator))
            generated += 1
    return output, SpeculativeStats(proposed, accepted, target_calls)


@dataclass(frozen=True)
class RequestTrace:
    """一个服务请求的 arrival-to-completion 观测。时间单位保持一致即可。"""

    arrival_time: float
    first_token_time: float
    completion_time: float
    output_tokens: int
    prompt_tokens: int = 0

    def __post_init__(self) -> None:
        if not self.arrival_time <= self.first_token_time <= self.completion_time:
            raise ValueError("时间必须满足 arrival <= first token <= completion")
        if self.output_tokens < 1 or self.prompt_tokens < 0:
            raise ValueError("output_tokens 必须 >=1，prompt_tokens 必须非负")

    @property
    def ttft(self) -> float:
        return self.first_token_time - self.arrival_time

    @property
    def e2e_latency(self) -> float:
        return self.completion_time - self.arrival_time

    @property
    def tpot(self) -> float | None:
        if self.output_tokens == 1:
            return None
        return (self.completion_time - self.first_token_time) / (self.output_tokens - 1)


@dataclass(frozen=True)
class LatencySummary:
    mean: float
    p50: float
    p95: float
    p99: float


@dataclass(frozen=True)
class ServingMetrics:
    ttft: LatencySummary
    e2e_latency: LatencySummary
    tpot: LatencySummary | None
    request_throughput: float
    output_token_throughput: float
    total_token_throughput: float
    goodput: float | None


def _quantile(sorted_values: list[float], probability: float) -> float:
    position = (len(sorted_values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def _latency_summary(values: Sequence[float]) -> LatencySummary:
    ordered = sorted(float(value) for value in values)
    return LatencySummary(
        mean=sum(ordered) / len(ordered),
        p50=_quantile(ordered, 0.50),
        p95=_quantile(ordered, 0.95),
        p99=_quantile(ordered, 0.99),
    )


def summarize_serving(
    traces: Sequence[RequestTrace],
    *,
    ttft_slo: float | None = None,
    tpot_slo: float | None = None,
) -> ServingMetrics:
    """汇总排队在内的 TTFT、TPOT、吞吐与可选 SLO goodput。"""

    if not traces:
        raise ValueError("traces 不能为空")
    if ttft_slo is not None and ttft_slo < 0:
        raise ValueError("ttft_slo 不能为负")
    if tpot_slo is not None and tpot_slo < 0:
        raise ValueError("tpot_slo 不能为负")
    duration = max(trace.completion_time for trace in traces) - min(
        trace.arrival_time for trace in traces
    )
    if duration <= 0:
        raise ValueError("观测窗口 duration 必须为正")
    tpot_values = [trace.tpot for trace in traces if trace.tpot is not None]
    output_tokens = sum(trace.output_tokens for trace in traces)
    total_tokens = output_tokens + sum(trace.prompt_tokens for trace in traces)
    goodput = None
    if ttft_slo is not None or tpot_slo is not None:
        meeting = 0
        for trace in traces:
            ttft_ok = ttft_slo is None or trace.ttft <= ttft_slo
            tpot_ok = tpot_slo is None or trace.tpot is None or trace.tpot <= tpot_slo
            meeting += int(ttft_ok and tpot_ok)
        goodput = meeting / duration
    return ServingMetrics(
        ttft=_latency_summary([trace.ttft for trace in traces]),
        e2e_latency=_latency_summary([trace.e2e_latency for trace in traces]),
        tpot=_latency_summary(tpot_values) if tpot_values else None,
        request_throughput=len(traces) / duration,
        output_token_throughput=output_tokens / duration,
        total_token_throughput=total_tokens / duration,
        goodput=goodput,
    )
