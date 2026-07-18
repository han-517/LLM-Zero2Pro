from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import torch
from torch import Tensor


def symmetric_quantize(x: Tensor, bits: int = 8) -> tuple[Tensor, Tensor]:
    """教学版 per-tensor 对称伪量化，返回整数张量和 scale。"""

    if not 2 <= bits <= 8:
        raise ValueError("bits 必须在 2..8")
    qmax = 2 ** (bits - 1) - 1
    max_value = x.abs().max()
    scale = torch.where(max_value > 0, max_value / qmax, torch.ones_like(max_value))
    quantized = torch.round(x / scale).clamp(-qmax, qmax).to(torch.int8)
    return quantized, scale


def symmetric_dequantize(quantized: Tensor, scale: Tensor) -> Tensor:
    return quantized.float() * scale


@dataclass
class PageTable:
    """只模拟 KV 物理页分配，不存真实 K/V。"""

    page_size: int
    free_pages: list[int]
    sequence_pages: dict[str, list[int]] = field(default_factory=dict)
    sequence_lengths: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.page_size < 1:
            raise ValueError("page_size 必须 >= 1")

    def append_tokens(self, sequence_id: str, count: int) -> list[int]:
        if count < 0:
            raise ValueError("count 不能为负")
        old_length = self.sequence_lengths.get(sequence_id, 0)
        new_length = old_length + count
        required_pages = (new_length + self.page_size - 1) // self.page_size
        pages = self.sequence_pages.setdefault(sequence_id, [])
        missing = required_pages - len(pages)
        if missing > len(self.free_pages):
            raise MemoryError("没有足够的 KV pages")
        for _ in range(missing):
            pages.append(self.free_pages.pop(0))
        self.sequence_lengths[sequence_id] = new_length
        return list(pages)

    def release(self, sequence_id: str) -> None:
        released = self.sequence_pages.pop(sequence_id, [])
        self.sequence_lengths.pop(sequence_id, None)
        self.free_pages.extend(released)
        self.free_pages.sort()


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
    """教学用块式贪心推测解码。

    target_verify(prefix, candidates) 必须用一次目标模型调用返回 candidates 各位置的
    贪心 token。真正的随机采样版还需要接受概率与校正分布。
    """

    if max_new_tokens < 0 or draft_steps < 1:
        raise ValueError("max_new_tokens 必须非负且 draft_steps >= 1")
    output = list(prefix)
    generated = 0
    proposed = accepted = target_calls = 0
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

