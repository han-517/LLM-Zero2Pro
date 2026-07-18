"""Week 9–10：实现 Byte BPE 最核心的计数、选择和合并步骤。"""

from __future__ import annotations

from collections import Counter

Pair = tuple[int, int]


def count_adjacent_pairs(sequence: list[int]) -> Counter[Pair]:
    """统计所有相邻 token pair；重叠出现也要计数。"""
    # TODO:
    # 1. 相邻 pair 来自 sequence[i] 与 sequence[i + 1]。
    # 2. 长度小于 2 时返回空 Counter。
    raise NotImplementedError


def choose_pair(counts: Counter[Pair]) -> Pair:
    """选择次数最多的 pair；并列时选择整数对字典序最小者。"""
    # TODO: 确定性 tie-break 是可复现 tokenizer 的一部分。
    raise NotImplementedError


def merge_pair(sequence: list[int], pair: Pair, new_id: int) -> list[int]:
    """从左到右非重叠合并 pair，不修改输入列表。"""
    # TODO: 命中后 index 前进 2，否则前进 1。
    raise NotImplementedError
