"""Week 22–23：补全确定性去重与不跨样本的 packing。"""

from collections.abc import Sequence


def exact_deduplicate(documents: Sequence[str]) -> tuple[list[str], list[int]]:
    """保留首次出现，返回文档与原始下标。"""
    # TODO: 保持顺序，显式处理空字符串。
    raise NotImplementedError


def pack_sequences(
    sequences: Sequence[Sequence[int]], block_size: int, eos_id: int
) -> list[list[int]]:
    """把样本装入定长块；每个样本末尾插入 EOS，不静默截断 token。"""
    # TODO: 定义长样本和最后不足一块的策略。
    raise NotImplementedError
