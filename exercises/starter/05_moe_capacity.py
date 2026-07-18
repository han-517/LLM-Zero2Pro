"""Week 35–36：计算 MoE 容量并显式处理溢出。"""

from __future__ import annotations

import math


def expert_capacity(tokens: int, experts: int, top_k: int, factor: float) -> int:
    # TODO: 使用向上取整。
    raise NotImplementedError


def accepted_and_dropped(loads: list[int], capacity: int) -> tuple[list[int], int]:
    # TODO: 每个专家最多接受 capacity 个 assignment，并统计总丢弃数。
    raise NotImplementedError


if __name__ == "__main__":
    capacity = expert_capacity(tokens=100, experts=8, top_k=2, factor=1.25)
    assert capacity == math.ceil(1.25 * 100 * 2 / 8) == 32
    accepted, dropped = accepted_and_dropped([50, 30, 20, 20, 20, 20, 20, 20], capacity)
    assert accepted == [32, 30, 20, 20, 20, 20, 20, 20]
    assert dropped == 18
    print("通过：容量和 dropping 显式可见")
