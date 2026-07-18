"""Week 4：实现数值稳定的 Softmax。"""

from __future__ import annotations

import math


def stable_softmax(logits: list[float]) -> list[float]:
    """返回与 logits 等长的概率；不得在大 logits 上溢出。"""
    # TODO: 先减最大值，再计算 exp 与归一化。
    raise NotImplementedError


if __name__ == "__main__":
    probabilities = stable_softmax([1000.0, 1001.0, 1002.0])
    assert len(probabilities) == 3
    assert all(0.0 < value < 1.0 for value in probabilities)
    assert math.isclose(sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-12)

    shifted = stable_softmax([-9000.0, -8999.0, -8998.0])
    assert all(
        math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12)
        for left, right in zip(probabilities, shifted, strict=True)
    )
    print("通过：稳定 Softmax 与平移不变性")
