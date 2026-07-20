"""Week 12–13：实现单头因果注意力。"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def causal_attention(query: Tensor, key: Tensor, value: Tensor) -> Tensor:
    """输入均为 [T,D]；输出 [T,Dv]。"""
    # TODO:
    # 1. 计算缩放点积分数。
    # 2. 将未来位置设为负无穷。
    # 3. Softmax 后乘 value。
    raise NotImplementedError


if __name__ == "__main__":
    torch.manual_seed(7)
    q = torch.randn(5, 4)
    k = torch.randn(5, 4)
    v = torch.randn(5, 3)
    actual = causal_attention(q, k, v)
    expected = F.scaled_dot_product_attention(
        q.view(1, 1, 5, 4),
        k.view(1, 1, 5, 4),
        v.view(1, 1, 5, 3),
        is_causal=True,
    )[0, 0]
    torch.testing.assert_close(actual, expected)

    changed = v.clone()
    changed[-1] += 1000
    after = causal_attention(q, k, changed)
    torch.testing.assert_close(actual[:-1], after[:-1])
    print("通过：数值 oracle 与未来信息扰动测试")
