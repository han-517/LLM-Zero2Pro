"""Week 23–24：实现不物化重复 K/V 的 Grouped-Query Attention。"""

from __future__ import annotations

from torch import Tensor


def grouped_query_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    causal: bool = True,
) -> Tensor:
    """q=[B,Hq,Tq,D]，k/v=[B,Hkv,Tk,D/Dv]，返回 [B,Hq,Tq,Dv]。"""
    # TODO:
    # 1. 验证 Hq 能被 Hkv 整除，令 groups = Hq // Hkv。
    # 2. 把 query reshape 为 [B,Hkv,groups,Tq,D]。
    # 3. 用 einsum 计算分数；因果 mask 要兼容 Tq < Tk 的 cache decode。
    # 4. Softmax 后与 value 聚合，再恢复 Hq 维。
    # 限制：不要用 repeat_interleave 物化 K/V。
    raise NotImplementedError
