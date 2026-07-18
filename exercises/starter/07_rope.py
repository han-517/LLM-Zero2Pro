"""Week 18：实现 Rotary Position Embedding。"""

from __future__ import annotations

from torch import Tensor


def apply_rope(x: Tensor, positions: Tensor | None = None, base: float = 10_000.0) -> Tensor:
    """对 [B,H,T,D] 的偶/奇维成对旋转，D 必须为偶数。"""
    # TODO:
    # 1. 生成 half 个 inverse frequencies 和 [T, half] 的旋转角。
    # 2. 分离 x[..., 0::2] 与 x[..., 1::2]。
    # 3. 应用二维旋转并交错写回；保持输入 dtype/device。
    raise NotImplementedError
