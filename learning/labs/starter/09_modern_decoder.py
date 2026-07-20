"""Week 21–22：实现现代 Decoder 常用的 RMSNorm 与 SwiGLU。"""

from __future__ import annotations

from torch import Tensor


def rms_norm(x: Tensor, weight: Tensor, eps: float = 1e-6) -> Tensor:
    """沿最后一维计算 RMSNorm；归约使用 float32，再还原 x.dtype。"""
    # TODO: RMSNorm 不减均值；weight 的形状应为 [D]。
    raise NotImplementedError


def swiglu(
    x: Tensor,
    gate_weight: Tensor,
    up_weight: Tensor,
    down_weight: Tensor,
) -> Tensor:
    """权重采用 F.linear 的 [out,in] 约定，实现 down(SiLU(gate(x))*up(x))。"""
    # TODO: 不要把 SiLU 误用在 up 分支或乘法之后。
    raise NotImplementedError
