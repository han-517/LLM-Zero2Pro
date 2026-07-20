"""Week 14–15：补全因果掩码和 Pre-Norm 残差骨架。"""

from torch import Tensor, nn


def causal_mask(query_length: int, key_length: int, offset: int = 0) -> Tensor:
    """返回 [Tq,Tk] 布尔可见性；支持 cache 解码的 query offset。"""
    # TODO: 第 i 个 query 只能看见 key <= offset+i。
    raise NotImplementedError


def prenorm_residual(x: Tensor, norm: nn.Module, sublayer: nn.Module) -> Tensor:
    """实现 x + sublayer(norm(x))。"""
    # TODO: 核心残差顺序留给学员填写。
    raise NotImplementedError
