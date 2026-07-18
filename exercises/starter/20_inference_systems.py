"""Week 45–47：补全对称量化和推测采样接受概率。"""

from torch import Tensor


def symmetric_quantize(x: Tensor, bits: int = 8) -> tuple[Tensor, Tensor]:
    """返回整数 q 与标量 scale；零张量必须安全。"""
    # TODO: 饱和到有符号范围，并拒绝不支持的 bit width。
    raise NotImplementedError


def acceptance_probability(draft_probability: Tensor, target_probability: Tensor) -> Tensor:
    """返回 min(1, p_target/p_draft)，并验证概率契约。"""
    # TODO: draft=0 时按数学事件是否可能显式处理。
    raise NotImplementedError
