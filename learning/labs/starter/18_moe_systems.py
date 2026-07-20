"""Week 37–39：补全容量计算与确定性 token dispatch 计划。"""

from torch import Tensor


def expert_capacity(tokens: int, experts: int, top_k: int, capacity_factor: float) -> int:
    """返回 ceil(tokens*top_k/experts*capacity_factor)。"""
    # TODO: 拒绝非正参数。
    raise NotImplementedError


def dispatch_counts(expert_indices: Tensor, num_experts: int) -> Tensor:
    """统计 [tokens,top_k] 路由到每个专家的 token-slot 数。"""
    # TODO: 验证索引范围并返回整数计数。
    raise NotImplementedError
