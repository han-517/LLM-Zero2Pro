"""Week 35–36：实现 Top-k MoE 路由权重与负载均衡损失。"""

from __future__ import annotations

from torch import Tensor


def topk_route(router_logits: Tensor, top_k: int) -> tuple[Tensor, Tensor, Tensor]:
    """输入 [tokens,experts]，返回完整概率、Top-k 下标和重新归一化的 Top-k 权重。"""
    # TODO:
    # 1. 对专家维做 softmax（建议用 float32）。
    # 2. 选择每个 token 的 Top-k。
    # 3. 让被选中的 k 个权重之和重新变为 1。
    raise NotImplementedError


def switch_balance_loss(router_probabilities: Tensor, top_indices: Tensor) -> Tensor:
    """实现 E * sum(mean_probability * selected_fraction)。"""
    # TODO: selected_fraction 的分母是 tokens * top_k，并需统计所有专家。
    raise NotImplementedError
