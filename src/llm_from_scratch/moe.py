from __future__ import annotations

import math
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from llm_from_scratch.transformer import SwiGLU

RoutingMode = Literal["softmax", "sigmoid"]


class Expert(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int) -> None:
        super().__init__()
        self.ffn = SwiGLU(d_model, hidden_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.ffn(x)


class TopKMoE(nn.Module):
    """可观察容量与 dropping 的教学版 token-choice MoE。

    ``routing_mode`` 决定 affinity score，``normalize_topk`` 决定被选权重是否归一。
    这两个选项用于比较公开架构，并不声称覆盖所有生产实现。特别地，当
    ``top_k=1`` 且 ``normalize_topk=True`` 时被选权重恒为 1，主任务不会通过混合
    权重给 router 梯度；router 仍可由 balance/z-loss 等辅助目标训练。
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
        num_experts: int = 4,
        top_k: int = 2,
        capacity_factor: float = 1.25,
        shared_expert: bool = False,
        *,
        routing_mode: RoutingMode = "softmax",
        normalize_topk: bool = True,
    ) -> None:
        super().__init__()
        if d_model < 1 or hidden_dim < 1 or num_experts < 1:
            raise ValueError("d_model、hidden_dim 和 num_experts 必须为正")
        if not 1 <= top_k <= num_experts:
            raise ValueError("top_k 必须在 1..num_experts")
        if capacity_factor <= 0 or not math.isfinite(capacity_factor):
            raise ValueError("capacity_factor 必须是有限正数")
        if routing_mode not in ("softmax", "sigmoid"):
            raise ValueError("routing_mode 必须是 'softmax' 或 'sigmoid'")
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.routing_mode = routing_mode
        self.normalize_topk = normalize_topk
        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList([Expert(d_model, hidden_dim) for _ in range(num_experts)])
        self.shared = Expert(d_model, hidden_dim) if shared_expert else None

    def _router_logits(self, flat: Tensor) -> Tensor:
        # 同时把输入和权重视图转为 FP32。即使外部对整个 MoE 调用了 .to(bfloat16)，
        # router matmul 仍在 FP32 中进行，梯度会正确回传到原参数 dtype。
        return F.linear(flat.float(), self.router.weight.float())

    def forward(self, x: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        if x.ndim < 2 or x.shape[-1] != self.d_model:
            raise ValueError("x 最后一维必须等于 d_model")
        original_shape = x.shape
        flat = x.reshape(-1, self.d_model)
        token_count = flat.shape[0]
        if token_count == 0:
            raise ValueError("MoE 至少需要一个 token")

        router_logits = self._router_logits(flat)
        router_probabilities = torch.softmax(router_logits, dim=-1)
        router_scores = (
            router_probabilities if self.routing_mode == "softmax" else torch.sigmoid(router_logits)
        )
        top_weights, top_indices = torch.topk(router_scores, self.top_k, dim=-1)
        if self.normalize_topk:
            top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        output = torch.zeros_like(flat)
        accepted_load = torch.zeros(self.num_experts, device=x.device, dtype=torch.long)
        selected_load = torch.bincount(top_indices.flatten(), minlength=self.num_experts)
        dropped = torch.zeros(token_count, self.top_k, device=x.device, dtype=torch.bool)
        capacity = math.ceil(self.capacity_factor * token_count * self.top_k / self.num_experts)

        for expert_index, expert in enumerate(self.experts):
            locations = (top_indices == expert_index).nonzero(as_tuple=False)
            if locations.numel() == 0:
                continue
            weights = top_weights[locations[:, 0], locations[:, 1]]
            if locations.shape[0] > capacity:
                order = torch.argsort(weights, descending=True, stable=True)
                rejected = locations[order[capacity:]]
                dropped[rejected[:, 0], rejected[:, 1]] = True
                locations = locations[order[:capacity]]
                weights = top_weights[locations[:, 0], locations[:, 1]]
            token_indices = locations[:, 0]
            expert_result = expert(flat[token_indices]).to(flat.dtype)
            output.index_add_(
                0, token_indices, expert_result * weights.to(flat.dtype).unsqueeze(-1)
            )
            accepted_load[expert_index] = locations.shape[0]

        if self.shared is not None:
            output = output + self.shared(flat)

        mean_probability = router_probabilities.mean(dim=0)
        selected_fraction = selected_load.float() / (token_count * self.top_k)
        balance_loss = self.num_experts * torch.sum(mean_probability * selected_fraction)
        z_loss = torch.mean(torch.logsumexp(router_logits, dim=-1).pow(2))
        accepted_per_token = (~dropped).sum(dim=-1)
        dropped_assignments = dropped.sum()
        accepted_assignments = accepted_load.sum()
        auxiliary = {
            "balance_loss": balance_loss,
            "z_loss": z_loss,
            "router_logits": router_logits,
            "router_probabilities": router_probabilities,
            "router_scores": router_scores,
            "top_indices": top_indices,
            "top_weights": top_weights,
            "selected_load": selected_load,
            "accepted_load": accepted_load,
            "accepted_per_token": accepted_per_token,
            "accepted_assignments": accepted_assignments,
            "dropped_assignments": dropped_assignments,
            "dropped": dropped,
            "capacity": torch.tensor(capacity, device=x.device),
            "normalized_top1": torch.tensor(
                self.top_k == 1 and self.normalize_topk, device=x.device
            ),
        }
        return output.reshape(original_shape), auxiliary


@torch.no_grad()
def upcycle_expert(source: Expert, target: TopKMoE, noise_std: float = 0.0) -> None:
    """把 Dense Expert 复制到 MoE，并在单 shared expert 下保持初始输出尺度。"""

    if noise_std < 0 or not math.isfinite(noise_std):
        raise ValueError("noise_std 必须是有限非负数")
    output_scale = 0.5 if target.shared is not None else 1.0

    for expert in target.experts:
        expert.load_state_dict(source.state_dict())
        expert.ffn.down.weight.mul_(output_scale)
        if expert.ffn.down.bias is not None:
            expert.ffn.down.bias.mul_(output_scale)
        if noise_std:
            for parameter in expert.parameters():
                parameter.add_(torch.randn_like(parameter) * noise_std)
    if target.shared is not None:
        target.shared.load_state_dict(source.state_dict())
        target.shared.ffn.down.weight.mul_(output_scale)
        if target.shared.ffn.down.bias is not None:
            target.shared.ffn.down.bias.mul_(output_scale)


def moe_parameter_accounting(
    d_model: int,
    hidden_dim: int,
    num_experts: int,
    top_k: int,
    *,
    shared_experts: int = 0,
) -> dict[str, int]:
    """统计无 bias SwiGLU MoE 的总参数和每 token 活跃参数。"""

    if min(d_model, hidden_dim, num_experts) < 1:
        raise ValueError("d_model、hidden_dim 和 num_experts 必须为正")
    if not 1 <= top_k <= num_experts or shared_experts < 0:
        raise ValueError("top_k 必须合法，shared_experts 不能为负")
    expert_parameters = 3 * d_model * hidden_dim
    router_parameters = d_model * num_experts
    return {
        "router_parameters": router_parameters,
        "parameters_per_expert": expert_parameters,
        "total_parameters": router_parameters + expert_parameters * (num_experts + shared_experts),
        "active_parameters_per_token": router_parameters
        + expert_parameters * (top_k + shared_experts),
    }


def expert_parallel_communication_ledger(
    token_count: int,
    d_model: int,
    top_k: int,
    *,
    element_bytes: int = 2,
    world_size: int = 1,
) -> dict[str, int]:
    """估算均匀专家放置时一次 dispatch+combine 的激活通信上界。

    账本忽略索引、padding、协议开销与通信重叠，因此不能代替真实性能测量。
    ``remote_bytes`` 使用均匀路由下远端比例 ``(world_size-1)/world_size``。
    """

    if min(token_count, d_model, top_k, element_bytes) < 0 or world_size < 1:
        raise ValueError("计数必须非负且 world_size 必须 >= 1")
    if d_model == 0 or top_k == 0 or element_bytes == 0:
        raise ValueError("d_model、top_k 和 element_bytes 必须为正")
    assignments = token_count * top_k
    one_way_bytes = assignments * d_model * element_bytes
    total_bytes = 2 * one_way_bytes
    remote_bytes = math.ceil(total_bytes * (world_size - 1) / world_size)
    return {
        "assignments": assignments,
        "dispatch_bytes": one_way_bytes,
        "combine_bytes": one_way_bytes,
        "total_bytes": total_bytes,
        "remote_bytes_uniform_assumption": remote_bytes,
    }
