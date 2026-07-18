from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from llm_from_scratch.transformer import SwiGLU


class Expert(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int) -> None:
        super().__init__()
        self.ffn = SwiGLU(d_model, hidden_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.ffn(x)


class TopKMoE(nn.Module):
    """可观察容量、dropping 和辅助损失的教学版 token-choice MoE。"""

    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
        num_experts: int = 4,
        top_k: int = 2,
        capacity_factor: float = 1.25,
        shared_expert: bool = False,
    ) -> None:
        super().__init__()
        if not 1 <= top_k <= num_experts:
            raise ValueError("top_k 必须在 1..num_experts")
        if capacity_factor <= 0:
            raise ValueError("capacity_factor 必须为正")
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList([Expert(d_model, hidden_dim) for _ in range(num_experts)])
        self.shared = Expert(d_model, hidden_dim) if shared_expert else None

    def forward(self, x: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        if x.ndim < 2 or x.shape[-1] != self.d_model:
            raise ValueError("x 最后一维必须等于 d_model")
        original_shape = x.shape
        flat = x.reshape(-1, self.d_model)
        token_count = flat.shape[0]
        router_logits = self.router(flat.float())
        router_probabilities = torch.softmax(router_logits, dim=-1)
        top_weights, top_indices = torch.topk(router_probabilities, self.top_k, dim=-1)
        top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)
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
        auxiliary = {
            "balance_loss": balance_loss,
            "z_loss": z_loss,
            "router_logits": router_logits,
            "router_probabilities": router_probabilities,
            "top_indices": top_indices,
            "top_weights": top_weights,
            "selected_load": selected_load,
            "accepted_load": accepted_load,
            "dropped": dropped,
            "capacity": torch.tensor(capacity, device=x.device),
        }
        return output.reshape(original_shape), auxiliary


@torch.no_grad()
def upcycle_expert(source: Expert, target: TopKMoE, noise_std: float = 0.0) -> None:
    """把 Dense Expert 复制到 MoE，并在有共享专家时保持初始输出尺度。"""

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
