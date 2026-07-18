from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    """冻结基础 Linear，仅训练低秩 B@A 修正。"""

    def __init__(
        self,
        base: nn.Linear,
        rank: int = 4,
        alpha: float = 8.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError("rank 必须 >= 1")
        self.base = base
        self.rank = rank
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.a = nn.Parameter(torch.empty(rank, base.in_features))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
        for parameter in self.base.parameters():
            parameter.requires_grad = False

    def forward(self, x: Tensor) -> Tensor:
        update = F.linear(F.linear(self.dropout(x), self.a), self.b)
        return self.base(x) + self.scale * update

    def merged_weight(self) -> Tensor:
        return self.base.weight + self.scale * (self.b @ self.a)


def masked_sft_loss(logits: Tensor, labels: Tensor, loss_mask: Tensor) -> Tensor:
    """对已经完成 next-token 对齐的 labels/mask 计算 SFT loss。"""

    if logits.shape[:-1] != labels.shape or labels.shape != loss_mask.shape:
        raise ValueError("logits 前几维、labels 和 loss_mask 必须匹配")
    token_loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="none"
    ).reshape_as(labels)
    weights = loss_mask.to(token_loss.dtype)
    denominator = weights.sum()
    if denominator.item() == 0:
        raise ValueError("loss_mask 至少要包含一个有效 token")
    return torch.sum(token_loss * weights) / denominator


def causal_sft_loss(logits: Tensor, token_ids: Tensor, assistant_mask: Tensor) -> Tensor:
    """从未移位的对话 token 构造因果 SFT loss，避免 answer mask 错一位。

    logits[:, t] 预测 token_ids[:, t+1]，因此 labels 和 assistant mask 都丢弃首位，
    logits 丢弃末位。
    """

    if logits.ndim != 3 or token_ids.ndim != 2 or assistant_mask.ndim != 2:
        raise ValueError("logits 应为 [B,T,V]，token_ids/assistant_mask 应为 [B,T]")
    if logits.shape[:2] != token_ids.shape or token_ids.shape != assistant_mask.shape:
        raise ValueError("logits、token_ids 和 assistant_mask 的 B/T 必须匹配")
    if token_ids.shape[1] < 2:
        raise ValueError("因果 SFT 至少需要两个 token")
    return masked_sft_loss(logits[:, :-1], token_ids[:, 1:], assistant_mask[:, 1:])


def pairwise_reward_loss(chosen_reward: Tensor, rejected_reward: Tensor) -> Tensor:
    if chosen_reward.shape != rejected_reward.shape:
        raise ValueError("chosen_reward 与 rejected_reward 必须同形状")
    return -F.logsigmoid(chosen_reward - rejected_reward).mean()


def dpo_loss(
    policy_chosen_logp: Tensor,
    policy_rejected_logp: Tensor,
    reference_chosen_logp: Tensor,
    reference_rejected_logp: Tensor,
    beta: float = 0.1,
) -> tuple[Tensor, Tensor, Tensor]:
    shapes = {
        policy_chosen_logp.shape,
        policy_rejected_logp.shape,
        reference_chosen_logp.shape,
        reference_rejected_logp.shape,
    }
    if len(shapes) != 1:
        raise ValueError("四组 log-prob 必须同形状")
    policy_ratio = policy_chosen_logp - policy_rejected_logp
    reference_ratio = reference_chosen_logp - reference_rejected_logp
    logits = beta * (policy_ratio - reference_ratio)
    loss = -F.logsigmoid(logits).mean()
    chosen_reward = beta * (policy_chosen_logp - reference_chosen_logp).detach()
    rejected_reward = beta * (policy_rejected_logp - reference_rejected_logp).detach()
    return loss, chosen_reward, rejected_reward


def group_relative_advantages(rewards: Tensor, eps: float = 1e-6) -> Tensor:
    """按每个问题的一组回答计算标准化相对优势；输入 [batch, group]。"""

    if rewards.ndim != 2:
        raise ValueError("rewards 必须是 [batch, group]")
    centered = rewards - rewards.mean(dim=-1, keepdim=True)
    scale = rewards.std(dim=-1, keepdim=True, unbiased=False)
    return torch.where(scale > eps, centered / scale.clamp_min(eps), torch.zeros_like(centered))

