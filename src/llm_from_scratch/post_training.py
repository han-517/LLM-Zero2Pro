from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _require_finite(name: str, value: Tensor) -> None:
    if value.numel() == 0:
        raise ValueError(f"{name} 不能为空")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} 必须全部为有限值")


class LoRALinear(nn.Module):
    """冻结基础 Linear，仅训练低秩 ``B @ A`` 修正。"""

    def __init__(
        self,
        base: nn.Linear,
        rank: int = 4,
        alpha: float = 8.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("base 必须是 nn.Linear")
        if rank < 1:
            raise ValueError("rank 必须 >= 1")
        if alpha <= 0 or not math.isfinite(alpha):
            raise ValueError("alpha 必须是有限正数")
        if not 0 <= dropout < 1:
            raise ValueError("dropout 必须位于 [0, 1)")
        self.base = base
        self.rank = rank
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.a = nn.Parameter(torch.empty(rank, base.in_features))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank))
        self._merged = False
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
        for parameter in self.base.parameters():
            parameter.requires_grad = False

    @property
    def is_merged(self) -> bool:
        return self._merged

    def forward(self, x: Tensor) -> Tensor:
        if self._merged:
            return self.base(x)
        update = F.linear(F.linear(self.dropout(x), self.a), self.b)
        return self.base(x) + self.scale * update

    def merged_weight(self) -> Tensor:
        if self._merged:
            return self.base.weight
        return self.base.weight + self.scale * (self.b @ self.a)

    @torch.no_grad()
    def merge_(self) -> LoRALinear:
        """把 adapter 合入底座；有 dropout 时要求 eval，避免误解训练语义。"""

        if self._merged:
            return self
        if self.training and self.dropout.p:
            raise RuntimeError("含 dropout 的 LoRA 必须切换到 eval() 后再合并")
        self.base.weight.add_(self.scale * (self.b @ self.a).to(self.base.weight.dtype))
        self._merged = True
        return self

    @torch.no_grad()
    def unmerge_(self) -> LoRALinear:
        """撤销 ``merge_``；重复调用是安全的。"""

        if not self._merged:
            return self
        self.base.weight.sub_(self.scale * (self.b @ self.a).to(self.base.weight.dtype))
        self._merged = False
        return self


def response_only_collator(
    token_sequences: Sequence[Sequence[int]],
    assistant_masks: Sequence[Sequence[bool | int]],
    *,
    pad_token_id: int,
) -> dict[str, Tensor]:
    """右侧 padding 对话并产生 response-only labels。

    ``labels`` 与 ``input_ids`` 尚未因果右移；prompt/pad 位置为 -100。可把
    ``input_ids`` 和 ``assistant_mask`` 交给 :func:`causal_sft_loss`，或让框架内部
    使用 ``labels`` 处理 shift。
    """

    if not token_sequences or len(token_sequences) != len(assistant_masks):
        raise ValueError("token_sequences/assistant_masks 必须是等长非空 batch")
    maximum = max(len(sequence) for sequence in token_sequences)
    if maximum < 2:
        raise ValueError("每个 batch 至少需要可容纳两个 token")
    batch_size = len(token_sequences)
    input_ids = torch.full((batch_size, maximum), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, maximum), dtype=torch.bool)
    assistant_mask = torch.zeros((batch_size, maximum), dtype=torch.bool)
    for row, (tokens, mask) in enumerate(zip(token_sequences, assistant_masks, strict=True)):
        if len(tokens) < 2 or len(tokens) != len(mask):
            raise ValueError("每条 token 序列至少两个 token，且必须与 assistant mask 等长")
        if any(not isinstance(token, int) for token in tokens):
            raise TypeError("token 必须是整数")
        if any(value not in (False, True, 0, 1) for value in mask):
            raise ValueError("assistant mask 只能包含 0/1 或布尔值")
        length = len(tokens)
        input_ids[row, :length] = torch.tensor(tokens, dtype=torch.long)
        attention_mask[row, :length] = True
        assistant_mask[row, :length] = torch.tensor(mask, dtype=torch.bool)
    if not assistant_mask.any(dim=-1).all():
        raise ValueError("每条序列至少需要一个 assistant token")
    labels = input_ids.masked_fill(~assistant_mask, -100)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "assistant_mask": assistant_mask,
        "labels": labels,
    }


def sequence_logprob(
    logits: Tensor,
    token_ids: Tensor,
    response_mask: Tensor,
    *,
    reduction: Literal["sum", "mean"] = "sum",
) -> Tensor:
    """从未移位的因果 LM 输出计算每条 response 的序列 log-prob。"""

    if logits.ndim != 3 or token_ids.ndim != 2 or response_mask.ndim != 2:
        raise ValueError("logits 应为 [B,T,V]，token_ids/response_mask 应为 [B,T]")
    if logits.shape[:2] != token_ids.shape or token_ids.shape != response_mask.shape:
        raise ValueError("logits、token_ids 和 response_mask 的 B/T 必须匹配")
    if token_ids.shape[1] < 2 or reduction not in ("sum", "mean"):
        raise ValueError("序列至少两个 token，reduction 必须是 sum 或 mean")
    _require_finite("logits", logits)
    if token_ids.dtype != torch.long:
        raise TypeError("token_ids 必须是 torch.long")
    if not torch.all((response_mask == 0) | (response_mask == 1)):
        raise ValueError("response_mask 必须是二值 mask")
    labels = token_ids[:, 1:]
    if labels.min() < 0 or labels.max() >= logits.shape[-1]:
        raise ValueError("token_ids 超出词表范围")
    mask = response_mask[:, 1:].bool()
    counts = mask.sum(dim=-1)
    if (counts == 0).any():
        raise ValueError("每条序列至少需要一个可计分 response token")
    token_logp = F.log_softmax(logits[:, :-1], dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    result = (token_logp * mask).sum(dim=-1)
    if reduction == "mean":
        result = result / counts
    return result


def masked_sft_loss(logits: Tensor, labels: Tensor, loss_mask: Tensor) -> Tensor:
    """对已经完成 next-token 对齐的 labels/mask 计算 SFT loss。"""

    if logits.ndim < 2 or logits.shape[:-1] != labels.shape or labels.shape != loss_mask.shape:
        raise ValueError("logits 前几维、labels 和 loss_mask 必须匹配")
    _require_finite("logits", logits)
    if labels.dtype != torch.long:
        raise TypeError("labels 必须是 torch.long")
    if not torch.all((loss_mask == 0) | (loss_mask == 1)):
        raise ValueError("loss_mask 必须是二值 mask")
    selected = loss_mask.bool()
    if not selected.any():
        raise ValueError("loss_mask 至少要包含一个有效 token")
    vocabulary_size = logits.shape[-1]
    valid_labels = (labels == -100) | ((labels >= 0) & (labels < vocabulary_size))
    if not valid_labels.all():
        raise ValueError("labels 必须是 -100 或位于词表范围内")
    if (labels[selected] == -100).any():
        raise ValueError("loss_mask 选中的 label 不能是 -100")
    token_loss = F.cross_entropy(
        logits.reshape(-1, vocabulary_size), labels.reshape(-1), reduction="none"
    ).reshape_as(labels)
    return token_loss[selected].mean()


def causal_sft_loss(logits: Tensor, token_ids: Tensor, assistant_mask: Tensor) -> Tensor:
    """从未移位的对话 token 构造因果 SFT loss，避免 answer mask 错一位。"""

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
    _require_finite("chosen_reward", chosen_reward)
    _require_finite("rejected_reward", rejected_reward)
    return -F.logsigmoid(chosen_reward - rejected_reward).mean()


def dpo_loss(
    policy_chosen_logp: Tensor,
    policy_rejected_logp: Tensor,
    reference_chosen_logp: Tensor,
    reference_rejected_logp: Tensor,
    beta: float = 0.1,
) -> tuple[Tensor, Tensor, Tensor]:
    values = (
        policy_chosen_logp,
        policy_rejected_logp,
        reference_chosen_logp,
        reference_rejected_logp,
    )
    if len({value.shape for value in values}) != 1:
        raise ValueError("四组 log-prob 必须同形状")
    if beta <= 0 or not math.isfinite(beta):
        raise ValueError("beta 必须是有限正数")
    for name, value in zip(
        ("policy_chosen", "policy_rejected", "reference_chosen", "reference_rejected"),
        values,
        strict=True,
    ):
        _require_finite(name, value)
    policy_ratio = policy_chosen_logp - policy_rejected_logp
    reference_ratio = reference_chosen_logp - reference_rejected_logp
    logits = beta * (policy_ratio - reference_ratio)
    loss = -F.logsigmoid(logits).mean()
    chosen_reward = beta * (policy_chosen_logp - reference_chosen_logp).detach()
    rejected_reward = beta * (policy_rejected_logp - reference_rejected_logp).detach()
    return loss, chosen_reward, rejected_reward


def group_relative_advantages(rewards: Tensor, eps: float = 1e-6) -> Tensor:
    """只实现 GRPO 家族中的组内 advantage 标准化；不是完整 GRPO。"""

    if rewards.ndim != 2 or rewards.shape[1] < 2:
        raise ValueError("rewards 必须是 [batch, group] 且 group >= 2")
    if not rewards.is_floating_point():
        raise TypeError("rewards 必须是浮点 tensor")
    if eps <= 0 or not math.isfinite(eps):
        raise ValueError("eps 必须是有限正数")
    _require_finite("rewards", rewards)
    centered = rewards - rewards.mean(dim=-1, keepdim=True)
    scale = rewards.std(dim=-1, keepdim=True, unbiased=False)
    return torch.where(scale > eps, centered / scale.clamp_min(eps), torch.zeros_like(centered))


def toy_ppo_clipped_loss(
    new_logp: Tensor,
    old_logp: Tensor,
    advantages: Tensor,
    *,
    clip_epsilon: float = 0.2,
    mask: Tensor | None = None,
) -> Tensor:
    """已给定 log-prob/advantage 的 PPO clipped toy objective，不含 rollout/critic/KL。"""

    if new_logp.shape != old_logp.shape or new_logp.shape != advantages.shape:
        raise ValueError("new_logp、old_logp 和 advantages 必须同形状")
    if not 0 < clip_epsilon < 1 or not math.isfinite(clip_epsilon):
        raise ValueError("clip_epsilon 必须位于 (0, 1)")
    for name, value in (("new_logp", new_logp), ("old_logp", old_logp), ("advantages", advantages)):
        _require_finite(name, value)
    if mask is None:
        selected = torch.ones_like(new_logp, dtype=torch.bool)
    else:
        if mask.shape != new_logp.shape:
            raise ValueError("mask 必须与 log-prob 同形状")
        if not torch.all((mask == 0) | (mask == 1)):
            raise ValueError("mask 必须是二值 mask")
        selected = mask.bool()
    if not selected.any():
        raise ValueError("mask 至少要选择一个位置")
    ratio = torch.exp(new_logp - old_logp.detach())
    fixed_advantages = advantages.detach()
    unclipped = ratio * fixed_advantages
    clipped = ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon) * fixed_advantages
    return -torch.minimum(unclipped, clipped)[selected].mean()


def toy_grpo_clipped_loss(
    new_logp: Tensor,
    old_logp: Tensor,
    rewards: Tensor,
    response_mask: Tensor,
    *,
    clip_epsilon: float = 0.2,
    eps: float = 1e-6,
) -> Tensor:
    """token-ratio GRPO toy objective；不含 rollout、KL、verifier 或策略刷新。

    ``new_logp/old_logp/response_mask`` 为 ``[batch, group, tokens]``，``rewards``
    为 ``[batch, group]``。该函数用于核查符号和 mask，不能称为完整 GRPO 训练器。
    """

    if new_logp.ndim != 3 or new_logp.shape != old_logp.shape:
        raise ValueError("new_logp/old_logp 必须是同形状 [batch, group, tokens]")
    if response_mask.shape != new_logp.shape or rewards.shape != new_logp.shape[:2]:
        raise ValueError("response_mask/rewards 形状必须与 rollout group 对齐")
    advantages = group_relative_advantages(rewards, eps=eps).unsqueeze(-1).expand_as(new_logp)
    return toy_ppo_clipped_loss(
        new_logp,
        old_logp,
        advantages,
        clip_epsilon=clip_epsilon,
        mask=response_mask,
    )
