"""Project 05: SFT, grouped rollout and RLVR policy-update starter."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class RolloutBatch:
    sequences: Tensor
    response_mask: Tensor
    old_token_log_probs: Tensor
    prompt_length: int


@dataclass(frozen=True)
class RLVRMetrics:
    loss: float
    reward_mean: float
    reward_std: float
    approximate_kl: float
    gradient_norm: float


def masked_token_log_probs(logits: Tensor, token_ids: Tensor, response_mask: Tensor) -> Tensor:
    """Return shifted token log-probs ``[B,T-1]``, zero outside response targets.

    ``response_mask[:, t]`` states whether token ``token_ids[:, t]`` belongs to the response;
    token zero has no preceding logit and must therefore be false.
    """

    # TODO: validate B/T/V shapes, apply log_softmax and gather targets token_ids[:, 1:].
    raise NotImplementedError


def sft_loss(logits: Tensor, token_ids: Tensor, response_mask: Tensor) -> Tensor:
    """Mean negative log-prob over response tokens only."""

    # TODO: use masked_token_log_probs and reject batches with no response targets.
    raise NotImplementedError


def sft_train_step(
    model: nn.Module,
    token_ids: Tensor,
    response_mask: Tensor,
    optimizer,
    *,
    max_grad_norm: float = 1.0,
) -> Tensor:
    """Run a real response-only SFT optimizer step and return detached loss."""

    # TODO: zero, forward, masked loss, backward, finite gradients, clip and step.
    raise NotImplementedError


def sample_grouped(
    policy: nn.Module,
    prompts: Tensor,
    *,
    group_size: int,
    max_new_tokens: int,
    temperature: float,
    generator: torch.Generator,
    eos_token_id: int | None = None,
) -> RolloutBatch:
    """Sample G responses per prompt and retain behavior-policy token log-probs.

    Prompts are an unpadded ``[B,P]`` teaching baseline.  Returned sequences and masks are
    ``[B,G,P+N]``; old token log-probs are ``[B,G,P+N-1]`` and detached.
    """

    # TODO: repeat prompts, sample autoregressively, track EOS masks and behavior log-probs.
    raise NotImplementedError


def verifiable_answer_reward(
    sequences: Tensor,
    response_mask: Tensor,
    answer_tokens: Tensor,
) -> Tensor:
    """Give binary reward when the first generated response token equals the answer."""

    # TODO: validate [B,G,T] and [B] contracts, find the first true response position per sample.
    raise NotImplementedError


def group_advantages(rewards: Tensor, *, eps: float = 1e-6) -> Tensor:
    """Normalize rewards within each prompt group using population standard deviation."""

    # TODO: keep zero-variance groups at exactly zero and reject non-finite rewards.
    raise NotImplementedError


def grpo_loss(
    current_token_log_probs: Tensor,
    old_token_log_probs: Tensor,
    reference_token_log_probs: Tensor,
    advantages: Tensor,
    response_mask: Tensor,
    *,
    clip_epsilon: float = 0.2,
    beta: float = 0.01,
) -> Tensor:
    """Token-level clipped surrogate plus the non-negative GRPO KL estimator.

    The response mask has shape ``[B,G,L]`` and advantages ``[B,G]``.  This teaching objective
    averages within each response before averaging samples, so long responses do not dominate.
    """

    # TODO: compute ratios, sign-aware clipped surrogate, exp(ref-current)-delta-1 KL, and masks.
    raise NotImplementedError


def rlvr_train_step(
    policy: nn.Module,
    reference: nn.Module,
    rollout: RolloutBatch,
    rewards: Tensor,
    optimizer,
    *,
    clip_epsilon: float = 0.2,
    beta: float = 0.01,
    max_grad_norm: float = 1.0,
) -> RLVRMetrics:
    """Run one actual grouped verifiable-reward policy update."""

    # TODO: recompute policy/reference log-probs, normalize rewards, backprop GRPO, clip and step.
    raise NotImplementedError
