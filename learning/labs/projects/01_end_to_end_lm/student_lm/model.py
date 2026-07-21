"""Project 01 decoder-only Transformer starter."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int
    d_model: int = 64
    num_heads: int = 4
    num_layers: int = 2
    d_ff: int = 176
    max_seq_len: int = 128
    rope_base: float = 10_000.0
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if (
            min(
                self.vocab_size,
                self.d_model,
                self.num_heads,
                self.num_layers,
                self.d_ff,
                self.max_seq_len,
            )
            <= 0
        ):
            raise ValueError("all sizes must be positive")
        if self.d_model % self.num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        if (self.d_model // self.num_heads) % 2:
            raise ValueError("RoPE head dimension must be even")
        if self.rope_base <= 1:
            raise ValueError("rope_base must be greater than one")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        # TODO: normalize by root mean square in a stable dtype; do not subtract the mean.
        raise NotImplementedError


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        # TODO: SiLU belongs on the gate branch before elementwise multiplication.
        raise NotImplementedError


def apply_rope(x: Tensor, positions: Tensor, base: float = 10_000.0) -> Tensor:
    """Apply RoPE to ``x:[B,H,T,D]`` using absolute ``positions:[T]``."""

    # TODO: rotate even/odd pairs, preserve shape/dtype/device, and reject invalid inputs.
    raise NotImplementedError


def causal_attention(query: Tensor, key: Tensor, value: Tensor, dropout_p: float = 0.0) -> Tensor:
    """Scaled causal attention for tensors shaped ``[B,H,T,D]``."""

    # TODO: scale by sqrt(D), mask future keys before softmax, then combine values.
    raise NotImplementedError


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.output = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(self, x: Tensor, positions: Tensor) -> Tensor:
        # TODO: project/split heads, apply RoPE only to Q/K, attend, merge heads, project.
        raise NotImplementedError


class TransformerBlock(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(config.d_model)
        self.attention = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.d_model)
        self.ffn = SwiGLU(config.d_model, config.d_ff)

    def forward(self, x: Tensor, positions: Tensor) -> Tensor:
        # TODO: implement Pre-Norm residual order for attention and feed-forward branches.
        raise NotImplementedError


class TransformerLM(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.num_layers))
        self.final_norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, token_ids: Tensor) -> Tensor:
        """Return next-token logits shaped ``[B,T,V]``."""

        # TODO: validate IDs/length, create positions, run blocks, norm and tied output head.
        raise NotImplementedError

    @torch.no_grad()
    def generate(
        self,
        prefix: Tensor,
        max_new_tokens: int,
        *,
        temperature: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Autoregressively sample from the model without relying on notebook state."""

        # TODO: crop to max_seq_len, sample only the last logit, append without in-place aliasing.
        raise NotImplementedError
