from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from llm_from_scratch.attention import GroupedQueryAttention, KVCache, MultiHeadAttention


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        normalized = x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (normalized * self.weight.float()).to(x.dtype)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, bias: bool = False) -> None:
        super().__init__()
        self.gate = nn.Linear(d_model, hidden_dim, bias=bias)
        self.up = nn.Linear(d_model, hidden_dim, bias=bias)
        self.down = nn.Linear(hidden_dim, d_model, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_kv_head: int | None = None
    d_model: int = 128
    hidden_dim: int | None = None
    dropout: float = 0.0
    bias: bool = False
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.vocab_size < 2 or self.block_size < 1 or self.n_layer < 1:
            raise ValueError("vocab_size、block_size 和 n_layer 非法")
        if self.d_model % self.n_head:
            raise ValueError("d_model 必须能被 n_head 整除")
        kv_heads = self.n_kv_head or self.n_head
        if self.n_head % kv_heads:
            raise ValueError("n_head 必须能被 n_kv_head 整除")

    @property
    def mlp_hidden_dim(self) -> int:
        # SwiGLU 有两个升维投影；约 8D/3 能接近普通 4D MLP 的参数量。
        return self.hidden_dim or ((8 * self.d_model // 3 + 7) // 8 * 8)


class TransformerBlock(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(config.d_model)
        kv_heads = config.n_kv_head or config.n_head
        if kv_heads == config.n_head:
            self.attention = MultiHeadAttention(
                config.d_model, config.n_head, config.dropout, config.bias
            )
        else:
            self.attention = GroupedQueryAttention(
                config.d_model,
                config.n_head,
                kv_heads,
                config.dropout,
                config.bias,
            )
        self.mlp_norm = RMSNorm(config.d_model)
        self.mlp = SwiGLU(config.d_model, config.mlp_hidden_dim, config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor, cache: KVCache | None = None) -> tuple[Tensor, KVCache]:
        attention_output, new_cache = self.attention(self.attention_norm(x), cache)
        x = x + self.dropout(attention_output)
        x = x + self.dropout(self.mlp(self.mlp_norm(x)))
        return x, new_cache


class TinyGPT(nn.Module):
    """CPU 友好的 Pre-Norm Decoder-only Transformer。"""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.block_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.final_norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        token_ids: Tensor,
        targets: Tensor | None = None,
        caches: list[KVCache] | None = None,
    ) -> tuple[Tensor, Tensor | None, list[KVCache]]:
        if token_ids.ndim != 2:
            raise ValueError("token_ids 必须是 [B,T]")
        batch, time = token_ids.shape
        del batch
        cached_length = 0 if caches is None else caches[0][0].shape[-2]
        if cached_length + time > self.config.block_size:
            raise ValueError("输入与 cache 超过 block_size")
        positions = torch.arange(cached_length, cached_length + time, device=token_ids.device)
        x = self.token_embedding(token_ids) + self.position_embedding(positions)
        x = self.dropout(x)
        if caches is None:
            caches = [None] * len(self.blocks)  # type: ignore[list-item]
        if len(caches) != len(self.blocks):
            raise ValueError("caches 数量必须等于 n_layer")
        new_caches: list[KVCache] = []
        for block, cache in zip(self.blocks, caches, strict=True):
            x, new_cache = block(x, cache)
            new_caches.append(new_cache)
        logits = self.lm_head(self.final_norm(x))
        loss = None
        if targets is not None:
            if targets.shape != token_ids.shape:
                raise ValueError("targets 必须与 token_ids 同形状")
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        return logits, loss, new_caches

    @torch.no_grad()
    def generate(
        self,
        token_ids: Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        *,
        use_cache: bool = True,
    ) -> Tensor:
        if token_ids.ndim != 2 or token_ids.shape[1] == 0:
            raise ValueError("token_ids 必须是非空 [B,T]")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens 不能为负")
        if temperature < 0:
            raise ValueError("temperature 不能为负数")
        if top_k is not None and top_k < 1:
            raise ValueError("top_k 必须 >= 1")

        was_training = self.training
        self.eval()
        try:
            output = token_ids
            context = output[:, -self.config.block_size :]
            logits, _, caches = self(context)
            for step in range(max_new_tokens):
                next_logits = logits[:, -1]
                if temperature == 0:
                    next_token = next_logits.argmax(dim=-1, keepdim=True)
                else:
                    next_logits = next_logits / temperature
                    if top_k is not None:
                        k = min(top_k, next_logits.shape[-1])
                        threshold = torch.topk(next_logits, k).values[:, -1, None]
                        next_logits = next_logits.masked_fill(
                            next_logits < threshold, float("-inf")
                        )
                    probabilities = torch.softmax(next_logits, dim=-1)
                    next_token = torch.multinomial(probabilities, num_samples=1)
                output = torch.cat((output, next_token), dim=1)
                if step + 1 == max_new_tokens:
                    break

                if use_cache and caches[0][0].shape[-2] < self.config.block_size:
                    logits, _, caches = self(next_token, caches=caches)
                else:
                    # 学习式绝对位置在滑动窗口后会重新编号，因此 cache 满时必须重建。
                    context = output[:, -self.config.block_size :]
                    logits, _, caches = self(context)
            return output
        finally:
            self.train(was_training)

    def parameter_count(self, trainable_only: bool = False) -> int:
        parameters = self.parameters()
        if trainable_only:
            return sum(parameter.numel() for parameter in parameters if parameter.requires_grad)
        return sum(parameter.numel() for parameter in parameters)

