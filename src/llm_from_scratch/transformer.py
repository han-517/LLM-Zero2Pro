from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from llm_from_scratch.attention import GroupedQueryAttention, KVCache, MultiHeadAttention


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float = 1e-6) -> None:
        super().__init__()
        if width < 1 or eps <= 0:
            raise ValueError("width 和 eps 必须为正")
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        normalized = x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (normalized * self.weight.float()).to(x.dtype)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, bias: bool = False) -> None:
        super().__init__()
        if d_model < 1 or hidden_dim < 1:
            raise ValueError("d_model 和 hidden_dim 必须为正")
        self.gate = nn.Linear(d_model, hidden_dim, bias=bias)
        self.up = nn.Linear(d_model, hidden_dim, bias=bias)
        self.down = nn.Linear(hidden_dim, d_model, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class GELUMLP(nn.Module):
    """2017/GPT 风格的两层前馈网络，用于经典配置对照。"""

    def __init__(self, d_model: int, hidden_dim: int, bias: bool = True) -> None:
        super().__init__()
        self.up = nn.Linear(d_model, hidden_dim, bias=bias)
        self.down = nn.Linear(hidden_dim, d_model, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(F.gelu(self.up(x)))


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
    norm_type: str = "rmsnorm"
    mlp_type: str = "swiglu"
    position_encoding: str = "learned_absolute"
    rope_base: float = 10_000.0
    rope_fraction: float = 1.0

    def __post_init__(self) -> None:
        if self.vocab_size < 2 or self.block_size < 1 or self.n_layer < 1:
            raise ValueError("vocab_size、block_size 和 n_layer 非法")
        if self.n_head < 1 or self.d_model < 1 or self.d_model % self.n_head:
            raise ValueError("n_head/d_model 必须为正且 d_model 能被 n_head 整除")
        if self.n_kv_head is not None and self.n_kv_head < 1:
            raise ValueError("n_kv_head 必须为正或 None")
        if self.n_head % self.kv_heads:
            raise ValueError("n_head 必须能被 n_kv_head 整除")
        if self.hidden_dim is not None and self.hidden_dim < 1:
            raise ValueError("hidden_dim 必须为正或 None")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout 必须位于 [0, 1)")
        if self.norm_type not in {"layernorm", "rmsnorm"}:
            raise ValueError("norm_type 必须是 layernorm 或 rmsnorm")
        if self.mlp_type not in {"gelu", "swiglu"}:
            raise ValueError("mlp_type 必须是 gelu 或 swiglu")
        if self.position_encoding not in {"learned_absolute", "rope"}:
            raise ValueError("position_encoding 必须是 learned_absolute 或 rope")
        if self.rope_base <= 0 or not 0.0 < self.rope_fraction <= 1.0:
            raise ValueError("rope_base 必须为正且 rope_fraction 位于 (0, 1]")
        rotary_dim = int(self.head_dim * self.rope_fraction) // 2 * 2
        if self.position_encoding == "rope" and rotary_dim < 2:
            raise ValueError("启用 RoPE 时至少需要两个 rotary head dimensions")

    @classmethod
    def classic(cls, vocab_size: int, **overrides: object) -> GPTConfig:
        """经典对照：LayerNorm、GELU MLP、MHA、学习式绝对位置。"""

        values: dict[str, object] = {
            "norm_type": "layernorm",
            "mlp_type": "gelu",
            "position_encoding": "learned_absolute",
            "bias": True,
        }
        values.update(overrides)
        return cls(vocab_size=vocab_size, **values)  # type: ignore[arg-type]

    @classmethod
    def modern(cls, vocab_size: int, **overrides: object) -> GPTConfig:
        """现代教学预设：RMSNorm、SwiGLU、RoPE，默认使用 GQA。"""

        n_head = int(overrides.get("n_head", 4))
        values: dict[str, object] = {
            "norm_type": "rmsnorm",
            "mlp_type": "swiglu",
            "position_encoding": "rope",
            "bias": False,
            "n_kv_head": max(1, n_head // 4),
        }
        values.update(overrides)
        return cls(vocab_size=vocab_size, **values)  # type: ignore[arg-type]

    @property
    def kv_heads(self) -> int:
        return self.n_head if self.n_kv_head is None else self.n_kv_head

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_head

    @property
    def mlp_hidden_dim(self) -> int:
        if self.hidden_dim is not None:
            return self.hidden_dim
        if self.mlp_type == "gelu":
            return 4 * self.d_model
        # SwiGLU 有两个升维投影；约 8D/3 接近普通 4D MLP 的参数量。
        return (8 * self.d_model // 3 + 7) // 8 * 8


class TransformerBlock(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        norm = RMSNorm if config.norm_type == "rmsnorm" else nn.LayerNorm
        self.attention_norm = norm(config.d_model)
        rope_base = config.rope_base if config.position_encoding == "rope" else None
        if config.kv_heads == config.n_head:
            self.attention = MultiHeadAttention(
                config.d_model,
                config.n_head,
                config.dropout,
                config.bias,
                rope_base,
                config.rope_fraction,
            )
        else:
            self.attention = GroupedQueryAttention(
                config.d_model,
                config.n_head,
                config.kv_heads,
                config.dropout,
                config.bias,
                rope_base,
                config.rope_fraction,
            )
        self.mlp_norm = norm(config.d_model)
        if config.mlp_type == "swiglu":
            self.mlp = SwiGLU(config.d_model, config.mlp_hidden_dim, config.bias)
        else:
            self.mlp = GELUMLP(config.d_model, config.mlp_hidden_dim, config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: Tensor,
        cache: KVCache | None = None,
        *,
        position_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        return_cache: bool = True,
    ) -> tuple[Tensor, KVCache | None]:
        attention_output, new_cache = self.attention(
            self.attention_norm(x),
            cache,
            position_ids=position_ids,
            attention_mask=attention_mask,
            return_cache=return_cache,
        )
        x = x + self.dropout(attention_output)
        x = x + self.dropout(self.mlp(self.mlp_norm(x)))
        return x, new_cache


class TinyGPT(nn.Module):
    """CPU 友好的 Pre-Norm Decoder-only Transformer。

    ``GPTConfig.classic`` 与 ``GPTConfig.modern`` 用同一训练接口对照经典和
    现代组件。RoPE 在 attention 内旋转 Q/K，cache 中保存的是旋转后的 K。
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = (
            nn.Embedding(config.block_size, config.d_model)
            if config.position_encoding == "learned_absolute"
            else None
        )
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.final_norm = (
            RMSNorm(config.d_model)
            if config.norm_type == "rmsnorm"
            else nn.LayerNorm(config.d_model)
        )
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
        *,
        position_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        return_caches: bool = True,
    ) -> tuple[Tensor, Tensor | None, list[KVCache] | None]:
        if token_ids.ndim != 2 or token_ids.shape[1] < 1:
            raise ValueError("token_ids 必须是非空 [B,T]")
        batch, time = token_ids.shape
        if caches is not None:
            if len(caches) != len(self.blocks):
                raise ValueError("caches 数量必须等于 n_layer")
            cached_lengths = {cache[0].shape[-2] for cache in caches}
            if len(cached_lengths) != 1:
                raise ValueError("所有层 cache 长度必须相同")
            cached_length = next(iter(cached_lengths))
        else:
            cached_length = 0
        if cached_length + time > self.config.block_size:
            raise ValueError("输入与 cache 超过 block_size")
        if position_ids is None:
            position_ids = torch.arange(
                cached_length, cached_length + time, device=token_ids.device
            )
        elif position_ids.shape not in {(time,), (batch, time)}:
            raise ValueError("position_ids 必须是 [T] 或 [B,T]")
        position_ids = position_ids.to(token_ids.device)

        x = self.token_embedding(token_ids)
        if self.position_embedding is not None:
            if position_ids.min() < 0 or position_ids.max() >= self.config.block_size:
                raise ValueError("绝对 position_ids 超出 block_size")
            x = x + self.position_embedding(position_ids)
        x = self.dropout(x)
        block_caches: list[KVCache | None]
        if caches is None:
            block_caches = [None] * len(self.blocks)
        else:
            block_caches = list(caches)
        new_caches: list[KVCache] = []
        for block, cache in zip(self.blocks, block_caches, strict=True):
            x, new_cache = block(
                x,
                cache,
                position_ids=position_ids,
                attention_mask=attention_mask,
                return_cache=return_caches,
            )
            if return_caches:
                assert new_cache is not None
                new_caches.append(new_cache)
        logits = self.lm_head(self.final_norm(x))
        loss = None
        if targets is not None:
            if targets.shape != token_ids.shape:
                raise ValueError("targets 必须与 token_ids 同形状")
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        return logits, loss, new_caches if return_caches else None

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
            logits, _, caches = self(context, return_caches=use_cache)
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
                    next_token = torch.multinomial(
                        torch.softmax(next_logits, dim=-1), num_samples=1
                    )
                output = torch.cat((output, next_token), dim=1)
                if step + 1 == max_new_tokens:
                    break
                if (
                    use_cache
                    and caches is not None
                    and caches[0][0].shape[-2] < self.config.block_size
                ):
                    logits, _, caches = self(next_token, caches=caches)
                else:
                    # cache 满时重建固定窗口；相对 RoPE 与绝对位置均重新从窗口零点编号。
                    context = output[:, -self.config.block_size :]
                    logits, _, caches = self(context, return_caches=use_cache)
            return output
        finally:
            self.train(was_training)

    def parameter_count(self, trainable_only: bool = False) -> int:
        parameters = self.parameters()
        if trainable_only:
            return sum(parameter.numel() for parameter in parameters if parameter.requires_grad)
        return sum(parameter.numel() for parameter in parameters)
