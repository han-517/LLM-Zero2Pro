from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

KVCache = tuple[Tensor, Tensor]


def causal_mask(query_length: int, key_length: int, device: torch.device | None = None) -> Tensor:
    """返回 bottom-right 对齐的因果 mask，兼容完整训练与 cached decode。"""

    if query_length < 1 or key_length < 1:
        raise ValueError("query_length 和 key_length 必须 >= 1")
    if query_length > key_length:
        raise ValueError("query_length 不能大于 key_length")
    query_positions = torch.arange(key_length - query_length, key_length, device=device).unsqueeze(
        -1
    )
    key_positions = torch.arange(key_length, device=device).unsqueeze(0)
    return key_positions <= query_positions


def _safe_attention_softmax(scores: Tensor, allowed: Tensor | None) -> Tensor:
    """让完全遮蔽的行返回全零，而不是均匀权重或 NaN。"""

    float_scores = scores.float()
    if allowed is not None:
        float_scores = float_scores.masked_fill(~allowed, -torch.inf)
    weights = torch.softmax(float_scores, dim=-1)
    return torch.nan_to_num(weights, nan=0.0).to(scores.dtype)


def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    causal: bool = False,
    attention_mask: Tensor | None = None,
    dropout_p: float = 0.0,
    training: bool = False,
) -> tuple[Tensor, Tensor]:
    """可读的注意力参考实现，支持组合 causal 与 padding mask。

    bool mask 中 True 表示允许读取；浮点 mask 会加到 logits。完全遮蔽的
    query 行返回零权重与零输出。
    """

    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query/key/value 必须是 [B,H,T,D] 四维张量")
    if query.shape[:2] != key.shape[:2] or key.shape[:3] != value.shape[:3]:
        raise ValueError("batch/head/key length 必须匹配")
    if query.shape[-1] != key.shape[-1]:
        raise ValueError("query 与 key 的 head_dim 必须相同")
    if not 0.0 <= dropout_p < 1.0:
        raise ValueError("dropout_p 必须位于 [0, 1)")
    scores = query @ key.transpose(-2, -1) / math.sqrt(query.shape[-1])
    allowed: Tensor | None = None
    if causal:
        allowed = causal_mask(query.shape[-2], key.shape[-2], query.device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(query.device)
        if attention_mask.dtype == torch.bool:
            allowed = attention_mask if allowed is None else (allowed & attention_mask)
        elif attention_mask.is_floating_point():
            scores = scores + attention_mask
        else:
            raise TypeError("attention_mask 必须是 bool 或浮点张量")
    weights = _safe_attention_softmax(scores, allowed)
    weights = F.dropout(weights, p=dropout_p, training=training)
    return weights @ value, weights


def grouped_query_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    causal: bool = False,
    attention_mask: Tensor | None = None,
    dropout_p: float = 0.0,
    training: bool = False,
) -> tuple[Tensor, Tensor]:
    """不物化重复 K/V 的 GQA；Hkv=1 即 MQA。"""

    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query/key/value 必须是四维张量")
    if query.shape[0] != key.shape[0] or key.shape[:3] != value.shape[:3]:
        raise ValueError("batch、KV head 和 key length 必须匹配")
    if query.shape[-1] != key.shape[-1]:
        raise ValueError("query 与 key 的 head_dim 必须相同")
    if not 0.0 <= dropout_p < 1.0:
        raise ValueError("dropout_p 必须位于 [0, 1)")
    query_heads = query.shape[1]
    kv_heads = key.shape[1]
    if kv_heads < 1 or query_heads % kv_heads:
        raise ValueError("query head 数必须能被非零 KV head 数整除")
    batch, _, query_length, head_dim = query.shape
    key_length = key.shape[-2]
    groups = query_heads // kv_heads
    grouped_query = query.reshape(batch, kv_heads, groups, query_length, head_dim)
    scores = torch.einsum("bhgqd,bhkd->bhgqk", grouped_query, key) / math.sqrt(head_dim)
    allowed: Tensor | None = None
    if causal:
        allowed = causal_mask(query_length, key_length, query.device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(query.device)
        if attention_mask.ndim == 4:
            attention_mask = attention_mask.unsqueeze(2)
        if attention_mask.dtype == torch.bool:
            allowed = attention_mask if allowed is None else (allowed & attention_mask)
        elif attention_mask.is_floating_point():
            scores = scores + attention_mask
        else:
            raise TypeError("attention_mask 必须是 bool 或浮点张量")
    weights = _safe_attention_softmax(scores, allowed)
    weights = F.dropout(weights, p=dropout_p, training=training)
    attended = torch.einsum("bhgqk,bhkv->bhgqv", weights, value)
    return attended.reshape(batch, query_heads, query_length, value.shape[-1]), weights


def split_heads(x: Tensor, num_heads: int) -> Tensor:
    if x.ndim != 3 or num_heads < 1:
        raise ValueError("x 必须是 [B,T,D] 且 num_heads >= 1")
    batch, time, width = x.shape
    if width % num_heads:
        raise ValueError("width 必须能被 num_heads 整除")
    return x.view(batch, time, num_heads, width // num_heads).transpose(1, 2)


def merge_heads(x: Tensor) -> Tensor:
    if x.ndim != 4:
        raise ValueError("x 必须是 [B,H,T,D]")
    batch, heads, time, head_dim = x.shape
    return x.transpose(1, 2).contiguous().view(batch, time, heads * head_dim)


def apply_rope(
    x: Tensor,
    positions: Tensor | None = None,
    base: float = 10_000.0,
    rotary_dim: int | None = None,
) -> Tensor:
    """对 [B,H,T,D] 的前 rotary_dim 维应用 RoPE。

    positions 可为共享的 [T]，也可为每个样本不同的 [B,T]；函数会把它移到
    x 所在设备。rotary_dim 之后的维度保持 NoPE。
    """

    if x.ndim != 4:
        raise ValueError("RoPE 输入必须是 [B,H,T,D]")
    if base <= 0:
        raise ValueError("base 必须为正数")
    time = x.shape[-2]
    rotary_dim = x.shape[-1] if rotary_dim is None else rotary_dim
    if rotary_dim < 2 or rotary_dim > x.shape[-1] or rotary_dim % 2:
        raise ValueError("rotary_dim 必须是不超过 head_dim 的正偶数")
    if positions is None:
        positions = torch.arange(time, device=x.device)
    if positions.ndim == 1:
        if positions.shape != (time,):
            raise ValueError("一维 positions 必须是 [T]")
        position_angles = positions.to(device=x.device, dtype=torch.float32).view(1, time, 1)
    elif positions.ndim == 2:
        if positions.shape != (x.shape[0], time):
            raise ValueError("二维 positions 必须是 [B,T]")
        position_angles = positions.to(device=x.device, dtype=torch.float32).unsqueeze(-1)
    else:
        raise ValueError("positions 必须是 [T] 或 [B,T]")
    half = rotary_dim // 2
    inverse_frequency = base ** (
        -torch.arange(0, half, device=x.device, dtype=torch.float32) / half
    )
    angles = position_angles * inverse_frequency.view(1, 1, half)
    cos = angles.cos().to(x.dtype).unsqueeze(1)
    sin = angles.sin().to(x.dtype).unsqueeze(1)
    rotated_input = x[..., :rotary_dim]
    even = rotated_input[..., 0::2]
    odd = rotated_input[..., 1::2]
    rotated = torch.empty_like(rotated_input)
    rotated[..., 0::2] = even * cos - odd * sin
    rotated[..., 1::2] = even * sin + odd * cos
    if rotary_dim == x.shape[-1]:
        return rotated
    return torch.cat((rotated, x[..., rotary_dim:]), dim=-1)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = False,
        rope_base: float | None = None,
        rope_fraction: float = 1.0,
    ) -> None:
        super().__init__()
        if d_model < 1 or num_heads < 1 or d_model % num_heads:
            raise ValueError("d_model 和 num_heads 必须为正，且 d_model 能被 num_heads 整除")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout 必须位于 [0, 1)")
        if not 0.0 < rope_fraction <= 1.0:
            raise ValueError("rope_fraction 必须位于 (0, 1]")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.rope_base = rope_base
        self.rotary_dim = int(self.head_dim * rope_fraction) // 2 * 2
        if rope_base is not None and self.rotary_dim < 2:
            raise ValueError("启用 RoPE 时 rotary_dim 必须至少为 2")
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.output = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = dropout

    def forward(
        self,
        x: Tensor,
        cache: KVCache | None = None,
        *,
        position_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        return_cache: bool = True,
    ) -> tuple[Tensor, KVCache | None]:
        query, key, value = self.qkv(x).chunk(3, dim=-1)
        query = split_heads(query, self.num_heads)
        key = split_heads(key, self.num_heads)
        value = split_heads(value, self.num_heads)
        cached_length = 0 if cache is None else cache[0].shape[-2]
        if self.rope_base is not None:
            if position_ids is None:
                position_ids = torch.arange(
                    cached_length, cached_length + x.shape[1], device=x.device
                )
            query = apply_rope(query, position_ids, self.rope_base, self.rotary_dim)
            key = apply_rope(key, position_ids, self.rope_base, self.rotary_dim)
        if cache is not None:
            if cache[0].shape[:-2] != key.shape[:-2] or cache[0].shape[-1] != key.shape[-1]:
                raise ValueError("cache K 与当前 K 的 batch/head/head_dim 不匹配")
            if cache[1].shape[:-2] != value.shape[:-2] or cache[1].shape[-1] != value.shape[-1]:
                raise ValueError("cache V 与当前 V 的 batch/head/head_dim 不匹配")
            key = torch.cat((cache[0], key), dim=-2)
            value = torch.cat((cache[1], value), dim=-2)
        attended, _ = scaled_dot_product_attention(
            query,
            key,
            value,
            causal=True,
            attention_mask=attention_mask,
            dropout_p=self.dropout,
            training=self.training,
        )
        return self.output(merge_heads(attended)), (key, value) if return_cache else None


class GroupedQueryAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_query_heads: int,
        num_kv_heads: int,
        dropout: float = 0.0,
        bias: bool = False,
        rope_base: float | None = None,
        rope_fraction: float = 1.0,
    ) -> None:
        super().__init__()
        if d_model < 1 or num_query_heads < 1 or num_kv_heads < 1:
            raise ValueError("d_model、num_query_heads、num_kv_heads 必须为正")
        if d_model % num_query_heads or num_query_heads % num_kv_heads:
            raise ValueError("head 配置必须整除")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout 必须位于 [0, 1)")
        if not 0.0 < rope_fraction <= 1.0:
            raise ValueError("rope_fraction 必须位于 (0, 1]")
        self.d_model = d_model
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_query_heads
        self.groups = num_query_heads // num_kv_heads
        self.rope_base = rope_base
        self.rotary_dim = int(self.head_dim * rope_fraction) // 2 * 2
        if rope_base is not None and self.rotary_dim < 2:
            raise ValueError("启用 RoPE 时 rotary_dim 必须至少为 2")
        self.q_proj = nn.Linear(d_model, num_query_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=bias)
        self.output = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = dropout

    def forward(
        self,
        x: Tensor,
        cache: KVCache | None = None,
        *,
        position_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        return_cache: bool = True,
    ) -> tuple[Tensor, KVCache | None]:
        query = split_heads(self.q_proj(x), self.num_query_heads)
        key = split_heads(self.k_proj(x), self.num_kv_heads)
        value = split_heads(self.v_proj(x), self.num_kv_heads)
        cached_length = 0 if cache is None else cache[0].shape[-2]
        if self.rope_base is not None:
            if position_ids is None:
                position_ids = torch.arange(
                    cached_length, cached_length + x.shape[1], device=x.device
                )
            query = apply_rope(query, position_ids, self.rope_base, self.rotary_dim)
            key = apply_rope(key, position_ids, self.rope_base, self.rotary_dim)
        if cache is not None:
            if cache[0].shape[:-2] != key.shape[:-2] or cache[0].shape[-1] != key.shape[-1]:
                raise ValueError("cache K 与当前 K 的 batch/head/head_dim 不匹配")
            if cache[1].shape[:-2] != value.shape[:-2] or cache[1].shape[-1] != value.shape[-1]:
                raise ValueError("cache V 与当前 V 的 batch/head/head_dim 不匹配")
            key = torch.cat((cache[0], key), dim=-2)
            value = torch.cat((cache[1], value), dim=-2)
        attended, _ = grouped_query_attention(
            query,
            key,
            value,
            causal=True,
            attention_mask=attention_mask,
            dropout_p=self.dropout,
            training=self.training,
        )
        return self.output(merge_heads(attended)), (key, value) if return_cache else None


@dataclass(frozen=True)
class MLACost:
    dense_cache_bytes: int
    latent_cache_bytes: int
    reconstruction_macs_per_decode_step: int

    @property
    def compression_ratio(self) -> float:
        return self.dense_cache_bytes / self.latent_cache_bytes


def mla_cache_cost(
    *,
    batch_size: int,
    layers: int,
    sequence_length: int,
    d_model: int,
    latent_dim: int,
    bytes_per_element: int = 2,
) -> MLACost:
    """比较完整 K/V 与教学重建基线的存储和重建 MAC。"""

    values = (batch_size, layers, sequence_length, d_model, latent_dim, bytes_per_element)
    if any(value < 1 for value in values):
        raise ValueError("所有成本参数必须 >= 1")
    dense = batch_size * layers * sequence_length * 2 * d_model * bytes_per_element
    latent = batch_size * layers * sequence_length * latent_dim * bytes_per_element
    reconstruction = batch_size * layers * sequence_length * 2 * latent_dim * d_model
    return MLACost(dense, latent, reconstruction)


class LatentCacheMLABaseline(nn.Module):
    """只演示 latent cache 压缩、每步重建全部 K/V 的教学基线。

    它没有实现 decoupled RoPE 或投影吸收，不能代表生产 MLA 的 decode 计算路径。
    """

    def __init__(self, d_model: int, num_heads: int, latent_dim: int, bias: bool = False):
        super().__init__()
        if d_model < 1 or num_heads < 1 or latent_dim < 1 or d_model % num_heads:
            raise ValueError("d_model/num_heads/latent_dim 必须为正且维度可整除")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.latent_dim = latent_dim
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.kv_down = nn.Linear(d_model, latent_dim, bias=bias)
        self.k_up = nn.Linear(latent_dim, d_model, bias=bias)
        self.v_up = nn.Linear(latent_dim, d_model, bias=bias)
        self.output = nn.Linear(d_model, d_model, bias=bias)

    def forward(self, x: Tensor, latent_cache: Tensor | None = None) -> tuple[Tensor, Tensor]:
        query = split_heads(self.q_proj(x), self.num_heads)
        latent = self.kv_down(x)
        if latent_cache is not None:
            if latent_cache.shape[0] != x.shape[0] or latent_cache.shape[-1] != self.latent_dim:
                raise ValueError("latent_cache 的 batch 或 latent_dim 不匹配")
            latent = torch.cat((latent_cache, latent), dim=1)
        key = split_heads(self.k_up(latent), self.num_heads)
        value = split_heads(self.v_up(latent), self.num_heads)
        attended, _ = scaled_dot_product_attention(query, key, value, causal=True)
        return self.output(merge_heads(attended)), latent


SimpleMLA = LatentCacheMLABaseline


def sliding_window_mask(
    query_length: int,
    window: int,
    device: torch.device | None = None,
    *,
    key_length: int | None = None,
) -> Tensor:
    """完整或 cached attention 使用的 bottom-right 因果滑窗 mask。"""

    key_length = query_length if key_length is None else key_length
    if query_length < 1 or key_length < 1 or query_length > key_length:
        raise ValueError("长度必须为正，且 query_length <= key_length")
    if window < 1:
        raise ValueError("window 必须 >= 1")
    query = torch.arange(key_length - query_length, key_length, device=device).unsqueeze(-1)
    key = torch.arange(key_length, device=device).unsqueeze(0)
    return (key <= query) & (key >= query - window + 1)


def _linear_inputs(query: Tensor, key: Tensor, value: Tensor) -> tuple[Tensor, Tensor]:
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query/key/value 必须是 [B,H,T,D]")
    if query.shape != key.shape or query.shape[:-1] != value.shape[:-1]:
        raise ValueError("query/key 形状相同，且 value 的 B/H/T 必须匹配")
    return F.elu(query) + 1.0, F.elu(key) + 1.0


def causal_linear_attention(query: Tensor, key: Tensor, value: Tensor, eps: float = 1e-6) -> Tensor:
    """ELU+1 特征映射的因果线性注意力递归参考实现。"""

    if eps <= 0:
        raise ValueError("eps 必须为正")
    q, k = _linear_inputs(query, key, value)
    batch, heads, time, key_dim = q.shape
    value_dim = value.shape[-1]
    state = torch.zeros(batch, heads, key_dim, value_dim, device=q.device, dtype=q.dtype)
    normalizer = torch.zeros(batch, heads, key_dim, device=q.device, dtype=q.dtype)
    outputs: list[Tensor] = []
    for index in range(time):
        current_k = k[:, :, index]
        current_v = value[:, :, index]
        state = state + torch.einsum("bhd,bhv->bhdv", current_k, current_v)
        normalizer = normalizer + current_k
        current_q = q[:, :, index]
        numerator = torch.einsum("bhd,bhdv->bhv", current_q, state)
        denominator = torch.einsum("bhd,bhd->bh", current_q, normalizer).unsqueeze(-1)
        outputs.append(numerator / denominator.clamp_min(eps))
    return torch.stack(outputs, dim=2)


def causal_linear_attention_parallel(
    query: Tensor, key: Tensor, value: Tensor, eps: float = 1e-6
) -> Tensor:
    """与递归形式相同的 ELU+1 因果线性注意力 parallel-prefix 实现。"""

    if eps <= 0:
        raise ValueError("eps 必须为正")
    q, k = _linear_inputs(query, key, value)
    states = torch.einsum("bhtd,bhtv->bhtdv", k, value).cumsum(dim=2)
    normalizers = k.cumsum(dim=2)
    numerator = torch.einsum("bhtd,bhtdv->bhtv", q, states)
    denominator = torch.einsum("bhtd,bhtd->bht", q, normalizers).unsqueeze(-1)
    return numerator / denominator.clamp_min(eps)


def gated_delta_rule(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    beta: Tensor,
    decay: Tensor,
    initial_state: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """教学版 gated delta-rule 状态更新，不是生产融合训练内核。"""

    if query.ndim != 4 or query.shape != key.shape or query.shape[:-1] != value.shape[:-1]:
        raise ValueError("q/k 必须同为 [B,H,T,Dk]，value 的 B/H/T 必须匹配")
    if beta.shape != (*query.shape[:-1], 1) or decay.shape != beta.shape:
        raise ValueError("beta 和 decay 必须是 [B,H,T,1]")
    batch, heads, time, key_dim = query.shape
    value_dim = value.shape[-1]
    if initial_state is None:
        state = torch.zeros(
            batch, heads, key_dim, value_dim, device=query.device, dtype=query.dtype
        )
    else:
        expected = (batch, heads, key_dim, value_dim)
        if initial_state.shape != expected:
            raise ValueError(f"initial_state 应为 {expected}")
        state = initial_state
    normalized_key = F.normalize(key, dim=-1)
    outputs: list[Tensor] = []
    for index in range(time):
        current_key = normalized_key[:, :, index]
        current_value = value[:, :, index]
        prediction = torch.einsum("bhd,bhdv->bhv", current_key, state)
        delta = beta[:, :, index] * (current_value - prediction)
        state = decay[:, :, index].unsqueeze(-1) * state
        state = state + torch.einsum("bhd,bhv->bhdv", current_key, delta)
        outputs.append(torch.einsum("bhd,bhdv->bhv", query[:, :, index], state))
    return torch.stack(outputs, dim=2), state
