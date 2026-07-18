from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

KVCache = tuple[Tensor, Tensor]


def causal_mask(query_length: int, key_length: int, device: torch.device | None = None) -> Tensor:
    """返回 [query_length, key_length] 的 bool mask；True 表示允许读取。

    query 被视为 key 序列最后 query_length 个位置，因此同一函数同时支持
    完整训练 (Tq=Tk) 和带 KV Cache 的 decode (Tq<Tk)。
    """

    if query_length > key_length:
        raise ValueError("query_length 不能大于 key_length")
    query_positions = torch.arange(
        key_length - query_length, key_length, device=device
    ).unsqueeze(-1)
    key_positions = torch.arange(key_length, device=device).unsqueeze(0)
    return key_positions <= query_positions


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
    """清晰的注意力参考实现。

    query/key/value 形状分别为 [B,H,Tq,D]、[B,H,Tk,D]、[B,H,Tk,Dv]。
    返回 output 与 softmax weights。
    """

    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query/key/value 必须是 [B,H,T,D] 四维张量")
    if query.shape[:2] != key.shape[:2] or key.shape[:3] != value.shape[:3]:
        raise ValueError("batch/head/key length 必须匹配")
    if query.shape[-1] != key.shape[-1]:
        raise ValueError("query 与 key 的 head_dim 必须相同")
    scores = query @ key.transpose(-2, -1) / math.sqrt(query.shape[-1])
    allowed: Tensor | None = None
    if causal:
        allowed = causal_mask(query.shape[-2], key.shape[-2], query.device)
    if attention_mask is not None:
        if attention_mask.dtype == torch.bool:
            allowed = attention_mask if allowed is None else (allowed & attention_mask)
        else:
            scores = scores + attention_mask
    if allowed is not None:
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
    weights = torch.softmax(scores.float(), dim=-1).to(query.dtype)
    weights = F.dropout(weights, p=dropout_p, training=training)
    return weights @ value, weights


def grouped_query_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    causal: bool = False,
    dropout_p: float = 0.0,
    training: bool = False,
) -> tuple[Tensor, Tensor]:
    """不物化重复 K/V 的 Grouped-Query Attention 参考实现。

    query: [B,Hq,Tq,D]；key/value: [B,Hkv,Tk,D/Dv]，且 Hq 能被 Hkv 整除。
    返回 output [B,Hq,Tq,Dv] 和分组权重 [B,Hkv,G,Tq,Tk]。
    """

    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query/key/value 必须是四维张量")
    if query.shape[0] != key.shape[0] or key.shape[:3] != value.shape[:3]:
        raise ValueError("batch、KV head 和 key length 必须匹配")
    if query.shape[-1] != key.shape[-1]:
        raise ValueError("query 与 key 的 head_dim 必须相同")
    query_heads = query.shape[1]
    kv_heads = key.shape[1]
    if query_heads % kv_heads:
        raise ValueError("query head 数必须能被 KV head 数整除")

    batch, _, query_length, head_dim = query.shape
    key_length = key.shape[-2]
    groups = query_heads // kv_heads
    grouped_query = query.reshape(batch, kv_heads, groups, query_length, head_dim)
    scores = torch.einsum("bhgqd,bhkd->bhgqk", grouped_query, key) / math.sqrt(head_dim)
    if causal:
        allowed = causal_mask(query_length, key_length, query.device)
        scores = scores.masked_fill(~allowed.view(1, 1, 1, query_length, key_length), -torch.inf)
    weights = torch.softmax(scores.float(), dim=-1).to(query.dtype)
    weights = F.dropout(weights, p=dropout_p, training=training)
    attended = torch.einsum("bhgqk,bhkv->bhgqv", weights, value)
    return attended.reshape(batch, query_heads, query_length, value.shape[-1]), weights


def split_heads(x: Tensor, num_heads: int) -> Tensor:
    batch, time, width = x.shape
    if width % num_heads:
        raise ValueError("width 必须能被 num_heads 整除")
    return x.view(batch, time, num_heads, width // num_heads).transpose(1, 2)


def merge_heads(x: Tensor) -> Tensor:
    batch, heads, time, head_dim = x.shape
    return x.transpose(1, 2).contiguous().view(batch, time, heads * head_dim)


def apply_rope(x: Tensor, positions: Tensor | None = None, base: float = 10_000.0) -> Tensor:
    """对 [B,H,T,D] 的偶/奇维应用 RoPE。"""

    if x.ndim != 4 or x.shape[-1] % 2:
        raise ValueError("RoPE 输入必须是 [B,H,T,D] 且 D 为偶数")
    time = x.shape[-2]
    half = x.shape[-1] // 2
    if positions is None:
        positions = torch.arange(time, device=x.device)
    if positions.shape != (time,):
        raise ValueError("positions 必须是 [T]")
    inverse_frequency = base ** (
        -torch.arange(0, half, device=x.device, dtype=torch.float32) / half
    )
    angles = positions.float().unsqueeze(-1) * inverse_frequency.unsqueeze(0)
    cos = angles.cos().to(x.dtype).view(1, 1, time, half)
    sin = angles.sin().to(x.dtype).view(1, 1, time, half)
    even = x[..., 0::2]
    odd = x[..., 1::2]
    output = torch.empty_like(x)
    output[..., 0::2] = even * cos - odd * sin
    output[..., 1::2] = even * sin + odd * cos
    return output


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0, bias: bool = False):
        super().__init__()
        if d_model % num_heads:
            raise ValueError("d_model 必须能被 num_heads 整除")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.output = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = dropout

    def forward(self, x: Tensor, cache: KVCache | None = None) -> tuple[Tensor, KVCache]:
        query, key, value = self.qkv(x).chunk(3, dim=-1)
        query = split_heads(query, self.num_heads)
        key = split_heads(key, self.num_heads)
        value = split_heads(value, self.num_heads)
        if cache is not None:
            key = torch.cat((cache[0], key), dim=-2)
            value = torch.cat((cache[1], value), dim=-2)
        attended, _ = scaled_dot_product_attention(
            query,
            key,
            value,
            causal=True,
            dropout_p=self.dropout,
            training=self.training,
        )
        return self.output(merge_heads(attended)), (key, value)


class GroupedQueryAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_query_heads: int,
        num_kv_heads: int,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if d_model % num_query_heads:
            raise ValueError("d_model 必须能被 num_query_heads 整除")
        if num_query_heads % num_kv_heads:
            raise ValueError("num_query_heads 必须能被 num_kv_heads 整除")
        self.d_model = d_model
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_query_heads
        self.groups = num_query_heads // num_kv_heads
        self.q_proj = nn.Linear(d_model, num_query_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=bias)
        self.output = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = dropout

    def forward(self, x: Tensor, cache: KVCache | None = None) -> tuple[Tensor, KVCache]:
        query = split_heads(self.q_proj(x), self.num_query_heads)
        key = split_heads(self.k_proj(x), self.num_kv_heads)
        value = split_heads(self.v_proj(x), self.num_kv_heads)
        if cache is not None:
            key = torch.cat((cache[0], key), dim=-2)
            value = torch.cat((cache[1], value), dim=-2)
        attended, _ = grouped_query_attention(
            query,
            key,
            value,
            causal=True,
            dropout_p=self.dropout,
            training=self.training,
        )
        return self.output(merge_heads(attended)), (key, value)


class SimpleMLA(nn.Module):
    """教学版 Multi-head Latent Attention。

    Cache 保存 [B,T,latent_dim]，而不是每个 KV head 的完整 K/V。
    为突出压缩主线，本实现没有复刻 DeepSeek 的 decoupled RoPE 与权重吸收内核。
    """

    def __init__(self, d_model: int, num_heads: int, latent_dim: int, bias: bool = False):
        super().__init__()
        if d_model % num_heads:
            raise ValueError("d_model 必须能被 num_heads 整除")
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
            latent = torch.cat((latent_cache, latent), dim=1)
        key = split_heads(self.k_up(latent), self.num_heads)
        value = split_heads(self.v_up(latent), self.num_heads)
        attended, _ = scaled_dot_product_attention(query, key, value, causal=True)
        return self.output(merge_heads(attended)), latent


def sliding_window_mask(length: int, window: int, device: torch.device | None = None) -> Tensor:
    if window < 1:
        raise ValueError("window 必须 >= 1")
    query = torch.arange(length, device=device).unsqueeze(-1)
    key = torch.arange(length, device=device).unsqueeze(0)
    return (key <= query) & (key >= query - window + 1)


def causal_linear_attention(query: Tensor, key: Tensor, value: Tensor, eps: float = 1e-6) -> Tensor:
    """ELU+1 特征映射的因果线性注意力递归参考实现。"""

    if query.shape != key.shape or query.shape[:-1] != value.shape[:-1]:
        raise ValueError("query/key 形状相同，且 value 的 B/H/T 必须匹配")
    q = F.elu(query) + 1.0
    k = F.elu(key) + 1.0
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


def gated_delta_rule(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    beta: Tensor,
    decay: Tensor,
    initial_state: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """教学版 Gated DeltaNet 递归更新。

    q/k: [B,H,T,Dk], v: [B,H,T,Dv], beta/decay: [B,H,T,1]。
    state: [B,H,Dk,Dv]。
    """

    if query.shape != key.shape or query.shape[:-1] != value.shape[:-1]:
        raise ValueError("q/k 形状相同，value 的 B/H/T 必须匹配")
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

