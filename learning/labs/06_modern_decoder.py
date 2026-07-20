# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
# ---

# %% [markdown] llm_course_preamble=true
# **课程契约** · 周次 16, 17, 18, 19, 20, 21 · 预计 120 分钟 · Starter 03, 07, 08, 09 · 默认 CPU/离线。

# %% [markdown]
# # 经典 Decoder → 现代 Decoder
#
# 一次只改变一个组件：LayerNorm/RMSNorm、GELU/SwiGLU、绝对位置/RoPE、MHA/GQA。

# %%
import torch

from llm_from_scratch.transformer import GPTConfig, TinyGPT

common = dict(block_size=16, n_layer=1, n_head=4, d_model=32, dropout=0.0)
classic = TinyGPT(GPTConfig.classic(64, **common))
modern = TinyGPT(GPTConfig.modern(64, **common))
tokens = torch.randint(0, 64, (2, 6))
classic_logits, _, _ = classic(tokens, return_caches=False)
modern_logits, _, modern_cache = modern(tokens)
assert classic_logits.shape == modern_logits.shape == (2, 6, 64)
assert modern.position_embedding is None
assert modern_cache is not None
print(
    {
        "classic_parameters": sum(p.numel() for p in classic.parameters()),
        "modern_parameters": sum(p.numel() for p in modern.parameters()),
    }
)

# %% [markdown]
# RoPE 位于 attention 内部，只旋转 Q/K；cache 保存旋转后的 K。完成 07、08、03 后分别核查。

# %% [markdown] llm_course_enrichment=true
# ## 学习目标：做受控消融，不做组件堆砌
#
# 经典：LayerNorm + GELU + MHA + learned absolute position。现代教学预设：RMSNorm + SwiGLU + GQA + RoPE。一次只改一项，并保持 batch、种子、参数预算和训练 token 数可比较。

# %% llm_course_enrichment=true
import torch

from llm_from_scratch.transformer import RMSNorm

x = torch.randn(3, 5, 8)
y = RMSNorm(8)(x)
rms = y.float().pow(2).mean(dim=-1).sqrt()
torch.testing.assert_close(rms, torch.ones_like(rms), atol=2e-5, rtol=2e-5)
print("RMS≈1；RMSNorm 不减均值")

# %% [markdown] llm_course_enrichment=true
# ## 1. RoPE 在 Q/K 内部
#
# RoPE 将通道对按位置旋转，保持范数，并让注意力内积依赖相对位移。它不是残差流中的独立层；V 不旋转，cache 保存已按绝对位置旋转的 K。

# %% llm_course_enrichment=true
from llm_from_scratch.attention import apply_rope

q = torch.randn(1, 1, 1, 8)
k = torch.randn(1, 1, 1, 8)
left = (apply_rope(q, torch.tensor([0])) * apply_rope(k, torch.tensor([2]))).sum()
right = (apply_rope(q, torch.tensor([5])) * apply_rope(k, torch.tensor([7]))).sum()
torch.testing.assert_close(left, right, atol=1e-5, rtol=1e-5)
torch.testing.assert_close(apply_rope(q).norm(), q.norm())
print("共同平移后相对内积不变")

# %% [markdown] llm_course_enrichment=true
# ## 2. MHA → MQA/GQA 改变 KV head
#
# Query head 数保持，减少 KV head 会线性降低 KV cache。`Hkv=Hq` 是 MHA，`Hkv=1` 是 MQA，中间是 GQA。吞吐收益还依赖 kernel、batch 与带宽。

# %% llm_course_enrichment=true
from llm_from_scratch.transformer import GPTConfig, TinyGPT

common = dict(block_size=12, n_layer=1, n_head=4, d_model=32, dropout=0.0)
classic_cfg = GPTConfig.classic(64, **common)
modern_cfg = GPTConfig.modern(64, **common)
classic = TinyGPT(classic_cfg)
modern = TinyGPT(modern_cfg)
print(
    {
        "MHA_kv_heads": classic_cfg.kv_heads,
        "GQA_kv_heads": modern_cfg.kv_heads,
        "relative_KV_cache": modern_cfg.kv_heads / classic_cfg.kv_heads,
        "classic_params": classic.parameter_count(),
        "modern_params": modern.parameter_count(),
    }
)

# %% [markdown] llm_course_enrichment=true
# ## 3. Cache 与完整前向必须等价
#
# Dropout 关闭、position id 相同时，逐 token cached decode logits 应与完整 causal forward 一致。这是 RoPE、矩形 mask 和 K/V 拼接的联合测试。

# %% llm_course_enrichment=true
modern.eval()
tokens = torch.randint(0, 64, (1, 6))
full, _, _ = modern(tokens)
caches = None
pieces = []
for index in range(tokens.shape[1]):
    step, _, caches = modern(tokens[:, index : index + 1], caches=caches)
    pieces.append(step)
cached = torch.cat(pieces, dim=1)
torch.testing.assert_close(full, cached, atol=2e-5, rtol=2e-5)
print("full forward == cached decode")

# %% [markdown] llm_course_enrichment=true
# ## 练习、互动图与来源
#
# 使用 `../../interactive/rope_lab.html` 与架构演化图，再完成 starter 03/07/08。来源：[RoPE](https://arxiv.org/abs/2104.09864)、[GQA](https://arxiv.org/abs/2305.13245)、[SwiGLU](https://arxiv.org/abs/2002.05202)、[RMSNorm](https://arxiv.org/abs/1910.07467)。

# %% [markdown] llm_course_enrichment=true
# ## 完成断言
#
# - [ ] RoPE 只作用 Q/K；[ ] 能计算 KV cache 比例；[ ] 经典/现代配置清晰；[ ] full/cache logits 等价；[ ] 不把 scaling 外推成无限长度。
