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
# **课程契约** · 周次 29, 30, 31, 32, 33, 34 · 预计 120 分钟 · Starter 16, 17 · 默认 CPU/离线。

# %% [markdown]
# # 注意力前沿：IO、稀疏、latent cache 与线性状态
#
# 本实验关注数学契约和成本边界，不声称复现生产 kernel。

# %%
import torch

from llm_from_scratch.attention import (
    causal_linear_attention,
    causal_linear_attention_parallel,
    mla_cache_cost,
    sliding_window_mask,
)

mask = sliding_window_mask(2, window=2, key_length=4)
q = torch.randn(1, 2, 6, 4)
k = torch.randn(1, 2, 6, 4)
v = torch.randn(1, 2, 6, 3)
recurrent = causal_linear_attention(q, k, v)
parallel = causal_linear_attention_parallel(q, k, v)
torch.testing.assert_close(recurrent, parallel, atol=2e-5, rtol=2e-5)
cost = mla_cache_cost(batch_size=1, layers=1, sequence_length=128, d_model=32, latent_dim=16)
print(mask.int(), cost)

# %% [markdown]
# MLA 部分是 latent-cache reconstruction baseline；需要区分缓存压缩、历史重建与 absorbed decode。

# %% [markdown] llm_course_enrichment=true
# ## 学习目标：按精确计算、稀疏连接、压缩状态分类
#
# FlashAttention 改 IO 调度但仍是精确 softmax；滑窗改变可见边；MLA 压缩 KV 表示；线性注意力/DeltaNet/Mamba 类方法用固定状态替代显式全历史 KV。

# %% llm_course_enrichment=true
for length in (512, 2048, 8192):
    score_mib = length * length * 2 / 2**20
    print(f"T={length:5d} | 单个 FP16 score matrix≈{score_mib:8.1f} MiB")

# %% [markdown] llm_course_enrichment=true
# ## 1. FlashAttention 不等于稀疏注意力
#
# Online softmax 按块维护行最大值与归一化和，结果可与标准 attention 等价。收益来自减少 HBM 读写；速度取决于硬件、形状、dtype、mask 和版本。

# %% llm_course_enrichment=true
from llm_from_scratch.attention import sliding_window_mask

mask = sliding_window_mask(query_length=6, key_length=6, window=3)
print(mask.int())
assert mask[-1].sum() == 3 and not mask[0, -1]

# %% [markdown] llm_course_enrichment=true
# ## 2. 稀疏窗口改变感受野
#
# 单层窗口只读最近 W 个 key；堆叠层扩大间接感受野但路径变长。长上下文评测需区分 needle retrieval、语言建模与真实任务。

# %% llm_course_enrichment=true
from llm_from_scratch.attention import mla_cache_cost

cost = mla_cache_cost(batch_size=2, layers=24, sequence_length=4096, d_model=1024, latent_dim=64)
print(cost)
assert cost.compression_ratio > 1

# %% [markdown] llm_course_enrichment=true
# ## 3. MLA 的教学边界
#
# `LatentCacheMLABaseline` 演示 latent cache 与历史重建成本，不是完整 absorbed decode。报告时分别说明缓存什么、每步重建什么、投影权重是否吸收。

# %% llm_course_enrichment=true
import torch

from llm_from_scratch.attention import causal_linear_attention, causal_linear_attention_parallel

torch.manual_seed(0)
q = torch.randn(1, 2, 7, 4)
k = torch.randn(1, 2, 7, 4)
v = torch.randn(1, 2, 7, 3)
a = causal_linear_attention(q, k, v)
b = causal_linear_attention_parallel(q, k, v)
torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-5)
print("recurrent == parallel")

# %% [markdown] llm_course_enrichment=true
# ## 4. 线性状态模型的比较口径
#
# 必须写出正特征映射、归一化分母和状态更新；O(T) 不自动意味着短序列更快，也不说明记忆质量。Mamba-2/DeltaNet 是相关但不同的状态更新家族。

# %% [markdown] llm_course_enrichment=true
# ## 来源与验收
#
# [FlashAttention](https://arxiv.org/abs/2205.14135)、[DeepSeek-V2/MLA](https://arxiv.org/abs/2405.04434)、[Linear Transformers](https://arxiv.org/abs/2006.16236)、[Mamba-2](https://arxiv.org/abs/2405.21060)、[Gated DeltaNet](https://arxiv.org/abs/2412.06464)。完成 starter 16–17，并说明保持/近似了什么。
