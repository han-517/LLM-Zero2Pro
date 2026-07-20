# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown] llm_course_preamble=true
# **课程契约** · 周次 11, 12, 13 · 预计 100 分钟 · Starter 02 · 默认 CPU/离线。

# %% [markdown]
# # 01｜注意力实验室
#
# 把 Query 看成问题、Key 看成索引、Value 看成内容。先检查形状和因果性，再画权重。

# %%
import torch

from llm_from_scratch.attention import scaled_dot_product_attention

torch.manual_seed(7)
q = torch.randn(1, 1, 6, 4)
k = torch.randn(1, 1, 6, 4)
v = torch.arange(24, dtype=torch.float32).view(1, 1, 6, 4)
output, weights = scaled_dot_product_attention(q, k, v, causal=True)
print("output:", output.shape, "weights:", weights.shape)
assert torch.allclose(weights.sum(-1), torch.ones(1, 1, 6))

# %%
print("Causal attention weights [query,key]:\n", weights[0, 0].detach())

# %% [markdown]
# ## 反例练习
#
# 把最后一个 `v` 加 1000，重新计算。前 5 个位置的输出必须不变；这比只看三角形图片更能证明因果性。

# %% [markdown] llm_course_enrichment=true
# ## 学习目标与形状账本
#
# 手算 `QKᵀ/√d`、组合 causal/padding mask、验证全遮蔽行安全为零，并用未来扰动证明因果性。
#
# `Q:[B,H,Tq,Dh] · Kᵀ:[B,H,Dh,Tk] → scores:[B,H,Tq,Tk] → weights → V:[B,H,Tk,Dv]`。

# %% llm_course_enrichment=true
import math

import torch

from llm_from_scratch.attention import scaled_dot_product_attention

q = torch.tensor([[[[1.0, 0.0]]]])
k = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
v = torch.tensor([[[[10.0], [20.0]]]])
out, weights = scaled_dot_product_attention(q, k, v)
manual = torch.softmax((q @ k.transpose(-2, -1)) / math.sqrt(2), dim=-1)
torch.testing.assert_close(weights, manual)
print("weights=", weights.flatten().tolist(), "output=", out.item())

# %% [markdown] llm_course_enrichment=true
# ## 1. Mask 语义：`True` 表示允许读取
#
# Causal mask 约束时间方向，padding mask 约束样本有效长度，二者按允许集合的交集组合。浮点 mask 是加到 logits 的 bias，不能与 bool mask 的语义混用。

# %% llm_course_enrichment=true
q = k = v = torch.ones(1, 1, 2, 2)
blocked = torch.zeros(1, 1, 2, 2, dtype=torch.bool)
out, weights = scaled_dot_product_attention(q, k, v, attention_mask=blocked)
assert torch.count_nonzero(weights) == 0 and torch.count_nonzero(out) == 0
print("全遮蔽行安全返回零")

# %% [markdown] llm_course_enrichment=true
# ## 2. 因果性的可证伪实验
#
# 只修改未来位置的 value；若更早位置输出变化，mask 或对齐必然有错。这个测试比“看起来是下三角”更可靠，也适用于融合 kernel。

# %% llm_course_enrichment=true
torch.manual_seed(0)
q = torch.randn(1, 1, 4, 3)
k = torch.randn(1, 1, 4, 3)
v = torch.randn(1, 1, 4, 2)
base, _ = scaled_dot_product_attention(q, k, v, causal=True)
changed = v.clone()
changed[..., 3, :] += 1000
perturbed, _ = scaled_dot_product_attention(q, k, changed, causal=True)
torch.testing.assert_close(base[..., :3, :], perturbed[..., :3, :])
print("未来扰动不影响过去输出")

# %% [markdown] llm_course_enrichment=true
# ## 3. KV Cache 下是矩形 mask
#
# 单 token decode 时 `Tq=1, Tk=已缓存长度+1`。因果线要 bottom-right 对齐；截取普通方阵左上角会错误地只允许读取第一个 key。

# %% llm_course_enrichment=true
from llm_from_scratch.attention import causal_mask

rectangular = causal_mask(query_length=1, key_length=4)
assert rectangular.shape == (1, 4) and rectangular.all()
print(rectangular.int())

# %% [markdown] llm_course_enrichment=true
# ## 练习、互动图与来源
#
# 打开 `../../interactive/attention.html` 改 Q/K/V 和 mask，再填写 starter 02、07。来源：[Attention Is All You Need](https://arxiv.org/abs/1706.03762)、[PyTorch SDPA](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)。

# %% [markdown] llm_course_enrichment=true
# ## 完成断言
#
# - [ ] 缩放是 `√d_head`；[ ] causal+padding 同时生效；[ ] 全遮蔽行输出为零；[ ] 未来扰动测试通过；[ ] 能解释训练方形 mask 与 cache 矩形 mask。
