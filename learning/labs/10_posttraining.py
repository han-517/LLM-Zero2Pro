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
# **课程契约** · 周次 40, 41, 42, 43, 44 · 预计 120 分钟 · Starter 04, 19 · 默认 CPU/离线。

# %% [markdown]
# # 后训练：mask、log-prob 与相对目标
#
# 先验证 token 对齐，再看 SFT、DPO 和组内优势；PPO/GRPO 函数均是 toy objective。

# %%
import torch

from llm_from_scratch.post_training import group_relative_advantages, sequence_logprob

logits = torch.log_softmax(torch.randn(2, 4, 7), dim=-1)
labels = torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5]])
mask = torch.tensor([[0, 1, 1, 0], [0, 0, 1, 1]], dtype=torch.bool)
scores = sequence_logprob(logits, labels, mask)
advantages = group_relative_advantages(torch.tensor([[2.0, 2.0], [1.0, 3.0]]))
assert scores.shape == (2,)
torch.testing.assert_close(advantages[0], torch.zeros(2))
print({"sequence_logprob": scores, "advantages": advantages})

# %% [markdown]
# 完成 04 与 19。任何真实结论都需额外报告 reference policy、response mask、KL、奖励和安全评测。

# %% [markdown] llm_course_enrichment=true
# ## 学习目标：所有目标函数先从数据契约开始
#
# 对每批样本明确 prompt/response 边界、右移、padding、response mask、policy/reference 版本与聚合方式。SFT、DPO、PPO/GRPO 公式不同，但常见错误都在 token 对齐和 mask。

# %% llm_course_enrichment=true
from llm_from_scratch.post_training import response_only_collator

batch = response_only_collator([[1, 2, 3, 4], [5, 6, 7]], [[0, 0, 1, 1], [0, 1, 1]], pad_token_id=0)
print({k: v.int().tolist() for k, v in batch.items()})
assert (batch["labels"][~batch["assistant_mask"]] == -100).all()

# %% [markdown] llm_course_enrichment=true
# ## 1. 因果右移
#
# `logits[:,t]` 预测 `token_ids[:,t+1]`，response mask 也要取 `[:,1:]`。若在同一位置 gather，得到的是读到答案后的虚假分数。

# %% llm_course_enrichment=true
import torch

from llm_from_scratch.post_training import sequence_logprob

torch.manual_seed(0)
logits = torch.randn(2, 4, 10)
tokens = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
mask = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 0]], dtype=torch.bool)
summed = sequence_logprob(logits, tokens, mask)
meaned = sequence_logprob(logits, tokens, mask, reduction="mean")
print({"sum": summed.tolist(), "mean": meaned.tolist()})

# %% [markdown] llm_course_enrichment=true
# ## 2. LoRA 的低秩更新与合并
#
# 基础权重冻结，训练 `ΔW=(α/r)BA`。部署前可 merge；unmerge 应恢复底座。QLoRA 还涉及量化底座、计算 dtype、double quantization 与 paged optimizer。

# %% llm_course_enrichment=true
from torch import nn

from llm_from_scratch.post_training import LoRALinear

layer = LoRALinear(nn.Linear(6, 4), rank=2, alpha=4, dropout=0.0).eval()
with torch.no_grad():
    layer.b.normal_()
x = torch.randn(3, 6)
before = layer(x)
layer.merge_()
merged = layer(x)
torch.testing.assert_close(before, merged, atol=1e-6, rtol=1e-6)
layer.unmerge_()
torch.testing.assert_close(before, layer(x), atol=1e-6, rtol=1e-6)
print("merge/unmerge 数值等价")

# %% [markdown] llm_course_enrichment=true
# ## 3. DPO 比较相对偏好 margin
#
# 策略 chosen-vs-rejected log-ratio 与 reference 的同一差值比较。`β` 控制偏离 reference 的强度；数据质量、长度偏差和 reference 版本会改变结论。

# %% llm_course_enrichment=true
from llm_from_scratch.post_training import dpo_loss

pc = torch.tensor([-1.0, -0.8])
pr = torch.tensor([-2.0, -1.4])
rc = torch.tensor([-1.2, -1.0])
rr = torch.tensor([-1.5, -1.3])
loss, cr, rrw = dpo_loss(pc, pr, rc, rr, beta=0.1)
assert loss.ndim == 0
print({"loss": float(loss), "chosen_reward": cr.tolist(), "rejected_reward": rrw.tolist()})

# %% [markdown] llm_course_enrichment=true
# ## 4. GRPO/RLVR 的边界
#
# 组内标准化只定义 advantage，不包含 rollout、verifier、旧策略刷新、KL、重要性采样或分布式生成。零方差组应返回零而不是 NaN。

# %% llm_course_enrichment=true
from llm_from_scratch.post_training import group_relative_advantages

rewards = torch.tensor([[2.0, 2.0, 2.0], [1.0, 2.0, 3.0]])
advantages = group_relative_advantages(rewards)
torch.testing.assert_close(advantages[0], torch.zeros(3))
torch.testing.assert_close(advantages.mean(-1), torch.zeros(2), atol=1e-6, rtol=0)
print(advantages)

# %% [markdown] llm_course_enrichment=true
# ## 练习与来源
#
# 完成 starter 04、19。来源：[LoRA](https://arxiv.org/abs/2106.09685)、[QLoRA](https://arxiv.org/abs/2305.14314)、[DPO](https://arxiv.org/abs/2305.18290)、[PPO](https://arxiv.org/abs/1707.06347)、[DeepSeekMath/GRPO](https://arxiv.org/abs/2402.03300)。

# %% [markdown] llm_course_enrichment=true
# ## 完成断言
#
# - [ ] response mask 正确右移；[ ] sequence log-prob 可选 sum/mean；[ ] LoRA merge 等价；[ ] DPO 四组 log-prob 对齐；[ ] 不把 toy PPO/GRPO 称为完整系统。
