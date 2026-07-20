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
# **课程契约** · 周次 22, 23, 24, 25, 26, 27, 28 · 预计 110 分钟 · Starter 14, 15 · 默认 CPU/离线。

# %% [markdown]
# # 预训练数据与训练系统
#
# 玩具实验不替代真实数据治理；它强制记录去重、边界、污染和内存假设。

# %%
from llm_from_scratch.training import (
    exact_deduplicate,
    pack_documents,
    training_memory_ledger,
    warmup_cosine_lr,
)

docs, duplicates = exact_deduplicate(["alpha", "beta", "alpha"])
packed, mask = pack_documents([[1, 2], [3]], block_size=4, eos_token_id=9, pad_token_id=0)
ledger = training_memory_ledger(1_000, world_size=2, shard_optimizer=True)
curve = [warmup_cosine_lr(i, total_steps=8, warmup_steps=2, max_lr=1e-3) for i in range(9)]
assert docs == ["alpha", "beta"] and duplicates == [2]
assert packed.shape == mask.shape and ledger["total"] > 0
assert curve[0] == 0 and curve[2] == max(curve)
print({"packed": packed.tolist(), "loss_mask": mask.tolist(), "ledger": ledger})

# %% [markdown]
# 下一步完成 14、15；报告每条规则的保留率，不把 toy 内存账本称为峰值显存保证。

# %% [markdown] llm_course_enrichment=true
# ## 学习目标：把“训练模型”扩展成可审计系统
#
# 数据生命周期：许可与来源 → 清洗/去重 → 污染筛查 → 混合权重 → tokenization/packing → 训练 → 固定协议评测。每一步都应产生日志、版本和可复现统计。

# %% llm_course_enrichment=true
from llm_from_scratch.training import detect_ngram_contamination, exact_deduplicate

train = [
    "alpha beta gamma delta",
    "alpha beta gamma delta",
    "safe unique text",
    "exam secret answer",
]
unique, duplicates = exact_deduplicate(train)
flags = detect_ngram_contamination(unique, ["exam secret answer"], ngram_size=2)
print({"duplicates": duplicates, "contamination_candidates": flags})
assert duplicates == [1] and flags[-1]

# %% [markdown] llm_course_enrichment=true
# ## 1. Packing 必须保留文档边界
#
# 短文档拼块可提高 token 利用率，但新文档首 token 不能以旧文档末尾为条件。真实系统还需定义 EOS/BOS、跨文档 attention 与 position reset。

# %% llm_course_enrichment=true
from llm_from_scratch.training import pack_documents

packed, mask = pack_documents([[11, 12], [21]], block_size=4, eos_token_id=2, pad_token_id=0)
print("ids=", packed.tolist(), "label_mask=", mask.int().tolist())
assert packed.shape == mask.shape and not mask[0, 3]

# %% [markdown] llm_course_enrichment=true
# ## 2. AdamW 与学习率日程是独立机制
#
# AdamW 将 decay 与梯度更新解耦；warmup/cosine 控制逐步学习率。实验需写清全局 batch token、梯度累积、裁剪和 scheduler step 口径。

# %% llm_course_enrichment=true
from llm_from_scratch.training import warmup_cosine_lr

curve = [
    warmup_cosine_lr(s, total_steps=10, warmup_steps=2, max_lr=1e-3, min_lr=1e-4) for s in range(11)
]
assert curve[0] == 0 and curve[2] == max(curve) and abs(curve[-1] - 1e-4) < 1e-12
print([round(v, 6) for v in curve])

# %% [markdown] llm_course_enrichment=true
# ## 3. 内存账本不是峰值显存保证
#
# 参数、梯度、master weights、optimizer states 与 activation 分片策略不同。账本不含 allocator 碎片、通信 buffer、临时 workspace 与框架开销。

# %% llm_course_enrichment=true
from llm_from_scratch.training import training_memory_ledger

base = training_memory_ledger(1_000_000, world_size=4)
sharded = training_memory_ledger(
    1_000_000, world_size=4, shard_parameters=True, shard_gradients=True, shard_optimizer=True
)
print(
    {
        "replicated_MiB": round(base["total"] / 2**20, 2),
        "fully_sharded_MiB": round(sharded["total"] / 2**20, 2),
    }
)
assert sharded["total"] < base["total"]

# %% [markdown] llm_course_enrichment=true
# ## 4. 评测可靠性
#
# 固定 tokenizer、样本、prompt、解码和 metric 版本；混合前筛查污染；不跨 tokenizer 直接比较 perplexity。完成 starter 14、15。

# %% [markdown] llm_course_enrichment=true
# ## 验收、来源与边界
#
# [DataComp-LM](https://arxiv.org/abs/2406.11794)、[FineWeb](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1)、[ZeRO](https://arxiv.org/abs/1910.02054)、[Megatron-LM](https://arxiv.org/abs/1909.08053)、[Chinchilla](https://arxiv.org/abs/2203.15556)。本实验不替代法律审查、集群 profiler 或数据卡。
