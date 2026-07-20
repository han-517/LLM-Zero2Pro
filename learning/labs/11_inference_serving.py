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
# **课程契约** · 周次 45, 46, 47, 48 · 预计 120 分钟 · Starter 20 · 默认 CPU/离线。

# %% [markdown]
# # 推理服务：分页缓存、推测解码与服务指标
#
# PageTable 是内存管理模拟器；它不存真实 K/V，也不是 PagedAttention kernel。

# %%
from llm_from_scratch.inference import PageTable, RequestTrace, summarize_serving

table = PageTable(page_size=4, free_pages=list(range(8)))
table.append_tokens("a", 5)
table.share_prefix("a", "b")
table.append_tokens("b", 1)
assert table.sequence_pages["a"][-1] != table.sequence_pages["b"][-1]
traces = [
    RequestTrace(0.0, 0.2, 1.0, output_tokens=5, prompt_tokens=8),
    RequestTrace(0.1, 0.5, 1.4, output_tokens=4, prompt_tokens=3),
]
metrics = summarize_serving(traces, ttft_slo=0.5, tpot_slo=0.3)
print(
    {
        "fragmentation": table.internal_fragmentation_tokens,
        "ttft_p95": metrics.ttft.p95,
        "token_throughput": metrics.output_token_throughput,
    }
)

# %% [markdown]
# 完成 20，并区分 greedy 教学基线与具有接受率、残差分布和 bonus token 的随机正确版本。

# %% [markdown] llm_course_enrichment=true
# ## 学习目标：从单次生成走向服务系统
#
# 离线推理包括权重量化、KV cache 管理、请求调度、decode 策略与指标。必须同时报告正确性、TTFT、TPOT、吞吐、goodput、并发和硬件。

# %% llm_course_enrichment=true
import torch

from llm_from_scratch.inference import symmetric_dequantize, symmetric_quantize

weights = torch.linspace(-1, 1, 101)
for bits in (4, 8):
    q, scale = symmetric_quantize(weights, bits=bits)
    restored = symmetric_dequantize(q, scale)
    print(bits, "bit max error=", float((weights - restored).abs().max()))

# %% [markdown] llm_course_enrichment=true
# ## 1. 量化误差按粒度与异常值分析
#
# 本例是 per-tensor 对称伪量化，不代表 GPTQ/AWQ/KIVI 的校准、group-wise scale、outlier 处理或压缩 kernel。比较前固定模型、数据、上下文和解码。

# %% llm_course_enrichment=true
from llm_from_scratch.inference import PageTable

table = PageTable(page_size=4, free_pages=list(range(10)))
table.append_tokens("prompt", 6)
table.share_prefix("prompt", "branch")
shared = list(table.sequence_pages["branch"])
table.append_tokens("branch", 1)
assert table.sequence_pages["branch"][-1] != table.sequence_pages["prompt"][-1]
print(
    {
        "shared_before": shared,
        "after_COW": table.sequence_pages,
        "refcounts": table.page_refcounts,
        "fragmentation": table.internal_fragmentation_tokens,
    }
)

# %% [markdown] llm_course_enrichment=true
# ## 2. PagedAttention 是 kernel；PageTable 是内存语义
#
# PageTable 只模拟逻辑块、refcount、前缀共享、COW 和碎片，不存真实 K/V。连续批处理在每步接纳/移除请求，与静态 batch 不同。

# %% llm_course_enrichment=true
remaining = {"A": 2, "B": 4}
timeline = []
while remaining:
    timeline.append(tuple(sorted(remaining)))
    remaining = {n: left - 1 for n, left in remaining.items() if left - 1 > 0}
print("active batches:", timeline)
assert timeline == [("A", "B"), ("A", "B"), ("B",), ("B",)]

# %% [markdown] llm_course_enrichment=true
# ## 3. 随机推测解码必须保持目标分布
#
# 拒绝 draft token 后从 `(p-q)₊` 残差采样；全部接受还要 bonus token。接受率决定目标调用摊销，但 draft 成本和验证批大小同样重要。

# %% llm_course_enrichment=true
from llm_from_scratch.inference import stochastic_speculative_decode

g = torch.Generator().manual_seed(7)


def draft(_):
    return torch.tensor([0.6, 0.3, 0.1])


def target(_p, c):
    return [torch.tensor([0.5, 0.4, 0.1]) for _ in range(len(c) + 1)]


out, stats = stochastic_speculative_decode(draft, target, [0], 5, draft_steps=2, generator=g)
assert len(out) == 6 and stats.target_calls > 0
print(out, stats)

# %% [markdown] llm_course_enrichment=true
# ## 4. 指标口径
#
# TTFT 含排队与 prefill；TPOT 对第 2 个及以后输出定义；吞吐按观测窗口；goodput 只计满足 SLO 的请求。尾延迟用 p95/p99，不能用均值替代。

# %% [markdown] llm_course_enrichment=true
# ## 练习、来源与验收
#
# 完成 starter 20。来源：[PagedAttention](https://arxiv.org/abs/2309.06180)、[Speculative Decoding](https://arxiv.org/abs/2211.17192)、[Orca](https://www.usenix.org/conference/osdi22/presentation/yu)、[DistServe](https://arxiv.org/abs/2401.09670)、[KIVI](https://arxiv.org/abs/2402.02750)。同时给出正确性与 TTFT/TPOT/吞吐。
