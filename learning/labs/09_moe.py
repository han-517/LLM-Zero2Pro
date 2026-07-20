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
# **课程契约** · 周次 35, 36, 37, 38, 39 · 预计 110 分钟 · Starter 05, 10, 18 · 默认 CPU/离线。

# %% [markdown]
# # 02｜MoE 路由与负载
#
# MoE 的关键不是专家数量本身，而是每个 token 选择谁、是否超容量，以及最忙专家是否拖慢整批。

# %%
import torch

from llm_from_scratch.moe import TopKMoE

torch.manual_seed(9)
moe = TopKMoE(16, 32, num_experts=8, top_k=2, capacity_factor=1.0)
x = torch.randn(4, 12, 16)
output, stats = moe(x)
print("output:", output.shape)
print("capacity:", stats["capacity"].item())
print("dropped:", stats["dropped"].sum().item())
print("balance loss:", round(stats["balance_loss"].item(), 4))

# %%
selected = stats["selected_load"].cpu()
accepted = stats["accepted_load"].cpu()
indices = range(len(selected))
print({"selected": selected.tolist(), "accepted": accepted.tolist()})

# %% [markdown]
# ## 练习
#
# 把 Router 权重全部清零。概率会均匀，但离散 Top-k 负载是否均匀？解释 tie-breaking、balance loss 与 z-loss 各自解决什么。

# %% [markdown] llm_course_enrichment=true
# ## 学习目标与数据流
#
# `tokens [B,T,D] → router logits [B·T,E] → top-k assignments → capacity/drop → expert FFN → weighted combine`。MoE 增加总容量，但通信和负载不均可抵消算力收益。

# %% llm_course_enrichment=true
import torch

from llm_from_scratch.moe import TopKMoE

torch.manual_seed(0)
moe = TopKMoE(8, 16, num_experts=4, top_k=2, capacity_factor=1.0)
x = torch.randn(2, 6, 8)
y, info = moe(x)
assert y.shape == x.shape and info["top_indices"].shape == (12, 2)
print(
    {
        "selected": info["selected_load"].tolist(),
        "accepted": info["accepted_load"].tolist(),
        "dropped": int(info["dropped_assignments"]),
    }
)

# %% [markdown] llm_course_enrichment=true
# ## 1. Capacity 与 dropping
#
# 教学容量为 `ceil(capacity_factor × tokens × top_k / experts)`。selected load 是路由意图，accepted load 是真正执行的 assignment。Dropless 仍有动态调度或 padding/通信权衡。

# %% llm_course_enrichment=true
tight = TopKMoE(8, 16, num_experts=4, top_k=2, capacity_factor=0.25)
_, ti = tight(x)
assert int(ti["dropped_assignments"]) > 0
print("capacity=", int(ti["capacity"]), "dropped=", int(ti["dropped_assignments"]))

# %% [markdown] llm_course_enrichment=true
# ## 2. Top-1 归一化的 router 梯度陷阱
#
# top-k=1 且选中权重重归一为 1 时，主任务沿混合权重对 router 梯度为零；离散 top-k 索引也不可导。router 需 balance/z-loss 或其他稳定策略。

# %% llm_course_enrichment=true
top1 = TopKMoE(8, 16, num_experts=4, top_k=1, normalize_topk=True)
with torch.no_grad():
    top1.router.weight.copy_(
        torch.tensor(
            [
                [1.0] * 8,
                [0.0] * 8,
                [-1.0] * 8,
                [-2.0] * 8,
            ]
        )
    )
probe = torch.ones_like(x)
out, _ = top1(probe)
out.square().mean().backward()
main_grad = 0.0 if top1.router.weight.grad is None else float(top1.router.weight.grad.norm())

top1.zero_grad(set_to_none=True)
_, auxiliary = top1(probe)
(auxiliary["balance_loss"] + 0.01 * auxiliary["z_loss"]).backward()
auxiliary_grad = float(top1.router.weight.grad.norm())

tolerance = 1e-7
assert main_grad < tolerance
assert auxiliary_grad > tolerance
print({"main_task_router_grad": main_grad, "auxiliary_router_grad": auxiliary_grad})

# %% [markdown] llm_course_enrichment=true
# ## 3. 总参数、活跃参数与通信分开报告
#
# 共享专家总是激活；upcycling 从 dense FFN 初始化；expert parallel 需 dispatch/all-to-all/combine。参数更多不等于每 token FLOPs 同比例增加。

# %% llm_course_enrichment=true
from llm_from_scratch.moe import expert_parallel_communication_ledger, moe_parameter_accounting

p = moe_parameter_accounting(4096, 11008, 8, 2, shared_experts=1)
c = expert_parallel_communication_ledger(2048, 4096, 2, world_size=8)
print(
    {
        "total_B": round(p["total_parameters"] / 1e9, 2),
        "active_B": round(p["active_parameters_per_token"] / 1e9, 2),
        "remote_MiB": round(c["remote_bytes_uniform_assumption"] / 2**20, 1),
    }
)

# %% [markdown] llm_course_enrichment=true
# ## 练习、互动图与来源
#
# 使用 `../../interactive/moe_lab.html`，完成 starter 05、11。来源：[Switch](https://arxiv.org/abs/2101.03961)、[GShard](https://arxiv.org/abs/2006.16668)、[OLMoE](https://arxiv.org/abs/2409.02060)、[DeepSeek-V3](https://arxiv.org/abs/2412.19437)。

# %% [markdown] llm_course_enrichment=true
# ## 完成断言
#
# - [ ] 区分 selected/accepted load；[ ] 能算 capacity；[ ] 解释 top-1 router 梯度；[ ] 区分 softmax/sigmoid 与重归一；[ ] 报告参数、dropping 和通信假设。
