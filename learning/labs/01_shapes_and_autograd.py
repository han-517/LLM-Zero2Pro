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
# **课程契约** · 周次 2, 3, 4 · 预计 100 分钟 · Starter 01, 11 · 默认 CPU/离线。

# %% [markdown]
# # 00｜形状、广播与自动微分
#
# 目标：为轴命名；区分矩阵乘法与广播；用分支求和和有限差分核对反向传播；最后让一个两层网络过拟合小 batch。所有单元默认在 CPU 运行。

# %% [markdown]
# ## 1. 矩阵乘法保留哪些轴？
#
# 运行前先在纸上写出 `x @ weight` 的形状，并圈出被求和的轴。

# %%
import math

import torch

torch.manual_seed(7)
batch, time, d_in, d_out = 2, 5, 3, 4
x = torch.randn(batch, time, d_in)
weight = torch.randn(d_in, d_out)
y = x @ weight
print("x:", x.shape, "weight:", weight.shape, "y:", y.shape)
assert y.shape == (batch, time, d_out)

# %% [markdown]
# ## 2. 能广播不等于语义相同
#
# `feature_bias:[D]` 会沿 batch/time 重复；`time_bias:[T,1]` 会沿 batch/feature 重复。两段代码都能运行，但它们修改的是不同轴。

# %%
feature_bias = torch.arange(d_in, dtype=x.dtype)
time_bias = torch.arange(time, dtype=x.dtype).view(time, 1)
by_feature = x + feature_bias
by_time = x + time_bias
print("feature bias:", feature_bias.shape, "->", by_feature.shape)
print("time bias:   ", time_bias.shape, "->", by_time.shape)
assert torch.allclose(by_feature[0, 0] - x[0, 0], feature_bias)
assert torch.allclose(by_time[0, :, 0] - x[0, :, 0], time_bias[:, 0])

# %% [markdown]
# ## 3. 分支求和与有限差分
#
# 对 `f(x)=tanh(x²+3x)`，`x` 通过两条路径影响结果。中心有限差分提供不依赖自动微分实现的数值 oracle。

# %%
from llm_from_scratch.autograd import Value

point, epsilon = 0.4, 1e-5


def function(value):
    return math.tanh(value * value + 3 * value)


numerical = (function(point + epsilon) - function(point - epsilon)) / (2 * epsilon)
node = Value(point, label="x")
result = (node * node + 3 * node).tanh()
result.backward()
print("finite difference:", numerical, "autograd:", node.grad)
assert math.isclose(node.grad, numerical, rel_tol=1e-7, abs_tol=1e-7)

# %% [markdown]
# 共享子表达式的梯度必须在汇合处相加。重复 `backward()` 时叶子梯度累加，但旧的中间梯度不能再次传播；`zero_grad()` 会清理整张可达图。

# %%
leaf = Value(2.0, label="leaf")
shared = leaf * leaf
output = shared + 3 * shared
output.backward()
assert leaf.grad == 16.0
output.backward()
assert leaf.grad == 32.0
output.zero_grad()
assert leaf.grad == 0.0 and output.grad == 0.0
print("通过：分支求和、重复反传与整图清零")

# %% [markdown]
# ## 4. 两层网络单 batch 过拟合
#
# 这是训练闭环的烟雾测试，不是泛化证明。先预测：去掉 `Tanh` 后，下面的 XOR 数据会发生什么？

# %%
features = torch.tensor([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
labels = torch.tensor([0, 1, 1, 0])
model = torch.nn.Sequential(
    torch.nn.Linear(2, 8),
    torch.nn.Tanh(),
    torch.nn.Linear(8, 2),
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
losses = []
for _ in range(120):
    logits = model(features)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

print("initial:", round(losses[0], 4), "final:", round(losses[-1], 4))
assert losses[-1] < losses[0] * 0.1
assert torch.equal(model(features).argmax(dim=-1), labels)
print("loss samples:", [round(value, 4) for value in losses[:: max(1, len(losses) // 6)]])

# %% [markdown]
# ## 练习与核查
#
# 1. 把有限差分点改为 `x=2`，先手算再运行。
# 2. 把 `shared + 3*shared` 改成三个不同分支，预测梯度。
# 3. 删除 MLP 的非线性并记录失败曲线。
# 4. 完成 `learning/labs/starter/01_stable_softmax.py`，运行 `uv run llm-course exercises check 01`。

# %% [markdown] llm_course_enrichment=true
# ## 5. 调试清单：先查形状，再查数值
#
# 遇到 loss 异常时依次记录：`shape → dtype → device → min/max → 是否有限 → grad norm`。广播成功只说明语法允许，不说明 batch、time、feature 的语义对齐。

# %% llm_course_enrichment=true
import torch

x = torch.tensor([1.0, 2.0], requires_grad=True)
y = x[0] * x[1] + x[0] ** 2
expected = torch.tensor([4.0, 1.0])
y.backward()
torch.testing.assert_close(x.grad, expected)
print("分支梯度 =", x.grad.tolist(), "（同一变量的贡献必须求和）")

# %% [markdown] llm_course_enrichment=true
# ## 6. 留给你的核心实现
#
# 打开 `../../learning/labs/starter/01_autograd_engine.py`，先补局部导数，再处理拓扑排序、分支求和和重复 `backward()`。运行 `uv run llm-course exercises check 01`。不要把 PyTorch 输出硬编码到 starter；公共核查会更换图结构和数值。

# %% [markdown] llm_course_enrichment=true
# ## 验收与一手来源
#
# - [ ] 能手算 `matmul` 输出轴；[ ] 能构造广播语义错误；[ ] 分支梯度与有限差分一致；[ ] 能解释叶子梯度累积与非叶节点清理。
# - 来源：[PyTorch Autograd mechanics](https://pytorch.org/docs/stable/notes/autograd.html)、[Matrix Calculus for Deep Learning](https://explained.ai/matrix-calculus/)。
