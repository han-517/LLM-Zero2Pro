# 第 4 周：概率、Softmax、交叉熵与 PyTorch

## 课程定位

语言模型的前向输出不是“下一个词”，而是词表上每个候选的未归一化分数 logits；训练则把正确 token 的负对数概率压低。本周从有限离散分布推导稳定 Softmax 与交叉熵，并完成最小训练循环。后续所有 next-token loss、注意力权重、采样温度和困惑度都建立在这里；若把 logits、概率与对数概率混用，代码往往仍能运行，但优化目标已经变了。

## 学习目标

学习者应能检查概率分布的非负与归一化条件；从 logits 推导 Softmax；解释平移不变性和减最大值的数值稳定性；证明单样本交叉熵等于正确类别负对数概率；推导 `∂L/∂z=p-y`；用 PyTorch 写一个无多余 Softmax 的线性分类训练循环，并区分训练态、评估态和无梯度上下文。

## 前置

需要第 2 周矩阵乘法与第 3 周链式法则。约定一个 batch 的输入 `X:[B,D]`、权重 `W:[D,V]`、偏置 `b:[V]`，logits `Z=XW+b:[B,V]`，整数标签 `targets:[B]` 且范围 `[0,V)`。分类维固定为最后一维 V；batch reduction 必须明确 `mean` 或 `sum`。

## 自洽直觉

logit 是相对偏好，不受必须为正或总和为 1 的限制。Softmax 先指数放大差距，再归一化成概率；给所有 logits 同加常数不会改变相对差距，因此概率不变。交叉熵只问模型给真实类别多少概率：给得越接近 1，损失越接近 0；给得越接近 0，负对数惩罚急剧增大。它不仅奖励“猜对”，还衡量置信度，因而比 0/1 正确率提供更平滑的训练信号。

## 张量/数据契约

`logits` 必须是有限浮点 `[B,V]`，不需要提前归一化；`targets` 必须是 `torch.long [B]`，每项是类别索引而非 one-hot。`F.cross_entropy(logits, targets)` 内部组合 log-softmax 与 NLL，默认沿类别维并对 batch 取均值。若语言模型 logits 为 `[B,T,V]`，通常 reshape 为 `[B*T,V]`，目标 reshape 为 `[B*T]`；被 padding 的位置需通过 `ignore_index` 或显式有效 mask 排除，不能把 padding 当普通词训练。

## 公式推导与机制

对类别 `i`，`p_i=exp(z_i)/Σ_j exp(z_j)`。因为任意常数 c 满足 `softmax(z+c)=softmax(z)`，取 `m=max_j z_j` 得

`p_i=exp(z_i-m)/Σ_j exp(z_j-m)`，所有指数输入都不大于 0，避免正向溢出。真实类别 k 的交叉熵

`L=-log p_k = -z_k + log Σ_j exp(z_j)`。

对任意 logit `z_i` 求导：`∂L/∂z_i=p_i-1[i=k]`。这说明正确类梯度为负，梯度下降会提高其 logit；其他类梯度为正，会降低它们。所有类别梯度之和为 0，对应 logits 平移不变性。`logsumexp` 是稳定计算核心，实际库会融合运算，避免先产生接近 0 的概率再取 log。

## 手算/数值例

取 logits `[2,1,0]`，减最大值后为 `[0,-1,-2]`，指数约 `[1,0.3679,0.1353]`，总和 `1.5032`，概率约 `[0.6652,0.2447,0.0900]`。若真实类为第 0 类，损失 `-log(0.6652)≈0.4076`，梯度约 `[-0.3348,0.2447,0.0900]`。把正确类 logit 从 2 增至 3，概率增大、损失下降。若直接算 `exp([1000,999,998])`，float 会溢出；减去 1000 后仍得到与前例相同概率，这就是稳定化而非近似。

## 最小可运行代码

```python
import torch
from torch.nn import functional as F

g = torch.Generator().manual_seed(2026)
x = torch.randn(8, 3, generator=g)
targets = (x[:, 0] + x[:, 1] > 0).long()
model = torch.nn.Linear(3, 2)
optimizer = torch.optim.SGD(model.parameters(), lr=0.2)

for _ in range(80):
    logits = model(x)                    # 原始分数，不先 softmax
    loss = F.cross_entropy(logits, targets)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

with torch.no_grad():
    probs = model(x).softmax(dim=-1)
assert torch.allclose(probs.sum(-1), torch.ones(8))
print(float(loss.detach()), probs.argmax(-1).tolist())
```

## 反例与调试

反例一是在 `cross_entropy` 前调用 softmax；函数会把概率再当 logits 做 log-softmax，目标与梯度都错误。反例二用 `exp(logits)/exp(logits).sum()`，大 logit 产生 inf/inf→NaN。反例三在 `[B,T,V]` 上 `softmax(dim=1)`，让时间位置而非词表竞争。反例四把 one-hot float 误当类别索引，或标签越界。反例五训练循环忘记 `zero_grad`，梯度跨 step 累加。调试时先断言 `torch.isfinite(logits).all()`、目标值域、最后一维 V，再用手算三分类样例对比 `torch.logsumexp`。

类别严重不平衡时准确率可能高而 NLL 很差；label smoothing 会改变目标分布，不能再把损失简单解释为单一正确类 NLL。评估生成模型时还要明确 token 平均方式和 padding mask，否则不同长度样本不可比。

## 主流工作与边界

大模型预训练仍以 token-level cross-entropy 为主；输出层可与输入 embedding 权重共享，超大词表下也可能使用分片或近似，但数学目标不变。低精度训练通常让关键归约保留较高精度。Softmax 不是唯一归一化选择，稀疏概率变换存在研究用途，却不是主流 decoder LM 的默认输出。本周不讨论校准、RLHF 奖励或序列级目标；它们都需先理解基准 NLL。

## 对应 Notebook、互动图与 starter

运行 `notebooks/core/01_shapes_and_autograd.ipynb` 的稳定 Softmax 与线性分类器单元，打开 `docs/interactive/core-concepts.html` 查看概率变化。补完 `exercises/starter/01_stable_softmax.py`，使用 `uv run llm-course exercises check 01` 核查。阶段概览是 `docs/stages/01_foundations.md`。

## 实验

实验一手算 `[2,1,0]` 的概率、损失和梯度并与 PyTorch 对照。实验二把所有 logits 加 1000，比较朴素实现、减最大值实现与 `F.cross_entropy`。实验三训练线性分类器，记录 loss、准确率和梯度范数；故意加入“双 Softmax”后比较收敛。实验四把 reduction 从 mean 换成 sum，验证梯度范数约放大 B 倍，并说明学习率为何需要相应调整。

## 验收 rubric

合格：starter 与线性分类器运行，概率归一且损失下降。良好：能从 NLL 推导 logsumexp 形式与 `p-y` 梯度，解释为什么不在 cross-entropy 前 softmax。优秀：能构造溢出、错误轴、错误 reduction 和 padding 泄漏测试，并给出容差合理的数值核验。只背 API、不能解释 logits 与概率区别者不通过。

## 一手来源

- PyTorch 官方 `cross_entropy` API：https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html
- PyTorch 官方 `logsumexp` API 与稳定计算说明：https://docs.pytorch.org/docs/stable/generated/torch.logsumexp.html
- PyTorch 官方分类训练教程：https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
- Bengio 等神经概率语言模型原论文：https://www.jmlr.org/papers/v3/bengio03a.html
