# 第 38 周：共享专家、细粒度专家与 Sparse Upcycling

> 课程定位：前几周从随机初始化训练 routed experts。本周回答两个更接近现代模型的问题：
> 怎样把“所有 token 都需要的公共知识”从路由竞争中分离出来，以及怎样复用一个已经训练好的
> Dense FFN，而不是从头支付全部 MoE 预训练成本。

## 1. 学习目标

完成后应能画出 shared experts 与 routed experts 的并行数据流；能在固定总参数或固定活跃 FLOPs
下比较粗粒度和细粒度专家；能把 Dense FFN 权重复制到 E 个专家并证明初始化输出等价的条件；
能解释复制初始化为何产生对称性、路由和数据如何打破对称；能设计 upcycling 消融，公平比较
继续训练 Dense、从零训练 MoE 与从 Dense checkpoint 稀疏升级三条路线。

## 2. 前置知识

需要掌握 Transformer FFN、Top-k gate 和活跃参数核算。先完成
[MoE Notebook](../../labs/09_moe.ipynb) 的 Top-k 部分并阅读
[starter 18](../../labs/starter/18_moe_systems.py)。本周不要求训练大模型，使用相同的小型
MLP 检查初始化输出与梯度。理解 shared expert 时要记住：它通常对每个 token 都执行，所以属于
活跃计算，不应藏在“稀疏专家只激活 k 个”的口径之外。

## 3. 核心直觉

如果每个 token 都要通过路由竞争来获得基本语言能力，routed experts 可能重复学习公共知识，
专家间冗余上升。Shared expert 像公共基础设施，对所有 token 常开；routed experts 再学习
更专门的变换。这个设计可能促进专化，但也增加密集 FLOPs，并不保证共享专家真的只学公共知识。

细粒度专家把一个大 FFN 拆成更多较小单元，再为每 token 选择更多小专家。假设传统方案有 E 个
宽度 M 的专家、激活 k 个；细粒度倍率 m 可以构造 mE 个宽度约 M/m 的专家并激活 mk 个，
理想活跃宽度仍约 kM，却有更多组合方式。参数、对齐、共享专家和门控开销必须一起核算，不能
只比较专家数量。

Sparse upcycling 则把已经训练的 Dense FFN 复制成多个专家，让稀疏模型从一个有用函数开始。
复制不是魔法：若所有专家完全相同，路由选择谁都得到同一函数，初始专化为零；后续不同 token
子集、优化噪声、router 和可能的扰动才逐渐打破对称。

## 4. 张量与数据契约

Dense FFN f 的输入输出都是 [N,D]。带 S 个 shared experts、E 个 routed experts、Top-k 的层可写：

[
y_i=sum_{s=1}^{S}a_s f^{shared}_s(x_i)
+sum_{ein R_i}g_{i,e} f^{routed}_e(x_i).
]

系数 a_s 可以是固定1、缩放或另一个 gate；必须在契约中说明。若 shared 输出与 routed 输出直接
相加，活跃专家数是 S+k。若为保持尺度再除以 S+1，初始化等价条件会改变。

Upcycling 时，Dense 权重 W_up、W_down 被复制到每个 routed expert。若 gate 在选中专家上和为1，
且没有 shared 分支，则

[
sum_{ein R_i}g_{i,e}f(x_i)=f(x_i),
]

所以初始 MoE 输出严格等于 Dense 输出，和 Top-k index 无关。若不重归一、专家含不同 dropout、
bias 未复制或激活宽度变化，等价不再成立。测试必须在 eval 模式、固定 dtype 下逐项核对。

## 5. 公式推导与算法机制

粗粒度专家每个约有 2DM 参数，总 routed 参数约2EDM，活跃约2kDM。细粒度 mE 个宽度 M/m
专家，总参数仍约2EDM，激活 mk 个的理想计算仍约2kDM。这只是代数公平点：真实 kernel 的矩阵
变小、专家数变多，调度和通信可能更差；更灵活组合带来的质量收益必须实验验证。

复制初始化的梯度并不完全相同。即使 expert weights 初值相同，专家 e 只接收路由给它的 token：

[

abla_{	heta_e}L=sum_{i:ein R_i}g_{i,e}
rac{partial L}{partial f_{	heta_e}(x_i)}
rac{partial f_{	heta_e}(x_i)}{partial	heta_e}.
]

只要 token 子集或 gate 不同，各专家第一步梯度就不同，对称开始破裂。若 router 把所有 token
均匀复制给所有专家且 gate 相同，对称会持续；若 Top-1 重归一，router 主任务梯度可能为零，
更需要辅助信号或初始化设计。

Shared expert 也影响梯度竞争。它为所有 token 提供一条稳定路径，可能减少 routed 分支承担公共
知识的压力；也可能主导输出，让 routed experts 梯度偏小。应记录 shared/routed 输出 RMS 和
梯度范数，而不是只看负载。

## 6. 手算与数值示例

设 Dense 函数 f(x)=2x，复制为四个 routed experts，Top-2 gate=[0.75,0.25]。无论选专家
[0,3] 还是 [1,2]，输出都是0.75×2x+0.25×2x=2x。若 gate 来自全局 softmax但不重归一，
选中和只有0.8，输出变为1.6x，初始化即产生20%尺度变化。

再加入 shared expert h(x)=x，若直接相加，层输出变成3x；若目标是保持原 Dense 2x，必须明确
怎样分配或缩放 copied weights，例如让 routed 与 shared 分支的初始和等于 f，而不是事后看到
loss spike 才调 learning rate。这个微型例子应在任何 upcycling 实现前通过。

## 7. 最小代码实现

~~~python
import copy
import torch
from torch import nn

torch.manual_seed(0)
dense = nn.Sequential(nn.Linear(4, 8), nn.GELU(), nn.Linear(8, 4))
experts = nn.ModuleList([copy.deepcopy(dense) for _ in range(4)])
x = torch.randn(3, 4)
indices = torch.tensor([[0, 1], [1, 3], [2, 0]])
gates = torch.tensor([[0.7, 0.3], [0.2, 0.8], [0.5, 0.5]])

moe = torch.zeros_like(x)
for token in range(x.shape[0]):
    for slot in range(2):
        e = int(indices[token, slot])
        moe[token] += gates[token, slot] * experts[e](x[token])
reference = dense(x)
torch.testing.assert_close(moe, reference, atol=1e-6, rtol=1e-6)
~~~

这段代码证明复制+重归一 gate 的函数等价，不包含容量、shared expert 或分布式状态迁移。大型
checkpoint upcycling 还要处理 optimizer state、参数命名、sharding、随机数和恢复验证。

## 8. 反例、常见误区与调试

误区一是认为复制后每个专家已经“继承不同知识”；它们继承的是同一 Dense 函数。误区二是用
更多细粒度专家却不缩小宽度，然后声称相同参数/FLOPs。误区三是只比较 upcycled MoE 与从零
MoE，不与“继续训练原 Dense 相同追加 token”比较，无法隔离额外训练预算。误区四是把 shared
expert 写进结构图，却从活跃参数和延迟表里删除。

调试 upcycling 先做 state_dict key/shape 清单，再逐专家比较权重 checksum，然后在 dropout
关闭时比较 Dense/MoE 输出，最后才跑一步优化并观察专家间权重距离是否从零增加。若初始输出
不等价，检查 gate 和、bias、激活、shared 分支和归一化；不要用更小学习率掩盖结构错误。

## 9. 主流工作与实现边界

Sparse Upcycling 系统研究从 Dense checkpoint 初始化稀疏模型，在追加训练预算下比较收益。
DeepSeekMoE 提出细粒度专家分割和 shared expert isolation，并用不同规模实验报告专化与效率。
DeepSeek-V2/V3 延续该专家结构，同时加入更复杂路由和系统设计。OLMoE 是从头训练的开放 MoE，
其开放数据、代码和日志为“专家何时专化”提供另一种证据，不应当被描述为 upcycling 实现。

现代工作也探索 cluster-aware 或分支合并初始化，但新预印本的收益需与成熟 sparse upcycling
分层标注。本课程实现复制等价、输出尺度和对称破缺实验，不复现大规模 checkpoint repartition
或生产 optimizer state 转换。

## 10. 实验与 Notebook 对照

实验一复制 Dense FFN 到 E 个专家，随机 Top-k 但保持 gate 和为1，验证输出等价；实验二取消
重归一，画输出 RMS 随 selected mass 变化；实验三运行一个优化 step，比较各 expert weight
distance 与接收 token 集；实验四加入 shared expert，扫描其缩放并记录 shared/routed 输出 RMS；
实验五在固定总参数和活跃宽度下比较粗/细粒度配置，只做正确性与成本账，不用 CPU 时间宣称
GPU 性能。

## 11. 验收标准

- 复制后每个 expert 参数与 Dense checksum 一致。
- Eval 模式、gate 和为1时，MoE 与 Dense 输出误差小于1e-6。
- 能手算不重归一 gate 导致的输出尺度变化。
- 参数表同时报告 routed、shared、router 的总参数和活跃计算。
- 一步训练后说明何种 token/gradient 差异打破专家对称。
- Upcycling 对照保持追加训练 token、优化器与数据一致。
- 完成 starter 18 并运行 uv run llm-course exercises check 18。

## 一手来源

- [Sparse Upcycling](https://arxiv.org/abs/2212.05055)：从 Dense checkpoint 训练稀疏模型。
- [DeepSeekMoE](https://arxiv.org/abs/2401.06066)：细粒度 routed experts 与 shared experts。
- [DeepSeek-V2](https://arxiv.org/abs/2405.04434)：DeepSeekMoE 与 MLA 的规模化模型报告。
- [OLMoE](https://arxiv.org/abs/2409.02060)：开放 MoE 训练、日志与专化分析。
- [Branch-Train-MiX](https://arxiv.org/abs/2403.07816)：从独立分支构建专家模型的另一条路线。
