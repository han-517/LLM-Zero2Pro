# 第 33 周：Mamba-2、SSD 与 Gated DeltaNet——选择性状态更新

## 课程定位

本周把两条选择性状态更新路线放在同一个比较框架：Mamba-2/SSD 从结构化状态空间、半可分矩阵和块算法连接递归推理与并行训练；Gated DeltaNet 把状态矩阵理解为在线快速权重，用 delta rule 只写入当前预测误差并以门控主动遗忘。两者都不是“删除 softmax 就自动得到线性注意力”，也不在所有任务上无条件优于注意力。课程分别建立 SSD 标量递归 oracle 与 gated delta 状态 oracle，再讨论论文完整层、chunkwise 算法和生产 fused kernel 的边界。

## 学习目标

完成本周后，你应能写出选择性 SSM 的 $h_t=A_th_{t-1}+B_tx_t$ 与输出读取，解释 SSD 如何把递归、矩阵结构和分块训练联系起来；能从张量契约推导 gated delta rule，说明归一化 key、写入强度 $\beta_t$ 与遗忘因子 $\alpha_t$；能用整段/分块续接测试核查两种状态更新；能辨别教学逐 token oracle、论文完整层与生产 fused kernel；能依据一手报告说明 Mamba-2 与 Gated DeltaNet 的证据范围。

## 前置

需要掌握矩阵乘法、外积、归一化、因果序列扫描，以及第 32 周线性注意力的 parallel/recurrent 双视角。Mamba-2 部分先把标量或对角 SSM 状态展开成因果矩阵，再理解半可分结构与分块；DeltaNet 部分复习“key 是地址、value 是内容、query 用于读取”。快速权重状态 $S_t\in\mathbb{R}^{d_k\times d_v}$ 容量有限，相似地址会干扰，delta rule 用预测误差进行修正。

## 直觉

选择性 SSM 把每个 token 看成一次“保留多少旧状态、写入多少新输入、从状态读出什么”的数据依赖决定。递归路径适合 decode，SSD 揭示其因果矩阵具有半可分结构，从而可用块矩阵乘法训练；这不是把递归循环简单并行化，而是利用结构重排相同变换。

朴素外积写入把 $k_tv_t^\top$ 不断加到状态中，重复遇到相同 key 时会反复累加，旧内容不容易被覆盖。delta rule 先问状态“按 $k_t$ 读出来是什么”，得到预测 $\hat v_t=k_t^\top S_{t-1}$；随后只写入目标与预测之差 $v_t-\hat v_t$。若 $\beta_t=1$ 且 $k_t$ 为单位向量，那么再次用同一 key 读取时，相关方向会被精确改写成新 value。

遗忘门 $\alpha_t\in(0,1)$ 则把状态整体或按通道衰减，为新关联释放容量。可把它类比成可学习缓存的过期策略：$\alpha_t$ 小表示强烈遗忘，$\alpha_t$ 接近 1 表示长时间保留。但这个类比不能替代公式，因为不同论文可能使用标量、逐头或更细粒度的门，也可能把衰减与写入重排以便块并行。

## 张量/数据契约

SSD 教学 oracle 使用输入 $x\in\mathbb{R}^{B\times T\times D}$、状态 $h\in\mathbb{R}^{B\times D}$，选择性系数 $A,B,C\in\mathbb{R}^{B\times T\times D}$，递归输出仍为 $[B,T,D]$。真实 Mamba-2 层含 head/state 维、离散化参数、输入依赖投影、短卷积、门控与归一化；标量逐通道版本只核查状态语义。

教学版输入采用 $q,k\in\mathbb{R}^{B\times H\times T\times D_k}$、$v\in\mathbb{R}^{B\times H\times T\times D_v}$，写入门 $\beta$ 与遗忘门 $\alpha$ 可广播为 $[B,H,T,1]$。每个 batch、head 独立维护状态 $S\in\mathbb{R}^{B\times H\times D_k\times D_v}$，输出 $y_t=q_t^\top S_t\in\mathbb{R}^{B\times H\times D_v}$。若支持流式续写，接口还要接收 initial state 并返回 final state；不能把不同序列的状态混在一起，也不能在 padding token 上无条件更新。

本文使用行向量记号，$k_t[...,D_k]$ 与 $v_t[...,D_v]$ 的外积为 $k_t[...,D_k,1]v_t[...,1,D_v]$。论文、CUDA kernel 或框架代码可能采用转置布局，核查时应比较语义和 shape，而不是机械比较公式外观。训练时常把时间轴分块并行，推理时则逐 token 递归；二者必须在容差内等价。

## 推导与机制

选择性 SSM 的最小递归为

$$h_t=A_t\odot h_{t-1}+B_t\odot x_t,\qquad y_t=C_t\odot h_t.$$

展开后，$y_t$ 对过去 $x_s$ 的系数包含 $C_t(\prod_{j=s+1}^{t}A_j)B_s$。这些乘积形成结构化因果矩阵；SSD 从半可分矩阵视角说明如何在 recurrent、convolution-like 与 blockwise 形式间转换。Mamba-2 的完整 SSD 头、状态维和块算法比逐通道式更丰富，本教程以递归式作正确性 oracle，不把 Python 循环称为 SSD 生产 kernel。

先将 key 归一化为 $\bar k_t=k_t/(\|k_t\|_2+\varepsilon)$。给定旧状态 $S_{t-1}$，旧预测为

$$
\hat v_t=\bar k_t^\top S_{t-1}.
$$

定义误差 $e_t=v_t-\hat v_t$，一种便于教学和测试的 gated delta 更新是

$$
\widetilde S_{t-1}=\alpha_tS_{t-1},\qquad
S_t=\widetilde S_{t-1}+\beta_t\bar k_te_t^\top,
\qquad y_t=q_t^\top S_t.
$$

这里“旧预测基于衰减前还是衰减后状态”是实现契约的一部分，不能含糊。本教程教学版采用先从旧状态计算预测误差，再衰减、再写入；若复现某篇论文或官方 kernel，应按其精确定义重新核对。完整模型还会包含输入投影、短卷积、门控输出、归一化、chunkwise 并行与硬件特定 kernel，本式只负责暴露 delta 状态更新，不声称等价于完整 Gated DeltaNet 层。

当 $\alpha=1,\beta=1$ 且 $\|k\|=1$ 时，有

$$
k^\top S_t=k^\top S_{t-1}+k^\top k(v-k^\top S_{t-1})=v,
$$

因此同一单位 key 的读取被校正到目标 value。若 key 不归一化，多出的 $\|k\|^2$ 会改变有效步长；若两个 key 高度相似，对一个地址的写入也会改变另一个地址的读取，这就是有限状态容量下的碰撞。

## 数值例

先看 SSM：$h_0=0$，$x=[2,4]$，$A=[0.5,0.25]$，$B=C=1$，则 $h_1=y_1=2$，$h_2=y_2=0.25\times2+4=4.5$。把序列切开并传递 $h_1$ 必须得到同一 $h_2$；丢弃状态则错误得到 4。

设单头 $D_k=2,D_v=1$，初始 $S_0=[0,0]^\top$。令 $k_1=[1,0]$、$v_1=3$、$\alpha_1=1$、$\beta_1=1$，则预测为 0，写入后 $S_1=[3,0]^\top$。第二步仍用 $k_2=[1,0]$，目标改为 5；旧预测是 3，误差是 2，更新后 $S_2=[5,0]^\top$，不是朴素累加得到 8。

若第三步 $\alpha_3=0.8$、$\beta_3=0$，状态只衰减为 $[4,0]^\top$。若改用 $k_3=[0,1]$、$v_3=7$、$\beta_3=1$，状态成为 $[4,7]^\top$。这个例子也揭示了门的分工：$\alpha$ 影响既有记忆，$\beta$ 影响当前误差写入。真实模型中的门由数据决定，不能把固定常数实验当成模型能力结论。

## 最小代码

下面两个函数刻意使用显式时间循环，分别作为 SSM 与 delta update 的小规模 oracle。它们是教学 baseline，不是吞吐优化实现；训练生产环境应使用论文对应的 chunkwise/fused kernel。

```python
import torch
import torch.nn.functional as F


def selective_ssm_scan(x, A, B, C, state=None):
    # x,A,B,C: [batch,time,dim]
    h = torch.zeros_like(x[:, 0]) if state is None else state.clone()
    ys = []
    for t in range(x.shape[1]):
        h = A[:, t] * h + B[:, t] * x[:, t]
        ys.append(C[:, t] * h)
    return torch.stack(ys, dim=1), h


def gated_delta_scan(q, k, v, beta, alpha, state=None, eps=1e-6):
    # q,k: [batch,heads,time,Dk], v: [batch,heads,time,Dv]
    batch, heads, time, key_dim = k.shape
    value_dim = v.shape[-1]
    S = (
        k.new_zeros(batch, heads, key_dim, value_dim)
        if state is None
        else state.clone()
    )
    ys = []
    k = F.normalize(k, dim=-1, eps=eps)
    for t in range(time):
        kt, vt, qt = k[:, :, t], v[:, :, t], q[:, :, t]
        old_prediction = torch.einsum("bhd,bhdv->bhv", kt, S)
        error = vt - old_prediction
        S = alpha[:, :, t].unsqueeze(-1) * S
        write = torch.einsum("bhd,bhv->bhdv", kt, error)
        S = S + beta[:, :, t].unsqueeze(-1) * write
        ys.append(torch.einsum("bhd,bhdv->bhv", qt, S))
    return torch.stack(ys, dim=2), S


x = torch.tensor([[[2.0], [4.0]]])
A = torch.tensor([[[0.5], [0.25]]])
ones = torch.ones_like(x)
full_y, _ = selective_ssm_scan(x, A, ones, ones)
first_y, state = selective_ssm_scan(x[:, :1], A[:, :1], ones[:, :1], ones[:, :1])
second_y, _ = selective_ssm_scan(
    x[:, 1:], A[:, 1:], ones[:, 1:], ones[:, 1:], state
)
torch.testing.assert_close(full_y, torch.cat((first_y, second_y), dim=1))

k = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
q = k.clone()
v = torch.tensor([[[[3.0], [5.0]]]])
gates = torch.ones(1, 1, 2, 1)
out, _ = gated_delta_scan(q, k, v, gates, gates)
torch.testing.assert_close(out[0, 0, :, 0], torch.tensor([3.0, 5.0]))
```
最基本的核查包括：全零 $\beta$ 时只有衰减；$\alpha=1$ 且相同单位 key 重写时读取等于新 value；把序列切成两块并传递第一块 final state，结果应与整段扫描一致；自动求导后输入和门的梯度应有限。若优化版只在长序列上测试，很容易把布局、边界状态或 padding 错误误判为浮点误差。

## 反例与调试

第一类错误是把朴素线性注意力的外积累加称为 delta rule，却没有减去旧预测。它无法展示“只修正误差”的关键性质。第二类错误是在公式与代码中混用衰减前、衰减后预测，使手算、递归与 chunkwise 结果不一致。第三类错误是忘记 key 归一化，导致 key 范数暗中成为学习率，长序列上状态范数易漂移。

第四类错误是把 sigmoid 输出再次当 logits，或允许 $\alpha,\beta$ 无界而没有说明参数化。第五类错误是流式推理时丢弃 final state，或者 batch 重排后仍沿用旧索引的状态。第六类错误是看到线性时间复杂度便宣称显存和速度必然更好；生产性能还取决于 chunk 大小、矩阵形状、kernel 融合、状态精度、硬件利用率和完整混合层比例。

## 主流工作与证据等级

一级证据是同行评审论文、作者论文与官方代码；二级证据是机构技术报告和官方模型卡；三级证据才是博客或二手解读。本教程用 Gated Delta Networks 原论文确立算法语义，用 Mamba-2/SSD 作为相邻状态空间路线的比较基线，用 Kimi Linear 观察 delta-style 记忆在混合架构中的扩展，并用 Qwen3.5 官方模型卡确认 Gated DeltaNet 已进入公开的混合语言模型架构。2026 年的 Gated DeltaNet-2 属于前沿预印本，应标明日期与证据状态，不能回写成所有现有模型的既定标准。

“主流”也需要分层陈述：softmax attention 仍是通用基线；Gated DeltaNet 已有公开大模型采用，但不同模型的卷积、门控、层比例、位置编码和 kernel 并不相同。教学代码只实现共同的状态更新骨架，不能用它复现官方模型数值，也不能把官方模型的整体能力归因于单一 token mixer。

## Notebook、互动图与 starter

配套 Notebook 应先显示 $S_t$ 热力图、预测误差范数和门值随时间的变化，再允许拖动 $\alpha$、$\beta$、key 相似度与序列长度。互动图应同时画出“覆盖同一 key”和“相似 key 碰撞”两条曲线，并明确标注教学更新顺序。starter 代码保留 `normalize_key`、`prediction`、`error`、`decay`、`outer_write` 与 `final_state` 空缺；核查器分别检查 shape、同 key 覆盖、分块续接、有限梯度，避免只凭最终 loss 判断实现是否正确。

## 实验

实验一对选择性 SSM 展开递归系数矩阵，并核查整段扫描与两段 state continuation；改变 $A_t$，观察记忆半衰期和梯度。实验二构造正交或近正交 key，重复改写 value，比较朴素外积、delta 与 gated delta。实验三提高 key 相似度测串扰，并对两条路线分别比较逐 token 与分块路径。实验四记录 fp32、bf16 下状态、误差和梯度范数。报告必须给出 seed、dtype、设备、shape、更新顺序和教学/生产边界。

进阶实验可把 Mamba-2 或 Gated DeltaNet 与局部 softmax attention 交替堆叠，在相同参数量和训练 token 预算下比较关联召回、困惑度、prefill、decode 和状态/KV 内存。结果只对当前实现与硬件有效，不应外推为架构普遍排名。

## 验收 rubric

及格要求是两套公式、shape、状态方向和代码一致，能解释 SSD 展开、预测误差与两个门，并通过 SSM 续接和同 key 覆盖。良好要求是核查递归/分块等价、padding、有限梯度、dtype 与 key 碰撞。优秀要求是复现论文或官方实现的一种完整定义，清楚列出与教学 baseline 的差异，完成公平混合层对比，并按证据等级区分论文、官方模型事实和本地推断。

## 一手来源

- [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464)
- [Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality](https://arxiv.org/abs/2405.21060)
- [Kimi Linear: An Expressive, Efficient Attention Architecture](https://arxiv.org/abs/2510.26692)
- [Qwen3.5 官方模型卡：混合 Gated DeltaNet 与注意力架构](https://huggingface.co/Qwen/Qwen3.5-35B-A3B)
- [Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention](https://arxiv.org/abs/2605.22791)
