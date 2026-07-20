# 第 29 周：FlashAttention——数学不变，IO 路径重写

## 课程定位

从本周进入注意力前沿。第一课先建立最重要的分类：FlashAttention 不改变 Softmax attention 的连接或答案，而是通过 tiling 与 online softmax 减少 HBM 读写和中间矩阵存储。课程实现 CPU tiled reference 验证公式；生产价值来自 GPU SRAM、线程块、异步流水和融合 kernel，二者不可混为一谈。

## 学习目标

- 区分 FLOPs、HBM IO、片上 SRAM 与峰值中间激活。
- 推导分块 online softmax 的 running max、normalizer 和输出更新。
- 与 dense attention 比较输出、梯度、causal/padding mask 与大 logits 稳定性。
- 解释 FlashAttention 1/2/3/4 分别优化的层面和硬件证据等级。

## 前置

需要掌握稳定 Softmax、矩阵分块、causal mask 和 GPU 存储层级基本概念。应能说明减去行最大值不改变 Softmax，并理解 Python CPU 循环的速度与 GPU kernel 无直接可比性。

## 直觉

标准实现常把完整 `S=QK^T` 写到 HBM，再读回来做 Softmax 和 `PV`。长序列时，搬运这张巨大分数表比某些算术更昂贵。FlashAttention 把一块 Q 留在片上，逐块流过 K/V；每见到新块，就把旧块和新块的指数和放到统一最大值尺度下合并。像分批统计总票数：只保存每行当前最大票、归一化总和与加权结果，不保存所有选票。

## 张量/数据契约

输入 `Q,K,V:[B,H,T,D]`，输出与 dense attention 同 shape/dtype 语义。分块大小 `Br,Bc>=1`，尾块可不整除。每个 Query 行状态为 `m:[Br]`、`l:[Br]`、`o:[Br,Dv]`，初始 `m=-inf,l=0,o=0`。bool mask 的完全遮蔽行必须输出零。Dropout 在训练 kernel 中需要可重现随机策略；最小 CPU reference 可先禁用并明确边界。

## 推导与机制

旧块状态表示 `l_old=Σ exp(s_old-m_old)`、`o_old=Σ exp(s_old-m_old)v`。新块最大 `m_block`，统一最大值 `m_new=max(m_old,m_block)`：

\[
l_{new}=e^{m_{old}-m_{new}}l_{old}+\sum_j e^{s_j-m_{new}},
\]

\[
o_{new}=e^{m_{old}-m_{new}}o_{old}+\sum_j e^{s_j-m_{new}}v_j.
\]

最终 `o/l`。这个结合规则使任意 K/V 分块顺序得到同一精确 Softmax（浮点舍入范围内）。FlashAttention-2 改善工作划分与非矩阵 FLOPs；3 针对 Hopper 异步与低精度；4 是 2026 面向 Blackwell/CuTe DSL 的前沿预印本，应独立标注。

## 数值例

一行 logits 先看到 `[1000,999]`，`m_old=1000,l_old=1+e^-1≈1.3679`。第二块 `[1002]`，`m_new=1002`，旧归一化缩放 `e^-2`，所以 `l_new=e^-2·1.3679+1≈1.1851`。若直接算 `exp(1002)` 会溢出，online 更新保持有限。对应旧输出累积也乘 `e^-2` 后加新 V。

## 最小代码

```python
def online_row(blocks):
    m = torch.tensor(float("-inf"))
    l = torch.tensor(0.0)
    acc = None
    for scores, values in blocks:
        block_m = scores.max()
        new_m = torch.maximum(m, block_m)
        old_scale = torch.exp(m - new_m) if torch.isfinite(m) else 0.0
        p = torch.exp(scores - new_m)
        acc = p @ values if acc is None else old_scale * acc + p @ values
        l = old_scale * l + p.sum()
        m = new_m
    return acc / l
```

这是单行教学伪代码，未处理 batch/head、mask、反向重计算、dropout 和 GPU 调度。生产应使用 PyTorch SDPA、FlashAttention 官方 kernel 或目标 runtime，不能用此循环做性能结论。

## 反例与调试

只更新 `l` 而忘记按新最大值缩放旧 `o` 会在块最大值变化时出错；普通随机小 logits 可能看不出，应使用 `[1000,999,1002]`。causal mask 必须按全局 Query/Key index，而非每块局部 index。完全遮蔽块的 max 为 `-inf`，需避免 `-inf-(-inf)` 产生 NaN。尾块、非连续张量和 `Tq<Tk` 都要覆盖。CPU reference 更慢是预期，不是论文失效。

## 主流工作与证据等级

FlashAttention 1 是 IO-aware 精确算法基础；2、3 有论文与官方代码/硬件实验，属于成熟高性能路线。框架 SDPA 会按硬件和输入选择后端，是官方实现证据。FlashAttention-4 在 2026 是前沿预印本与新硬件实现证据，不能写成所有设备默认。公开吞吐只在相同 GPU、dtype、shape、causal/dropout 设置下可比。

## Notebook、互动图与 starter

使用 `learning/labs/08_attention_frontiers.ipynb` 比较 dense 与 tiled 输出、梯度、理论中间字节；完成 starter `16` 的 online softmax。`learning/readings/interactive/architecture-evolution.html` 用于区分 IO、cache、稀疏和状态路线，不把 FlashAttention 放进 KV 格式下拉框。

## 实验

覆盖 `T=1,7,16,31`、多 block size、大 logits、bool/additive mask、全遮蔽行。与 dense/PyTorch SDPA 比输出和梯度误差。再计算 dense 分数矩阵字节与 tiled state 上界。若有支持 GPU，调用官方 kernel 做独立墙钟，注明版本和后端；没有则只报告公式与 CPU correctness。

## 验收 rubric

- 35%：online 输出/梯度、mask 和数值稳定 oracle 通过。
- 25%：正确推导 running max/normalizer/output 合并。
- 20%：IO/激活与持久 KV Cache 分类准确。
- 20%：明确 CPU reference、官方 GPU kernel和FA4证据等级。

## 一手来源

- [FlashAttention](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2](https://arxiv.org/abs/2307.08691)
- [FlashAttention-3](https://arxiv.org/abs/2407.08608)
- [FlashAttention-4](https://arxiv.org/abs/2603.05451)
- [FlashAttention 官方代码](https://github.com/Dao-AILab/flash-attention)
