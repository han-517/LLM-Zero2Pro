# 阶段六：注意力与序列建模前沿

## 先画清两类成本

标准注意力训练/Prefill 会形成 `[T, T]` 分数矩阵，序列变长时计算和临时存储近似二次增长；自回归 Decode 每步只有一个新 Query，但需要读取越来越长的 KV Cache，常受内存带宽限制。

先在[注意力演化时间轴](../interactive/architecture-evolution.html#timeline)上区分 MQA/GQA、FlashAttention、
MLA、Gated DeltaNet 和 learned sparse attention 分别改变了缓存、IO、表示还是连接。

不同方法解决的不是同一个问题：FlashAttention 改执行顺序但保持数学精确；稀疏注意力减少连接；MLA 压缩 KV；线性注意力改变运算形式；Mamba/DeltaNet 用递归状态代替完整历史。

## FlashAttention：少搬数据，不少算答案

它把 Q/K/V 分块放入更快的片上存储，使用在线 Softmax 累积正确归一化结果，避免把完整分数矩阵写回高带宽内存。它是精确注意力，不是低秩近似。CPU 教学版只验证分块公式和数值一致，不声称能复现 GPU 加速。

## 稀疏注意力

滑窗只看附近 token；块稀疏按块连接；检索式稀疏注意力为每个 Query 选择少量 Key。成本下降的代价是可能断开关键长距离路径。评测必须包含模型需要跨很远位置复制或检索的信息。

## MLA

MLA 把 K/V 表示压进较小 latent，再通过投影恢复注意力所需结构。解码时缓存 latent 而不是所有 head 的完整 K/V。难点是 RoPE 的位置相关旋转不容易被静态吸收到投影中，因此现代实现保留一小部分非压缩的旋转维度。

课程实现是教学版：展示低维缓存和投影，不宣称逐项复刻 DeepSeek 内核。

## 线性注意力

去掉 Softmax 或使用可分解特征映射后，可以利用结合律：

```text
(Q K^T) V = Q (K^T V)
```

维护状态 `S_t = S_{t-1} + k_t v_t^T`，输出 `y_t = q_t^T S_t`，解码不再保存全部历史。它与 Softmax 注意力表达不同，数值相近不是默认保证。

## Mamba-2 与 Gated DeltaNet

门控状态更新可以选择遗忘旧状态。Delta rule 在写入新 `(k, v)` 前，先计算当前状态对 `k` 的预测，再只写入误差：

```text
prediction = k^T S
S <- decay * S + beta * k (v - prediction)^T
```

直觉是“不要重复抄已经知道的内容，只修正错的部分”。Gated DeltaNet 再加入可学习衰减和写入强度。

## 混合架构

线性/状态层擅长低成本持续记忆，完整注意力擅长精确检索。现代模型常按固定比例混合两者；
Qwen3.5 的公开模型卡给出 `3 × Gated DeltaNet + 1 × Gated Attention` 的重复布局。
公平实验要同时测训练 loss、长距离检索、Prefill、Decode、状态/KV 内存，不能仅凭理论复杂度宣布替代关系。

## 论文入口

- FlashAttention 系列：IO-aware 精确注意力。
- DeepSeek-V2：MLA 与 DeepSeekMoE。
- Mamba-2、Gated DeltaNet：状态空间/线性注意力的递归路径。
- Kimi Linear、DeepSeek-V3.2、Nemotron 3：混合、稀疏和系统协同设计。

## 常见误区

- 理论复杂度更低不保证短序列墙钟时间更快。
- 支持 1M context 不代表能可靠利用 1M context。
- 不能用不同硬件、不同 batch 的公开吞吐数字直接排名。
- 技术报告的结果应标注其证据等级和无法复现的训练细节。

