# 阶段六：注意力与序列建模前沿

> 本章对应第 28–34 周。先判断方法解决的是计算、HBM IO、KV Cache、连接数还是
> 固定状态，再比较质量与真实硬件性能；“理论复杂度更低”不是充分结论。

## 1. 用瓶颈而不是缩写组织知识

| 路线 | 主要改变 | 是否保持标准 Softmax attention |
|---|---|---|
| FlashAttention | 执行顺序与 HBM IO | 是 |
| MQA/GQA | KV heads 与缓存 | 是 |
| MLA | KV 参数化与 latent cache | 否/需专门推导 |
| local/sliding/sparse | Query-Key 连接图 | 否 |
| linear attention | 特征映射与结合顺序 | 否 |
| Mamba/DeltaNet | 用递归状态代替完整历史 | 否 |
| hybrid | 在精确检索与低成本状态间折中 | 取决于层类型 |

标准 attention 的 Prefill 形成 `Tq × Tk` 分数；Decode 虽只有一个新 Query，却要
读取越来越长的 KV Cache，常受内存带宽限制。先在[架构演化图](../interactive/architecture-evolution.html#timeline)
定位方法，再回到成本轴判断它究竟改了什么。

## 2. FlashAttention：精确结果，IO-aware 执行

FlashAttention 分块读取 Q/K/V，在片上存储维护在线 Softmax 的行最大值、归一化
因子和输出累积，避免把完整分数矩阵写回 HBM。它没有减少持久 KV Cache，也不是
稀疏或低秩近似。

CPU 分块 reference 的验收应包括：与 dense 输出/梯度一致；大 logits 稳定；causal
和 padding mask；非整除 block size。它只能教学在线归一化，不能用 CPU 墙钟结果
声称复现 GPU kernel 加速。FlashAttention-2/3/4 进一步针对工作划分和新硬件流水线；
其中 FlashAttention-4 是 2026 前沿预印本，应标成 frontier，不写成普遍部署标准。

## 3. 稀疏注意力与感受野

- local/sliding window：只连接附近 `w` 个 Key；
- global + local：少数全局 token 连接全序列；
- dilated/block sparse：按间隔或块建立连接；
- learned/retrieval sparse：由 indexer 为每个 Query 选少量 Key。

`sliding_window_mask(query_length, window, key_length=...)` 使用 bottom-right 对齐，
因此可直接验证 cached decode 的 `Tq<Tk`。单层窗口只有局部感受野，多层堆叠后
有效范围扩大；实验应画层数—可达位置图，并用远距离复制/检索任务测断路风险。

## 4. MLA：存储压缩不等于完整计算优化

MLA 把 K/V 压到较小 latent，decode 缓存 latent；位置相关 RoPE 分量通常需与可
吸收到矩阵乘法的内容分量解耦。投影吸收会重写 Query/输出路径，不能用“先恢复
全部历史 K/V”代替。

课程类名 `LatentCacheMLABaseline` 明确表示：它每步对全部历史 latent 执行
`k_up/v_up`，只演示存储压缩，不演示生产 MLA decode 计算路径。旧名 `SimpleMLA`
保留为兼容别名。`mla_cache_cost()` 同时报告 dense/latent bytes 与每步历史重建
MAC，防止只展示漂亮的压缩比。真正 absorbed MLA 需另行推导和 kernel 验证。

## 5. 线性注意力：parallel 与 recurrent 必须相等

课程使用正特征映射 `phi(x)=ELU(x)+1`。正确的因果归一化形式是：

\[
S_t=S_{t-1}+\phi(k_t)v_t^\top,\quad
z_t=z_{t-1}+\phi(k_t)
\]

\[
y_t=\frac{\phi(q_t)^\top S_t}
{\phi(q_t)^\top z_t+\epsilon}
\]

`causal_linear_attention()` 是逐 token recurrent reference；
`causal_linear_attention_parallel()` 用 outer product 的 prefix cumulative sum。
二者输出与梯度应在容差内一致。它们不是 Softmax attention 的恒等改写，不能只写
`Q(K^T V)` 而省略特征映射、因果 prefix 和归一化分母。

## 6. Mamba-2、Gated DeltaNet 与混合模型

Mamba-2/SSD 从 structured state space 与半可分矩阵角度连接 recurrent 与并行训练。
Delta rule 在写入新 `(k,v)` 前先读取预测，再写入误差：

\[
\hat v=k^\top S,\qquad
S\leftarrow \alpha S+\beta k(v-\hat v)^\top
\]

课程 `gated_delta_rule()` 只公开清晰状态转移，不是融合训练 kernel。测试应覆盖
`beta=0`、decay、覆盖冲突、初始状态以及分段续算等价。

状态层善于低成本持续记忆，完整注意力善于精确检索，所以公开模型常混合二者。
Qwen3.5 官方模型卡给出重复的 `3 × Gated DeltaNet + 1 × Gated Attention` 布局；
其 attention 还包含 output gating 与 partial RoPE，不能简写成普通 GQA。Kimi Linear
使用 KDA 与 MLA 的混合。Mamba-3、Gated DeltaNet-2 属 2026 新近前沿，证据等级应
与成熟基线分开。

## 7. 实验矩阵与证据规范

统一模型宽度、训练 token、数据、硬件和 dtype，至少报告：验证 loss、长距离检索、
Prefill、Decode、峰值临时内存、持久 KV/state、参数与激活 FLOPs。每项公开吞吐
都注明硬件、batch、prompt/output 分布和 kernel 版本。

建议依次完成：窗口 mask；linear parallel/recurrent oracle；delta 状态续算；MLA
缓存与重建成本；最后才比较墙钟。技术报告中的厂商数字应标明“作者报告”，不能
跨硬件直接排名。

## 8. 实现边界与常见误区

- 理论 `O(T)` 不保证短序列更快，kernel launch 和并行度可能主导。
- 支持 1M context 不代表能可靠检索 1M 距离信息。
- FlashAttention 与 GQA/MLA 可以组合，因为它们解决不同成本。
- 教学 MLA baseline、delta recurrence、CPU 分块公式均不是生产融合 kernel。
- “状态大小固定”不等于记忆无损；必须做内容冲突和长程检索评测。

## 一手来源

- [FlashAttention](https://arxiv.org/abs/2205.14135)、[FlashAttention-2](https://arxiv.org/abs/2307.08691)、[FlashAttention-3](https://arxiv.org/abs/2407.08608)、[FlashAttention-4](https://arxiv.org/abs/2603.05451)
- [Longformer](https://arxiv.org/abs/2004.05150)
- [DeepSeek-V2 / MLA](https://arxiv.org/abs/2405.04434)
- [Mamba-2 / SSD](https://arxiv.org/abs/2405.21060)
- [Gated DeltaNet](https://arxiv.org/abs/2412.06464)
- [Kimi Linear / KDA](https://arxiv.org/abs/2510.26692)
- [Mamba-3](https://arxiv.org/abs/2603.15569)、[Gated DeltaNet-2](https://arxiv.org/abs/2605.22791)
- [DeepSeek-V3.2 / DSA](https://arxiv.org/abs/2512.02556)
- [Qwen3.5 官方模型卡](https://huggingface.co/Qwen/Qwen3.5-35B-A3B)

