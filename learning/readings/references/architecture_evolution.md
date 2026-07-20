# RoPE、注意力与 MoE：演化、主流架构和阅读地图

> 核验日期：2026-07-19。“主流”在这里指已出现在多个公开模型或主流推理栈中的架构族，不代表它在所有任务、长度和硬件上都最好。

先打开[交互式架构演化图](../interactive/architecture-evolution.html)。点击时间轴节点可查看“解决了什么、保留了什么、付出了什么”，再切换当前模型观察三条技术线如何组合。

## 收录结论

原教程已经收录基础实现和大部分关键论文，但缺少一张把三条演化线串起来的地图，也缺少 Position Interpolation、YaRN、LongRoPE、Llama 4、Qwen3/Qwen3.5 等近年公开模型的结构化入口。本次补充后：

| 主题 | 已有内容 | 本次补充 |
|---|---|---|
| RoPE | 原始 RoPE、实现、范数测试、MLA 中的解耦 RoPE | Position Interpolation、YaRN、LongRoPE、iRoPE、partial RoPE 与 2026 年局限性研究 |
| 注意力 | MHA、MQA、GQA、FlashAttention、MLA、线性/稀疏注意力、Gated DeltaNet | 对齐时间轴，并加入 DeepSeek-V3.2 DSA 与 Qwen3.5 混合层的当前落点 |
| MoE | 稀疏门控、GShard、Switch、Mixtral、DeepSeekMoE、upcycling、专家并行 | 对齐 shared/routed、纯 routed、aux-loss-free 和混合线性注意力 MoE 的演化 |
| 当前模型 | DeepSeek-V3/V3.2、Kimi Linear、Nemotron 3 | 新增 Llama 4、Qwen3、Qwen3.5 的官方资料入口和架构对比 |

论文条目位于 [learning/readings/research/papers/catalog.yaml](../research/papers/catalog.yaml)，图中的 paper id 可直接用于论文阅读工作流。

## 位置编码与 RoPE 演化

| 年份 | 工作 | 核心变化 | 不应误读为 |
|---|---|---|---|
| 2017 | Transformer sinusoidal position | 把绝对位置加到 token 表示 | 已经解决任意长度外推 |
| 2021 | RoPE | 旋转 Q/K，使点积显式依赖相对位移 | 只要增大最大长度就能可靠外推 |
| 2023 | Position Interpolation | 把新位置压回训练过的角度范围 | 无需长上下文继续训练 |
| 2023 | YaRN | 分频率缩放并调整 attention 温度 | 对所有模型和任务都用同一 factor |
| 2024 | LongRoPE | 搜索非均匀频率缩放和渐进式扩展 | “支持 2M”就等于能利用 2M 信息 |
| 2024 | MLA decoupled RoPE | 只保留小部分带位置的 K/Q，主体 KV 可低秩缓存 | 完全移除 RoPE |
| 2025 | Llama 4 iRoPE | 交错使用无位置编码层与 RoPE 层 | NoPE 已普遍取代 RoPE |
| 2026 | partial RoPE / frontier critique | 一些混合模型只旋转部分维度；新工作研究 RoPE 的可辨识边界 | 单篇预印本已经形成共识 |

学习顺序应是：先证明二维旋转与相对位移关系，再做长度外推反例，最后比较 PI/YaRN/LongRoPE。不要从记忆各家的 `rope_theta` 开始。

## 注意力机制演化

| 年份 | 工作 | 主要解决的问题 | 主要代价或边界 |
|---|---|---|---|
| 2014 | Bahdanau attention | 固定长度编码器瓶颈 | 仍依赖 RNN 串行状态 |
| 2017 | Multi-Head Attention | 全局、并行的内容寻址 | Prefill 计算和分数矩阵随长度二次增长 |
| 2019 | MQA | 多个 Q head 共享一组 KV | KV 表达能力更受限 |
| 2022 | FlashAttention | 减少精确注意力的 HBM IO | 不改变二次注意力数学 |
| 2023 | GQA | 在 MHA 质量与 MQA 缓存间折中 | 需要选择 group 数并适配内核 |
| 2024 | MLA | 缓存低维 KV latent | 投影吸收、RoPE 和内核实现更复杂 |
| 2024–2026 | Gated DeltaNet / hybrid | 用固定状态承担大部分序列混合，保留少量完整注意力做精确检索 | 状态记忆不等价于 Softmax 检索 |
| 2025–2026 | learned sparse attention | 为每个 query 选择少量历史 token | indexer 本身也有质量和成本 |

先区分三个问题：FlashAttention 优化 IO；GQA/MLA 减少 decode cache；稀疏/线性注意力改变连接或状态。它们可以组合，不能只按一个“复杂度”列排名。

## MoE 演化

| 年份 | 工作 | 路由/专家变化 | 仍需测量 |
|---|---|---|---|
| 2017 | Sparsely-Gated MoE | 学习稀疏门控，按样本激活少数专家 | 专家坍缩、通信与稳定性 |
| 2020 | GShard | Top-2 与自动分片 | 集群 all-to-all |
| 2021 | Switch Transformer | Top-1 简化路由 | capacity 与 token dropping |
| 2022 | Expert Choice / Sparse Upcycling | 专家选择 token；Dense checkpoint 转 MoE | 在线因果推理约束；专家对称性 |
| 2024 | Mixtral | 每层 8 个 FFN、每 token 选 2 个 | 总参数与活跃参数必须分开报告 |
| 2024 | DeepSeekMoE | 细粒度 routed experts + shared experts | 路由组合、共享路径和通信 |
| 2024 | DeepSeek-V3 | 共享/路由专家并采用 auxiliary-loss-free balancing | balance 不等于 specialization |
| 2025–2026 | Llama 4 / Qwen3 / Qwen3.5 | shared+routed、纯 routed、以及与混合线性注意力组合 | 权重内存、活跃 FLOPs、最忙专家和服务拓扑 |

MoE 的核心指标至少包括：总参数、活跃参数、专家数、Top-k、共享专家数、最大专家负载、dropping 策略和跨设备通信。只给一个“B 参数”数字没有比较意义。

## 2026 年公开模型中的几种主流组合

| 架构族 | 公开代表 | 注意力 | 位置 | FFN/MoE | 适合观察的权衡 |
|---|---|---|---|---|---|
| 现代 Dense/GQA | Qwen3 dense 等 | GQA/完整 Softmax | RoPE，部署时可用 YaRN 扩展 | Dense SwiGLU | 实现和部署成熟，KV 随长度增长 |
| MLA + MoE | DeepSeek-V3/V3.2 | MLA；V3.2 叠加 DSA | 小部分解耦 RoPE | shared + fine-grained routed experts | 低 KV 与大容量，但系统复杂 |
| iRoPE + MoE | Llama 4 | 完整注意力，层间交错 NoPE/RoPE | iRoPE | shared + 1 routed expert，部分层为 Dense | 长度泛化与简单路由的组合 |
| Hybrid state + attention + MoE | Qwen3.5 | 3 个 Gated DeltaNet 层配 1 个 GQA 层 | 完整注意力层只旋转部分维度 | sparse MoE，shared + routed | decode 状态固定，但精确检索依赖少数注意力层 |
| Learned sparse attention | DeepSeek-V3.2 | DSA 在 MLA 上选择少量历史 token | 延续 MLA 的位置设计 | DeepSeekMoE | 长上下文成本下降，但 indexer 成为新组件 |

这张表不包含未公开架构的闭源模型；公开 benchmark 也不能证明某个组件单独带来全部增益。

## 建议的学习与实验顺序

1. 用 `apply_rope` 验证范数和相对位置性质，再构造训练长度外的失败样例。
2. 对 MHA、MQA、GQA、MLA 分别计算 KV Cache，不把 FlashAttention 当作第五种缓存格式。
3. 用复制/检索两类任务比较完整、滑窗和递归状态，避免只测语言建模 loss。
4. 在相同活跃 FLOPs 下比较 Dense 与 MoE，同时记录最忙专家和通信。
5. 最后选择一个公开模型，把它拆成“token mixing + position + FFN/routing + systems”四层，而不是背模型名。

## 原始资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [RoFormer / RoPE](https://arxiv.org/abs/2104.09864)
- [Position Interpolation](https://arxiv.org/abs/2306.15595)、[YaRN](https://arxiv.org/abs/2309.00071)、[LongRoPE](https://arxiv.org/abs/2402.13753)
- [GQA](https://arxiv.org/abs/2305.13245)、[FlashAttention](https://arxiv.org/abs/2205.14135)
- [DeepSeek-V3](https://arxiv.org/abs/2412.19437)、[DeepSeek-V3.2](https://arxiv.org/abs/2512.02556)
- [Sparsely-Gated MoE](https://arxiv.org/abs/1701.06538)、[Switch Transformer](https://arxiv.org/abs/2101.03961)、[Mixtral](https://arxiv.org/abs/2401.04088)、[DeepSeekMoE](https://arxiv.org/abs/2401.06066)
- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)、[Qwen3.5 官方模型卡](https://huggingface.co/Qwen/Qwen3.5-35B-A3B)
- [Llama 4 官方架构说明](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)
- [RoPE long-context limitations（2026 frontier preprint）](https://arxiv.org/abs/2605.15514)
