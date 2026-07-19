# 阶段七：MoE——稀疏激活不自动等于更快

## 逐周讲义导航

> 本页是阶段知识地图，用于预习和复盘；完整推导、代码、反例、实验与验收请进入下面的逐周讲义。

- [第 35 周：条件计算与 Top-k 路由](../weeks/35_topk_routing.md)
- [第 36 周：容量、token dropping 与负载均衡](../weeks/36_capacity_and_balance.md)
- [第 37 周：Router z-loss、数值精度与稳定性](../weeks/37_router_stability.md)
- [第 38 周：共享专家、细粒度专家与 upcycling](../weeks/38_shared_experts_upcycling.md)
- [第 39 周：专家并行、DeepSeekMoE 与现代变体](../weeks/39_expert_parallel.md)

Mixture of Experts 用条件计算扩大总参数容量：模型存有多个 FFN 专家，但每个 token 只激活少数
专家。学习本章时始终分开四件事：总参数、每 token 活跃参数、路由质量、真实系统吞吐。

## 五周学习顺序

1. Dense FFN、条件计算与 token-choice Top-k。
2. 容量、dropping、负载指标和辅助损失。
3. Router z-loss、FP32 计算与梯度路径。
4. Shared/fine-grained experts 与 sparse upcycling。
5. Expert Parallel、all-to-all 和现代模型比较。

先操作[MoE 路由交互图](../interactive/core-concepts.html#moe)，再看
[MoE 演化时间轴](../interactive/architecture-evolution.html#timeline)。图用于建立直觉；精确语义以
本章列出的路由契约和测试为准。

## 1. 参数账本

无 bias 的 SwiGLU 专家有三块矩阵：

```text
gate: d_model * hidden_dim
up:   d_model * hidden_dim
down: hidden_dim * d_model
```

课程函数 `moe_parameter_accounting` 分别报告 router、单专家、总参数和每 token 活跃参数。
必须同时写出：专家数、Top-k、shared expert 数和 expert hidden width。只说“671B”或
“37B active”都不足以推断内存、FLOPs 或延迟；router 和 attention 等非专家参数也始终活跃。

## 2. 路由是一个明确契约，不是一条通用公式

课程 `TopKMoE` 是 token-choice 教学实现：

```text
router_logits: [tokens, experts]
router_scores: affinity(router_logits)
top_indices:   [tokens, k]
top_weights:   [tokens, k]
output:        sum(top_weights * expert_output)
```

构造参数显式区分：

- `routing_mode="softmax"`：score 在所有专家间归一。
- `routing_mode="sigmoid"`：每专家独立 affinity。
- `normalize_topk=True/False`：选择后是否再次归一。

这些开关不是为了宣称生产系统只分两类，而是防止把某一种实现背成 MoE 定义。例如 Llama 4 的
公开实现使用 Top-k 后 sigmoid score，并同时运行 shared expert；DeepSeek-V3 使用 sigmoid
affinity 和用于选择的 expert bias；Switch 则以 Top-1 路由简化系统。

原始与代表性资料：

- [Sparsely-Gated MoE](https://arxiv.org/abs/1701.06538)
- [GShard](https://arxiv.org/abs/2006.16668)
- [Switch Transformer](https://arxiv.org/abs/2101.03961)
- [Llama 4 官方说明](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)
- [DeepSeek-V3](https://arxiv.org/abs/2412.19437)

### Top-1 梯度反例

离散 `topk` 对未选择专家没有主任务梯度。还有一个更容易遗漏的情况：若 `top_k=1` 且选择后
把唯一权重归一为 1，输出不再依赖被选 score，因此主任务给 router 的梯度为零。

```text
w = score / score = 1
output = expert(x)
d output / d score = 0
```

`tests/test_moe.py` 明确验证：normalized Top-1 的主任务 router gradient 为零；加入 balance/z-loss
后 router 才获得梯度。Top-2 中被选专家的相对混合权重通常能传递主任务梯度。不要把这条教学
结论外推到所有 Top-1 架构，因为有的实现保留未归一 gate score 或采用其他训练信号。

## 3. 容量、assignment 与 token survival

每批总 assignment 数是：

```text
assignments = tokens * top_k
capacity_per_expert = ceil(capacity_factor * assignments / experts)
```

容量是硬件友好的批次预算，不是 MoE 数学定义。最忙专家超过容量时必须明确选择策略：

- 丢弃超额 assignment。
- 送到备选专家。
- padding 到固定 capacity。
- 使用 dropless block-sparse/grouped GEMM。

课程实现按 gate weight 保留每专家最高权重 assignment，并返回：

- `selected_load`：容量前的硬选择次数。
- `accepted_load`：实际执行的 assignment。
- `dropped` 与 `dropped_assignments`。
- `accepted_per_token`：每个 token 还剩几条路径。

应检查守恒：

```text
sum(selected_load) = accepted_assignments + dropped_assignments
```

当前实现不在 dropping 后重新归一剩余路径，所以只保留一条路径时 FFN 分支幅值可能下降；若所有
路径被丢弃，routed FFN 分支为零。完整 Transformer 通常还有 residual，但这不等于 token 没受
影响。论文或代码说“token dropping”时，要确认它实际丢的是 token、assignment 还是 padding。

[MegaBlocks](https://arxiv.org/abs/2211.15841) 展示了 dropless block-sparse 路径为什么既是算法
语义问题，也是 GPU kernel 问题。

## 4. Balance loss、z-loss 与负载指标

Switch 风格 balance loss 可写成：

```text
L_balance = E * sum(mean_router_probability[e] * selected_fraction[e])
```

z-loss 控制 router logits 的 `logsumexp` 尺度：

```text
L_z = mean(logsumexp(router_logits)^2)
```

两者解决不同问题：balance loss 影响选择分布，z-loss 改善极端 logits 的数值稳定性。z-loss 的
系统研究见 [ST-MoE](https://arxiv.org/abs/2202.08906)。

辅助 loss 不能代替真实负载指标。一个重要反例是：router 概率完全均匀时，确定性 tie-breaking
仍可能把所有 token 送到同一专家；balance loss 的数值仍为 1，但梯度和硬负载并不表示“系统已
均衡”。训练日志还应记录最大/平均负载、coefficient of variation、零负载专家数、dropping 和
token survival。

辅助 loss 太强会干扰语言建模目标。另一条路线是在 Top-k 选择前给专家 score 加动态 bias，按
近期负载调整 bias 而不把均衡梯度加入主损失。应阅读
[Auxiliary-Loss-Free Load Balancing](https://arxiv.org/abs/2408.15664)。“aux-free”不是删除
负载控制，也不保证专家会形成语义专门化。

## 5. Router 精度

低精度下极小 score 差异可能改变离散 Top-k，router 因此常采用更高精度计算。课程实现用
FP32 输入和 FP32 router weight 视图计算 logits，即使外部对模块调用 `.to(bfloat16)`，
`router_logits` 仍是 FP32；梯度再回传到参数原 dtype。

这只保证教学 matmul 的 dtype，不代表完整模型都应使用 FP32。真实系统还要检查：

- logits、score、Top-k 和归一化各自的 dtype。
- rollout 引擎与训练引擎是否产生相同路由。
- 跨设备归约和 expert bias 的同步。
- NaN/Inf、tie 和确定性要求。

## 6. Shared、fine-grained experts 与 upcycling

Shared expert 对所有 token 开启，适合承载公共计算；routed experts 提供稀疏容量。Fine-grained
experts 将大 FFN 拆成更多小专家，使相似活跃宽度下有更多组合，但会增加路由、参数索引和通信
复杂度。DeepSeekMoE 对两者进行了系统组合，见
[DeepSeekMoE](https://arxiv.org/abs/2401.06066)。

不要把 shared expert 解释为已经证明的“公共知识模块”，也不要把 routed expert 的 token 统计
直接命名成人类学科。路由可能主要由 token identity、位置或数据分布驱动。完全开放的训练与路由
分析可参考 [OLMoE](https://arxiv.org/abs/2409.02060)。

Sparse upcycling 将一个 Dense FFN 复制到多个专家后继续稀疏训练。课程 `upcycle_expert` 在有
一个 shared expert 时，把 shared/routed 两路输出投影各缩放 0.5，使完全相同专家在初始化时与
Dense 输出相等。加入噪声可打破对称，但也会破坏严格等价；应先做零噪声完整 MoE 输出测试，再
比较不同噪声。原始工作见
[Sparse Upcycling](https://arxiv.org/abs/2212.05055)。

## 7. Expert Parallel 与通信

专家跨设备放置时，一次前向通常包含：

```text
route → permute/pack → all-to-all dispatch → grouped expert GEMM
      → all-to-all combine → unpermute/weighted sum
```

课程 `expert_parallel_communication_ledger` 估算激活 dispatch+combine：

```text
one_way_bytes = tokens * top_k * d_model * element_bytes
total_bytes   = 2 * one_way_bytes
```

`remote_bytes_uniform_assumption` 假设专家均匀放置和路由，仅用于纸面预算。它没有包含索引、
padding、协议、网络拓扑和通信计算重叠，不能替代 benchmark。最忙 expert/rank 决定批次完成时间，
所以相同 FLOPs 的两次 MoE 前向可能有不同墙钟时间。

生产系统还会组合 data/tensor/pipeline/expert parallel，并使用 topology-aware placement、grouped
GEMM 和通信重叠。可查看 [DeepSpeed-MoE](https://arxiv.org/abs/2201.05596) 与
[DeepEP 官方实现](https://github.com/deepseek-ai/DeepEP)。算法稀疏只说明少算了部分专家；是否
更快必须分别测 prefill/decode、batch size、吞吐、延迟、显存和网络字节。

## 8. 当前公开架构如何比较

| 架构 | 路由/专家重点 | 阅读时不要混淆 |
|---|---|---|
| Mixtral 8×7B | 8 routed、Top-2 | 总参数与活跃参数 |
| DeepSeekMoE/V2/V3 | fine-grained + shared；V3 aux-loss-free bias | 单组件与整套系统贡献 |
| Llama 4 | shared + 一个 routed；Dense/MoE 层交错 | 官方说明不等于充分消融 |
| Qwen3 MoE | 纯 routed MoE 型号，同时有 Dense 型号 | 不要说成 shared+routed |
| Qwen3.5-35B-A3B | 8 routed + 1 shared，配混合序列层 | MoE 与 Gated DeltaNet 共同变化 |

Qwen3 报告见 [Qwen3](https://arxiv.org/abs/2505.09388)。模型卡或技术报告可证明公开配置和作者
实验，但不能自动证明某组件是提升的唯一原因。

## 常见误区

- “8 个专家”不等于 8 倍计算，也不等于只需存 2 个专家。
- 负载均匀不等于专家有差异，专家有差异也不保证系统负载均匀。
- FP32 router 是局部稳定性策略，不会让整个模型变成 FP32。
- Dropless 不等于没有容量/通信问题，只是不通过静默丢 assignment 解决。
- 总 FLOPs 更少不保证单请求延迟更低。

## 阶段代码入口与验收

代码位于 `src/llm_from_scratch/moe.py`，测试位于 `tests/test_moe.py`。完成阶段时应能：

1. 报告总/活跃参数并写清 router、Top-k、shared expert。
2. 比较 softmax/sigmoid 和 normalize/no-normalize 契约。
3. 推导 normalized Top-1 主任务 router gradient 为零，并用测试验证辅助 loss 梯度。
4. 检查 selected/accepted/dropped 守恒和每 token survival。
5. 区分 balance loss、z-loss、aux-loss-free bias 和真实负载指标。
6. 验证 FP32 router logits、shared upcycling 等价性与噪声边界。
7. 估算 dispatch/combine 字节，并说明为什么它不是性能结论。
