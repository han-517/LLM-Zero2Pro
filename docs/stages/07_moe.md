# 阶段七：MoE——不是让所有专家同时工作

## 直觉

Dense MLP 像所有问题都交给同一个大部门。MoE 建立多个专家，但每个 token 只派给少数专家。这样总参数可以很大，而每 token 的活跃计算保持较小。

先看[MoE 演化时间轴](../interactive/architecture-evolution.html#timeline)：从稀疏门控、GShard、
Switch 和 Mixtral，走到 DeepSeekMoE、Llama 4、Qwen3/Qwen3.5 的 shared/routed 组合。

必须同时报告：总参数、活跃参数、专家数、Top-k 和共享专家数。只说“671B”或“37B active”都不完整。

## Top-k 路由

Router 把 token 表示映射为每个专家的 logits，选择最大的 `k` 个并归一化：

```text
router_logits: [tokens, experts]
topk_weight:   [tokens, k]
expert_output: [tokens, k, d_model]
output = sum(topk_weight * expert_output)
```

离散 Top-k 对未选择专家没有任务梯度，因此实践中依赖可微门控权重、扰动和辅助目标。

## 容量与负载

硬件希望专家负载接近，否则最忙专家决定整批速度。容量通常近似：

```text
capacity = ceil(capacity_factor * tokens * k / experts)
```

超出容量的 token 必须明确处理：丢弃、送到备选专家或使用无丢弃稀疏矩阵实现。教学代码返回 dropped mask，避免静默改变结果。

负载均衡损失鼓励“被选择的 token 比例”和“router 平均概率”更均匀；z-loss 控制 router logits 的整体尺度。二者解决不同问题。

在[MoE 路由交互图](../interactive/core-concepts.html#moe)中调高路由偏斜、降低容量因子，
比较 selected load、accepted load 和 dropping。随后完成
`exercises/starter/05_moe_capacity.py`，确保溢出行为永远显式可见。

## 共享和细粒度专家

共享专家对所有 token 开启，承担公共知识；路由专家学习专业行为。细粒度专家把大 FFN 切成更多小专家，在相似活跃宽度下提供更多组合，但增加通信和路由复杂度。

## Upcycling

把已有 Dense FFN 复制成多个专家，可以继承能力。若同时加入共享专家，路由路径与共享路径会相加，必须缩放输出投影以保持初始化时的 Dense 输出尺度。若所有专家完全相同且路由也对称，它们可能继续学成一样，
因此需要噪声、不同数据路由或训练动态打破对称，并用完整 MoE 输出而不是单个专家输出做等价性测试。

## 系统视角

专家分布在多设备时，token 先按路由结果 all-to-all 发送，专家计算后再送回。MoE 的 FLOPs 优势可能被通信抵消，因此“算法上稀疏”和“墙钟更快”要分别验证。

## 常见误区

- 每个 token 选择不同专家，不等于专家天然形成清晰人类学科。
- 负载均匀不等于专家学得有差异。
- Router 使用 FP32 是稳定性策略，不会让整个模型变成 FP32。
- 辅助损失太强可能牺牲主任务，太弱又可能专家坍缩。

## 阶段验收

实现 Top-k、容量、dropped mask、负载统计、balance loss 和 z-loss；能用热力图展示专家利用率，并解释算法计算与跨设备通信的区别。

