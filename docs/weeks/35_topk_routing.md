# 第 35 周：条件计算与 Top-k 路由

> 课程定位：本周把现代 Decoder 中每个 token 都经过同一个 MLP 的密集计算，改造成
> “先路由、再只执行少数专家”的条件计算。重点不是背诵 MoE 名称，而是把路由语义、
> 梯度路径、总参数与活跃计算分开核算，为第 36 周容量约束和第 39 周专家并行建立契约。

## 1. 学习目标

完成本周后，你应能从输入 shape 推导 router logits、Top-k assignment、专家输出和最终
聚合的 shape；能区分 token-choice 与 expert-choice；能解释 softmax router 与 sigmoid
router、Top-1 与 Top-k、选中权重是否重归一化带来的不同梯度；能用相同隐藏宽度比较
Dense FFN 与 MoE 的总参数、活跃参数和近似 FLOPs；能构造一个覆盖全部专家的批次，验证
被选专家和 router 的梯度，而不是只看 loss 是否下降。

## 2. 前置知识

需要掌握线性层、Softmax、交叉熵、PyTorch 自动微分以及 Transformer MLP。你不需要先懂
分布式 all-to-all；本周把所有专家放在单个 CPU 进程中。建议先复习
[现代 Decoder Notebook](../../notebooks/core/06_modern_decoder.ipynb) 中 SwiGLU 的输入输出，
再打开 [MoE Notebook](../../notebooks/core/09_moe.ipynb)。MoE 通常替换的是 FFN 子层，
注意力仍负责 token 间通信；“专家”不是一个完整语言模型。

## 3. 核心直觉

Dense MLP 像所有 token 共用同一组参数的工厂。MoE 增加 E 组工厂，但每个 token 只送到
k 组，因此可以在不让单 token 计算随总参数线性增长的前提下扩大模型容量。这里必须同时
保留两本账：**容量账**统计所有 E 个专家参数，**计算账**只统计本批实际激活的 k 个专家。
MoE 并不自动更快；路由、重排、padding、负载不均和跨设备通信可能吞掉稀疏计算收益。

token-choice routing 让每个 token 选择专家，容易出现热门专家过载；expert-choice 反过来
让每个专家选择固定数量 token，容量更直接，但一个 token 可能被零个或多个专家选择。
本课程主实现采用 token-choice，因为它与 Switch、GShard 以及多数 Decoder MoE 的教学
路径一致；两者不可只改 Top-k 维度便宣称等价。

## 4. 张量与数据契约

设输入 x 的 shape 为 [B,T,D]，展平后 token 数 N=B×T。Router 权重为 [E,D]：

[
z=xW_r^	op in mathbb{R}^{N	imes E},qquad p=operatorname{softmax}(z)
]

Top-k 返回 indices 与 gates，shape 都是 [N,k]。第 e 个专家 f_e 接收被分派到它的
若干行 [N_e,D]，输出仍为 [N_e,D]。聚合结果是

[
y_i=sum_{ein S_i}g_{i,e}f_e(x_i),qquad yinmathbb{R}^{N	imes D}.
]

契约必须显式说明：softmax 是在全部 E 个专家上做，还是先选 Top-k 再归一；sigmoid 是否
逐专家独立打分；相同 token 的 k 个 gate 是否和为 1；被拒绝 assignment 如何处理；
padding token 是否参与负载统计。课程 TopKMoE 使用 FP32 计算 router logits 和概率，
然后再按输入 dtype 执行专家；Top-k index 是离散选择，不能假设它本身可导。

## 5. 公式推导与算法机制

若先对 E 个 logits 做 softmax，再取 Top-k，保留 gate 的和通常小于 1；若对选中 gate
重新归一，则它们和为 1。以 logits [2,1,0,-1]、k=2 为例，选中专家 0 和 1。只在二者
之间重归一后，权重约为 [0.731,0.269]。不重归一时，它们是全局 softmax 中的两项，
还保留“未选专家概率质量”的缩放信息。

Top-1 有一个关键反例：如果选中后重归一，唯一 gate 恒等于 1。主任务输出只依赖被选专家，
在路由 index 不可导的实现里，主任务对 router 权重的梯度为零。此时 router 只能由负载均衡
损失、z-loss 或其他可导代理训练。这不是 PyTorch 故障，而是所选语义的直接结果。Top-2
重归一时两个 gate 仍随 logits 变化，主任务可沿 gate 回传，但 Top-k 集合边界仍是离散的。

参数账也要拆开。若每个专家是 D→M→D 的两层 MLP，忽略 bias，一个专家约有 2DM 参数，
E 个专家总计约 2EDM；每 token 激活 k 个专家，理想活跃乘加约 2kDM。Router 另有 ED
参数。这个估算没有包含激活函数、dispatch、通信和 padding，不能直接当作墙钟速度。

## 6. 手算与数值示例

考虑两个 token、三个专家、Top-2。token A 的 logits 为 [2,1,0]，token B 为 [0,2,1]。
重归一后 A 分给专家 0/1，B 分给专家 1/2。assignment load 是 [1,2,1]，但 token 数只有
2，所以负载之和为 N×k=4。若误把 load 除以 N 而不是 N×k，利用率会整体放大 k 倍。

再令三个专家对一个标量分别输出 10、20、30。A 的 Top-2 gate 约为 [0.731,0.269]，
聚合值约为 12.69；B 聚合专家 1/2 后约为 22.69。这个例子能同时检查专家索引、gate 与
scatter-add。如果输出变成 30 或 50，通常是忘记 gate；如果 shape 正确但 token 顺序错，
通常是 dispatch 后没有按原 token index 聚合。

## 7. 最小代码实现

下面是只展示语义的 CPU 片段；真实实现还需容量、drop/dropless、混合精度和并行通信。

~~~python
import torch

x = torch.tensor([[1.0, 0.0], [0.0, 1.0]])       # [N=2, D=2]
router = torch.tensor([[2.0, 0.0], [1.0, 2.0], [0.0, 1.0]])  # [E=3, D=2]
experts = [lambda t, scale=s: t * scale for s in (1.0, 2.0, 3.0)]

logits = x.float() @ router.float().T
top_values, top_indices = logits.topk(k=2, dim=-1)
gates = top_values.softmax(dim=-1)
out = torch.zeros_like(x)
for token in range(x.shape[0]):
    for slot in range(2):
        expert = int(top_indices[token, slot])
        out[token] += gates[token, slot] * experts[expert](x[token])
assert out.shape == x.shape
print(top_indices, gates, out)
~~~

该循环实现复杂度和内存布局都不适合生产，只用于暴露 token、slot、expert 三个索引。
完成 [starter 10](../../exercises/starter/10_moe_router.py) 时应保持相同数学语义，再用
参考实现测试梯度和边界。

## 8. 反例、常见误区与调试

第一类误区是把“671B 总参数、37B 激活参数”误写成每 token 完全只计算 37B：嵌入、
注意力、共享专家和输出层仍是密集部分，模型报告的口径也必须核对。第二类是只打印
selected load；容量机制生效后还要区分 selected 与 accepted，否则看不见 dropping。
第三类是用随机小批次要求每个专家都有梯度；随机路由可能没有覆盖全部专家，测试应手工
设置 router 权重和输入。

调试顺序建议固定为：检查 logits 是否有限；确认 Top-k 轴是 expert 轴；核对每 token
gate 和；统计 assignment 总数是否为 N×k；验证每个 token 聚合回原位置；最后再看梯度。
若 Top-1 重归一后的主任务 router gradient 恰为零，先判断这是否符合契约，不要用
retain_graph 或随机噪声掩盖机制。

## 9. 主流工作与实现边界

GShard 展示了 Top-2 gating 与自动分片的大规模条件计算；Switch Transformer 将路由简化
为 Top-1，并系统讨论容量、低精度和稳定性。DeepSeekMoE 使用更细粒度的 routed experts
与 shared experts，目标是减少知识冗余；OLMoE 提供权重、数据、代码和训练日志，适合
核查路由专化主张。它们的模型规模、训练数据和系统栈不同，不能只凭 active parameter
数字横向排名。

本周实现是单机 token-choice reference，不复现 XLA 自动分片、融合 grouped GEMM、
expert parallel 或通信重叠。论文中的质量/吞吐结论必须标注作者报告的硬件、batch、
专家数和路由语义。第 39 周才把算法路由接到 all-to-all 数据流。

## 10. 实验与 Notebook 对照

先在 [MoE 互动图](../interactive/core-concepts.html#moe) 中改变专家数、Top-k 与容量，
预测 selected load；再运行 MoE Notebook 的路由单元。实验一固定专家输出，只验证
dispatch/aggregate；实验二构造覆盖全部专家的输入并检查梯度；实验三比较 Top-1
重归一、Top-2 重归一和不重归一的 router gradient；实验四用
moe_parameter_accounting 比较总参数与活跃参数。每次记录 N、E、k、dtype 和 gate 语义。

## 11. 验收标准

- 对任意合法 [B,T,D] 输入，输出 shape、dtype、device 与输入一致且数值有限。
- 每个 token 恰有 k 个 selected assignment，gate 语义与配置一致。
- 手算的两 token、三专家例子与实现误差小于 1e-6。
- 构造覆盖全部专家的批次后，被选专家均有有限梯度。
- 能解释 Top-1 重归一时主任务 router 梯度为何可能为零。
- 提交 starter 10，并运行 uv run llm-course exercises check 10。
- 交付一张总参数、活跃参数、理想 FLOPs 与未计系统成本的四列表。

## 一手来源

- [GShard](https://arxiv.org/abs/2006.16668)：Top-2 gating、条件计算与自动分片。
- [Switch Transformer](https://arxiv.org/abs/2101.03961)：Top-1 路由、容量和低精度训练。
- [DeepSeekMoE](https://arxiv.org/abs/2401.06066)：细粒度专家与共享专家。
- [OLMoE](https://arxiv.org/abs/2409.02060)：开放权重、数据、代码、日志与路由分析。
- [Expert Choice Routing](https://arxiv.org/abs/2202.09368)：与 token-choice 不同的容量视角。
