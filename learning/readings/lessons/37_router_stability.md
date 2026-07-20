# 第 37 周：Router Z-loss、数值精度与训练稳定性

> 课程定位：容量解决“热门专家塞不下”，却没有限制 router logits 的尺度。本周研究稀疏路由
> 特有的数值与优化失败：为什么 logits 会漂大、为什么 BF16 路由可能提前饱和、z-loss 与
> balance loss 分别约束什么，以及怎样用确定性梯度实验判断 router 真正在学习。

## 1. 学习目标

完成后应能稳定计算极端 router logits 的 softmax、logsumexp 与 z-loss；能解释 z-loss 不是
负载均衡损失；能实现“router 投影与归约使用 FP32、专家保留模型 dtype”的精度契约；能分别
测量主任务、balance loss、z-loss 对 router 的梯度；能记录 router entropy、logit RMS/max、
expert load 和非有限值，并用这些指标区分数值爆炸、路由塌缩和正常专化。

## 2. 前置知识

需要理解第 35 周的 Top-k gate 和第 36 周的负载统计，并熟悉 log-sum-exp trick、BF16 动态范围、
梯度范数。运行 [MoE Notebook](../../labs/09_moe.ipynb) 前，先复习稳定 Softmax：
减最大值避免 exp overflow，但它只修复计算表达式，不会阻止模型学出越来越大的相对 logit。
本周对应 [starter 18](../../labs/starter/18_moe_systems.py) 的系统稳定性部分。

## 3. 核心直觉

Router 像一个必须在训练早期迅速做决定的分类器。若 logits 尺度不断增大，softmax 越来越接近
one-hot，未选专家得到的可导信号变弱，微小输入或低精度舍入可能改变 Top-k 边界。稳定 softmax
能避免 NaN，却不能把过度自信拉回来。Z-loss 给 log-partition 一个“橡皮筋”，惩罚整体
logsumexp 远离零，控制 logits 的共同尺度。

Balance loss 观察“谁被选了以及平均概率”，目标是利用率；z-loss 观察每个 token 的
logsumexp，目标是数值/优化尺度。一个 batch 完全均衡仍可能 logits 巨大，一个 batch logits
温和也可能全部偏爱专家0。二者必须分别调系数和画曲线。

## 4. 张量与数据契约

输入 x 为 [N,D]，router weight 为 [E,D]。即使 x 与专家权重是 BF16，课程契约规定：

[
z=operatorname{linear}(x_{mathrm{fp32}},W_{r,mathrm{fp32}})inmathbb{R}^{N	imes E},
quad p=operatorname{softmax}(z,	ext{dim}=-1).
]

Top-k index 为 [N,k] long，概率和 auxiliary losses 保持 FP32；选中 gate 在乘专家输出前可转回
输出 dtype。禁止先用 BF16 matmul 得到 logits 再 cast FP32，因为量化误差已经发生。空 batch、
E<k、NaN 输入和无效 dtype 要显式拒绝。

Z-loss 常用

[
L_z=rac1Nsum_i left(logsum_e exp z_{i,e}
ight)^2.
]

总 loss 是 L_task + αL_balance + βL_z。报告梯度时必须注明是否清空旧 grad、是否在同一计算图
重复 backward，以及系数 α、β；否则无法比较量级。

## 5. 公式推导与算法机制

令 a_i=logsumexp(z_i)。对一个 token，

[
rac{partial a_i^2}{partial z_{i,e}}=2a_i,p_{i,e}.
]

因此当 a_i 很大为正时，梯度会推动高概率 logits 降低；当 a_i 为负时方向相反。它不是简单的
L2(logits)：给所有 logits 同加常数 c，不改变 softmax 概率和 Top-k，却会让 a_i 增加 c，
z-loss 正好能观察这种 softmax 不可辨识的共同平移。也因此 z-loss 不直接要求专家利用率均衡。

数值稳定的 logsumexp 使用 m=max(z)：

[
logsum_e e^{z_e}=m+logsum_e e^{z_e-m}.
]

若 z=[1000,999]，直接 exp 溢出；稳定结果约为1000.313。Softmax 约为[0.731,0.269]。
Z-loss 仍约为 1,000,626，明确告诉优化器 logits 尺度过大。只说“softmax 输出有限”会漏掉
这个优化风险。

Top-1 且选中 gate 重归一为1时，主任务可能不给 router 梯度；balance/z-loss 仍能回传。
Top-2 gate 通常允许主任务沿连续权重回传。测试需要固定路由权重，不能依赖随机批次刚好覆盖
某个梯度路径。

## 6. 手算与数值示例

比较 zA=[2,1,0,-1] 与 zB=[102,101,100,99]。两者 softmax 和 Top-k 完全相同，因为只是共同
加100；但 logsumexp 分别约2.440和102.440，z-loss 相差三个数量级。这个例子证明 z-loss
控制的是 router 分区函数尺度，而不是改变当前路由排序。

再看 BF16 边界：两个很接近的大数可能舍入成相同值，Top-k tie-breaking 由实现细节决定。
FP32 router 能降低这种信息损失，但不能消除数学上的平局。验收不应强制相同分数时唯一 index，
而应检查选中集合合法、输出有限，并在需要确定性时人为加入可说明的 tie-break 规则。

## 7. 最小代码实现

~~~python
import torch

def router_losses(x, weight, top_k=2):
    logits = torch.nn.functional.linear(x.float(), weight.float())
    probabilities = logits.softmax(dim=-1)
    top_indices = probabilities.topk(top_k, dim=-1).indices
    selected = torch.nn.functional.one_hot(
        top_indices, num_classes=logits.shape[-1]
    ).float().sum(dim=1)
    selected_fraction = selected.mean(dim=0) / top_k
    mean_probability = probabilities.mean(dim=0)
    balance = logits.shape[-1] * (selected_fraction * mean_probability).sum()
    z_loss = torch.logsumexp(logits, dim=-1).square().mean()
    return logits, top_indices, balance, z_loss

x = torch.tensor([[1000.0, 999.0]])
weight = torch.eye(2)
logits, indices, balance, z_loss = router_losses(x, weight)
assert torch.isfinite(logits).all()
assert torch.isfinite(balance + z_loss)
~~~

这只是 loss oracle，不执行专家。生产训练还要处理 distributed reduction、loss scale、autocast
边界和日志聚合。不要把 router FP32 误写成整个 MoE 必须 FP32。

## 8. 反例、常见误区与调试

误区一：减去每行最大值后 logits 就“被正则化”了。减最大值只是等价计算 softmax，模型参数
仍可能漂移；如果用减过最大值的 logits 计算 z-loss，会消除它要约束的共同平移。误区二：
z-loss 越大越应提高 β。过强 β 会与主任务竞争，必须比较 validation loss、entropy 和梯度。
误区三：router entropy 下降就是塌缩。专家专化会降低某些 token 的 entropy，只有结合跨 batch
load、容量溢出和质量才能判断。

出现 NaN 时按顺序保存第一处非有限张量：输入 x、FP32 logits、logsumexp、probability、aux
loss、总 loss、router grad。出现 load collapse 但数值有限时，再检查数据顺序、初始化、辅助
系数和 capacity。不要先加 gradient clipping；它可能隐藏根因。

## 9. 主流工作与实现边界

Switch Transformer 讨论 selective precision：局部将 router 计算提升到 FP32，以稳定 BF16
训练。ST-MoE 系统研究 router z-loss、训练稳定性和迁移，并把 z-loss 作为重要设计组件。
DeepSeek-V3 报告 auxiliary-loss-free load balancing 与 FP8 混合精度，但其“稳定训练”来自
模型、数据、优化器、通信和精度系统共同设计，不能归因于单个 bias 或 z-loss。OLMoE 开放训练
日志，适合观察 router entropy、负载和专化如何随训练变化。

课程实现只验证 FP32 router、balance/z-loss 梯度和 BF16 输入边界，不复现大规模 loss
reduction、FP8 recipe 或硬件通信容错。

## 10. 实验与 Notebook 对照

实验一给 logits 加常数 c，验证 softmax 不变而 z-loss 改变；实验二从 c=0 扫到1000，比较 naive
与稳定 logsumexp；实验三分别 backward 主任务、balance、z-loss，记录 router grad norm；
实验四对相同 x/weight 比较 FP32 与 BF16 先投影后的 Top-k 差异；实验五人为让专家0极热门，
联合观察 entropy、selected/accepted load 与 capacity drop。所有图标注 seed、dtype 和系数。

## 11. 验收标准

- 对 [1000,999] 得到有限 softmax、logsumexp 与 z-loss。
- 给 logits 同加常数后 probability 误差小于1e-6，z-loss 必须变化。
- 能用导数 2a·p 解释 z-loss 梯度方向。
- Router 投影在 autocast 环境中仍以 FP32 完成，专家输出保持目标 dtype。
- 分开报告 task、balance、z-loss 的 router gradient，不复用旧 grad。
- 构造极端 logits 与 tie 两类反例且无 NaN。
- 完成 starter 18 对应函数，并运行 uv run llm-course exercises check 18。

## 一手来源

- [Switch Transformer](https://arxiv.org/abs/2101.03961)：selective precision 与稀疏训练稳定性。
- [ST-MoE](https://arxiv.org/abs/2202.08906)：router z-loss、稳定训练与迁移研究。
- [DeepSeek-V3 技术报告](https://arxiv.org/abs/2412.19437)：负载策略、FP8 与系统协同的作者报告。
- [OLMoE](https://arxiv.org/abs/2409.02060)：开放训练日志与路由专化分析。
- [PyTorch logsumexp 文档](https://pytorch.org/docs/stable/generated/torch.logsumexp.html)：稳定归约接口。
