# 第 48 周：毕业项目、可复现实验与知识答辩

> 课程定位：最后一周不再堆叠新名词，而是把 Dense、现代 Attention、MoE、训练和推理系统放进一个可证伪问题。项目必须交付代码、测试、原始记录、图表、失败案例与口述解释。完整 rubric 与目录建议见扩展阅读 [毕业项目说明](../references/capstone.md)；本讲义负责把它变成执行和验收契约。

## 1. 学习目标

完成后应能提出只改变一个核心因素的研究问题；能为 Dense/Attention/MoE 对照写清参数、训练 token、数据顺序与推理负载；能用多 seed 和配对差值区分证据与偶然波动；能报告 loss/质量、训练速度、推理 TTFT/TPOT/吞吐、峰值内存和 MoE 路由；能让第三方从干净环境复现实验；能在答辩中从 shape、公式和失败现象解释自己的实现，而不只是运行 Notebook。

## 2. 前置知识

需要完成核心 Notebook 和至少一个端到端模型实验。按选题复习 [现代解码器 Notebook](../../labs/06_modern_decoder.ipynb)、[注意力前沿 Notebook](../../labs/08_attention_frontiers.ipynb)、[MoE Notebook](../../labs/09_moe.ipynb) 与 [推理 Notebook](../../labs/11_inference_serving.ipynb)。先阅读 [毕业项目说明](../references/capstone.md)，它是本周扩展阅读和最终目录/rubric 依据。不要为了“新颖”临时引入无法测试的大模型或云服务。

## 3. 核心直觉

毕业项目不是功能展览，而是一条证据链：问题决定对照，契约保证只改变目标因素，原始记录支撑图表，失败分析限定结论，测试保护实现正确性。最可信的项目往往规模很小，但每个数字能追溯到配置、commit、seed 和原始行；最不可信的项目常有漂亮曲线，却同时改变参数量、数据、训练步数和硬件。

答辩是对证据链的“反向执行”：评审从一张图追问聚合公式，从公式追问张量 shape，从 shape 追问代码断言，再追问某个反例会怎样。若只能说“库就是这样”，说明项目尚未达到从零到专业的学习目标。

## 4. 张量与数据契约

每次 run 必须有不可变标识 `run_id`，配置至少含 variant、seed、git commit、数据版本/哈希、tokenizer、参数量、训练 token、batch/累积、优化器、学习率计划、dtype、设备和依赖锁。结果记录至少含 step、train/valid loss、tokens_per_second、peak_memory；推理选题再含 input/output token、TTFT、TPOT、throughput；MoE 再含 expert load、dropped rate、aux loss。

Dense/Attention/MoE 输出 logits 都应是 `[B,T,V]`，labels `[B,T]`，padding/ignore index 一致。公平匹配必须预声明“总参数”还是“每 token 激活参数/FLOPs”；二者不能同时默认为相等。所有变体使用相同 train/validation split、tokenizer、token budget、seed 集和评测代码。原始结果采用机器可读 JSONL/CSV，图表是派生产物，不得手工改数字。

## 5. 公式推导与算法机制

先写主假设，例如：“在相同每-token 活跃参数、训练 token 与数据顺序下，Top-2 MoE 的验证 loss 是否低于 Dense，代价是怎样的路由不均和推理延迟？”主终点只能少量且预注册。对每个 seed s 计算配对差 `d_s=m_s(treatment)-m_s(baseline)`，汇报 `mean(d)`、各 seed 原值和离散程度；只比较两个独立均值会丢掉同 seed/数据顺序带来的配对信息。

速度测量必须先定义分母：训练 throughput 是有效非 padding token/秒；推理 output throughput 是完成输出 token/墙钟；峰值内存注明采样 API 和测量区间。模型质量至少含 validation loss/perplexity，并报告生成失败案例。MoE 负载可用每专家 assignment 与均值之比、最大/均值、变异系数；仅有 auxiliary loss 不能证明实际均衡。

证据等级也要标注：代码断言证明某个局部不变量；toy 实验支持教学机制；指定硬件实测支持该环境性能；它们都不能自动推出生产集群或更大模型结论。任何外推必须列为“推断”并给限制。

## 6. 手算与数值示例

假设两个 seed 的 validation loss：Dense 为 `[2.10,2.06]`，改进模型为 `[2.04,2.08]`。配对差是 `[-0.06,+0.02]`，均值 `-0.02`。只报均值“改进0.02”会隐藏第二个 seed 变差；正确结论是样本很少、方向不稳定，需要更多 seed 或降低表述强度。

若两个模型分别在100秒处理1,000,000与1,100,000有效 token，throughput 为10,000与11,000 token/s，表面提升10%。但若后者 padding 比例更高而分子用了总 token，比较会偏置；必须从 attention/loss mask 统计有效 token。若推理设置的输出长度不同，output throughput 同样不可直接比较。

## 7. 最小代码实现

~~~python
from statistics import mean

def paired_delta(rows, baseline="dense", treatment="candidate"):
    table = {(r["variant"], r["seed"]): r["valid_loss"] for r in rows}
    seeds_a = {s for (v, s) in table if v == baseline}
    seeds_b = {s for (v, s) in table if v == treatment}
    if seeds_a != seeds_b or not seeds_a:
        raise ValueError("variants must have the same non-empty seed set")
    deltas = [table[(treatment, s)] - table[(baseline, s)] for s in sorted(seeds_a)]
    return deltas, mean(deltas)

results = [
    {"variant": "dense", "seed": 1, "valid_loss": 2.10},
    {"variant": "candidate", "seed": 1, "valid_loss": 2.04},
    {"variant": "dense", "seed": 2, "valid_loss": 2.06},
    {"variant": "candidate", "seed": 2, "valid_loss": 2.08},
]
deltas, avg = paired_delta(results)
assert abs(deltas[0] + 0.06) < 1e-12
assert abs(deltas[1] - 0.02) < 1e-12
assert abs(avg + 0.02) < 1e-12
~~~

代码只检查同 seed 配对和一个均值，不做统计显著性、置信区间、异常值策略或实验追踪。完整项目应保留逐步指标、环境快照和失败 run，自动生成表图，并用测试验证参数/FLOPs 计算和数据一致性。

## 8. 反例、常见误区与调试

反例一：MoE 总参数远大于 Dense，却宣称“架构本身更好”，未说明匹配口径。反例二：不同变体分别挑最优 seed/checkpoint。反例三：训练 loss 更低就声称泛化更好，没有固定 validation。反例四：删掉 OOM 或发散 run，导致成功率虚高。反例五：Notebook 依赖上一次运行的隐藏状态，重启 kernel 后无法复现。

调试按证据链逆序：先在干净环境执行安装与测试；从原始记录重新生成所有表图并核对行数；检查每个 variant 的 seed、token budget、数据哈希和超参集合；用极小 batch 做 shape/梯度/因果 mask 断言；对一条异常曲线定位到 step 日志和 checkpoint。最后让同学根据 README 从零复现一次，记录所有需要口头补充的步骤并写回文档。

## 9. 主流工作与实现边界

Transformer、FlashAttention、Switch Transformer 等论文的实验部分展示了架构/算法、规模和系统条件如何共同构成结论；HELM 强调场景、指标与透明度；MLPerf 通过固定任务、质量门槛和规则提升系统结果可比性。本项目借鉴这些方法论，不要求复现论文规模。

推荐主项目仍用仓库的 toy/小型 CPU 或单 GPU 路径，保证可重复完成。可选的大模型、云端服务或多 GPU 结果必须独立标注，不能成为通过课程的唯一证据。互动图是解释工具，Notebook 是实验入口，starter 是能力核查；三者都不是生产训练/服务栈。

## 10. 实验与 Notebook 对照

先从 [毕业项目说明](../references/capstone.md) 选择一个最小问题并写预注册表。架构对照使用 [架构演化互动图](../interactive/architecture-evolution.html)，Attention/MoE 分别对照 [注意力 Notebook](../../labs/08_attention_frontiers.ipynb) 与 [MoE Notebook](../../labs/09_moe.ipynb)，服务指标对照 [推理 Notebook](../../labs/11_inference_serving.ipynb) 和 [服务互动图](../interactive/serving-lab.html)。按选题完成 [starter 16](../../labs/starter/16_attention_frontiers.py)、[starter 18](../../labs/starter/18_moe_systems.py) 或 [starter 20](../../labs/starter/20_inference_systems.py)。

最低实验矩阵包含 baseline、一个 treatment、相同的至少两个 seed、固定 token budget 和一套 validation/serving trace。先跑 smoke test，再跑正式矩阵；所有 run 包括失败 run 都写 manifest。报告正文按“问题—方法—契约—结果—失败案例—限制—下一步”组织，并准备五分钟白板说明一个核心张量流和一个失败案例。

## 11. 验收标准

- 干净环境可安装，核心测试全通过，Notebook restart-and-run-all 无隐藏状态。
- 配置、commit、依赖、数据哈希、seed、原始 JSONL/CSV 与生成图表脚本齐全。
- Dense/Attention/MoE 的匹配口径明确；同数据、token budget、评测与硬件条件可核查。
- 报告每 seed 原值、配对差、质量/速度/内存；MoE 项目另报路由负载与 dropped。
- 至少两个真实失败案例，分别写症状、定位证据、修复或未解决限制。
- 口述答辩能手算一个核心公式、说清 shape/mask，并区分证据、推断与生产边界。
- 最终目录和 rubric 逐项对照 [毕业项目说明](../references/capstone.md)，他人可从零复现。

## 一手来源

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)：Transformer 架构与实验报告基线。
- [FlashAttention](https://arxiv.org/abs/2205.14135)：算法正确性、IO 复杂度与硬件实测如何分层论证。
- [Switch Transformers](https://arxiv.org/abs/2101.03961)：稀疏模型的容量、负载与规模对照。
- [HELM 原论文](https://arxiv.org/abs/2211.09110)及[官方仓库](https://github.com/stanford-crfm/helm)：多场景、多指标、透明可复现评测。
- [MLPerf Inference 原论文](https://arxiv.org/abs/1911.02549)：带质量约束和场景规则的系统基准方法。
