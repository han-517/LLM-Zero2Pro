# 第 34 周：混合架构、DSA 与长上下文评测——从组件到系统证据

## 课程定位

本阶段前六周分别研究 FlashAttention、稀疏/滑窗注意力、MLA、线性注意力、Mamba-2 和 Gated DeltaNet。最后一周不再把这些机制当作互斥选项，而是研究现代模型如何按层、按头或按 token 混合它们，并建立一套能同时衡量质量、长上下文利用率、训练效率与推理成本的证据框架。重点案例包括 Kimi Linear 的混合线性/注意力设计、Qwen3.5 的 Gated DeltaNet—注意力混合栈，以及 DeepSeek-V3.2 的 DeepSeek Sparse Attention（DSA）。

## 学习目标

完成本周后，你应能描述混合架构的层调度、状态与 KV cache 契约；解释 DSA 的轻量 indexer 与细粒度 token selection 如何减少注意力计算；区分“支持某个上下文长度”“能检索远处信息”“能在真实任务中有效使用上下文”三种不同主张；设计固定参数量、训练 token、数据、硬件和精度的公平对照；阅读 2025–2026 模型报告时标明事实、作者实验与自己的推断；为新架构建立质量—吞吐—延迟—内存的多目标验收，而不是用单个 needle 测试或峰值速度替代完整评价。

## 前置

需要掌握因果注意力、RoPE、MQA/GQA/MLA、KV cache、FlashAttention 的 IO 视角、稀疏注意力索引、线性 recurrent state、Mamba-2/SSD 和 Gated DeltaNet。还应理解 prefill 与 decode 的瓶颈不同：prefill 通常是大矩阵计算和 IO 问题，decode 更容易受缓存读取、kernel launch 与 batch 策略影响。评测部分需要均值、分位数、置信区间、数据污染和长度分桶的基本概念。

## 直觉

没有一种 token mixer 同时在所有序列长度、硬件和任务上占优。全注意力擅长任意 token 间直接内容寻址，却有二次计算或较大缓存；线性/状态空间层以固定大小状态压缩历史，效率好但可能出现记忆碰撞；局部或稀疏注意力减少候选，却依赖稀疏模式是否覆盖真正相关位置。因此现代设计常让大多数层承担低成本的局部或压缩建模，间隔插入少量全局注意力层纠正信息瓶颈。

DSA 则是在注意力内部学习“看哪些过去 token”。轻量 indexer 先计算相关性并选出 top-k 候选，昂贵的注意力只在被选 token 上执行。它与固定滑窗不同，也与 MLA 不同：MLA主要压缩 KV 表示，DSA主要稀疏化 token 连接；二者可以组合。top-k 选择引入离散索引、训练—推理一致性和 kernel 实现问题，不能只写一个 masking demo 就声称完成生产 DSA。

## 张量/数据契约

混合堆栈可用长度为 $L$ 的 `layer_types` 描述，每项属于 `{attention, local_attention, linear, ssm}`。每层输入输出保持 $x\in\mathbb{R}^{B\times T\times d_{model}}$，但推理状态不同：attention 层持有按层 KV cache；MLA 层持有潜变量缓存及位置相关分量；linear/SSM 层持有固定大小 recurrent state；局部注意力只保留窗口所需缓存。服务端必须把这些状态与请求、batch 重排、序列完成和滑窗截断同步管理。

教学版 DSA indexer 接收查询与索引键 $q^I,k^I\in\mathbb{R}^{B\times H_I\times T\times D_I}$，产生因果得分 $s\in\mathbb{R}^{B\times H_I\times T\times T}$ 和候选索引 `idx[B,H_I,T,K]`；主注意力再从 K/V 序列 gather 被选位置。完整算法可能共享/压缩 indexer 头、采用专门训练目标与 fused sparse kernel，且必须处理 causal、padding、分布式分片和重复索引。教学实现只用于验证选择语义与因果性。

评测数据契约至少包含 `example_id`、任务、原始长度、有效 token 长度、目标证据位置、答案、模板版本和污染标记。系统指标要记录设备、软件版本、dtype、batch、并发、上下文长度、输出长度、是否含 tokenizer/传输时间以及 warm-up 方式，否则不同架构的数字不可比较。

## 推导与机制

设第 $\ell$ 层 mixer 为 $M_{z_\ell}$，其中 $z_\ell$ 是层类型，则残差块可统一写为

$$
x_{\ell+1}=x_\ell+M_{z_\ell}(\operatorname{Norm}(x_\ell)).
$$

统一接口不代表内部状态相同。若每 $r$ 层插入一个注意力层，attention 比例约为 $1/r$；但总成本不能只乘层数，因为不同 mixer 的投影维度、状态大小、卷积、门控与 kernel 效率不同。应分别实测或建模

$$
C_{total}=\sum_{\ell=1}^{L} C_{z_\ell}(B,T,d,\text{dtype},\text{hardware}),
$$

并报告参数、训练 FLOPs、激活、推理状态和 wall-clock。

对 DSA，indexer 可抽象为 $s_{t,j}=f(q_t^I,k_j^I)$，在 $j\le t$ 中选择集合 $I_t=\operatorname{TopK}(s_{t,:},K)$，再计算

$$
y_t=\sum_{j\in I_t}\operatorname{softmax}_{j\in I_t}
\left(\frac{q_tk_j^\top}{\sqrt d}\right)v_j.
$$

若 $K\ll T$，主注意力的候选数从 $T$ 降为 $K$，但 indexer 自身也有成本。论文中的具体 indexer、训练损失、候选规模与 kernel 才决定实际收益。排序/选择还必须先施加 causal mask；若先在全序列 top-k 再过滤未来位置，会造成候选不足甚至信息泄漏。

长上下文质量应按长度和证据位置分桶。有效上下文不是配置里的最大位置编号，而是在指定正确率或损失阈值下仍能利用证据的长度。系统侧至少分别报告 prefill tokens/s、time-to-first-token、decode tokens/s、p50/p95 延迟、峰值显存与并发吞吐，并与短上下文质量、训练稳定性一起构成 Pareto 前沿。

## 数值例

设 24 层模型采用“三个线性层加一个注意力层”的重复模式，则有 18 个线性层和 6 个注意力层，attention 比例为 25%。如果上下文 8192、每个 token 的注意力层 KV 缓存为 16 KiB，而线性层状态每层每请求为 256 KiB，则不能简单说“固定状态一定更省”：注意力缓存约为 $6\times8192\times16$ KiB，线性状态约为 $18\times256$ KiB，实际还要加入 batch、量化、对齐和其他张量。

再设 DSA 在长度 4096 上每个 query 选 512 个历史 token，主注意力候选数理论上约降到八分之一，但若 indexer 仍显式构造 $4096^2$ 得分矩阵，教学代码不会获得线性内存优势。生产实现必须避免物化完整矩阵或用高效索引 kernel；这正是算法复杂度、参考实现和硬件性能三层边界的典型例子。

## 最小代码

下面的代码只建立混合层调度和可审计的理论成本账本。它不实现真实层，也不将符号估算冒充 benchmark；真实模型必须用相同环境计时并核查输出质量。

```python
from dataclasses import dataclass

def repeated_schedule(depth: int, pattern=("linear", "linear", "linear", "attention")):
    if depth <= 0 or not pattern:
        raise ValueError("depth and pattern must be positive/non-empty")
    return [pattern[i % len(pattern)] for i in range(depth)]

@dataclass
class Cost:
    flops: float = 0.0
    state_bytes: int = 0

def estimate_schedule(schedule, batch, seq, d_model, kv_bytes_per_token,
                      recurrent_bytes_per_layer):
    total = Cost()
    for kind in schedule:
        if kind == "attention":
            total.flops += 2 * batch * seq * seq * d_model
            total.state_bytes += batch * seq * kv_bytes_per_token
        elif kind in {"linear", "ssm"}:
            total.flops += 2 * batch * seq * d_model * d_model
            total.state_bytes += batch * recurrent_bytes_per_layer
        else:
            raise ValueError(f"unknown mixer: {kind}")
    return total

schedule = repeated_schedule(24)
assert schedule.count("attention") == 6
```

DSA starter 应另外留出 `causal_index_scores`、`topk_indices`、`gather_kv` 与 `selected_softmax` 空缺。核查器使用极小张量与 dense masked attention 对照：当 $K=T$ 时输出应相等；任何被选索引都必须不大于当前位置；padding 不得进入候选；相同得分时要有可复现的 tie 行为或明确允许集合等价。

## 反例与调试

最常见的概念错误是把“混合架构”画成若干顺序模块，却没有给出层比例、每层状态和位置编码作用范围。部分层使用 RoPE 不等于把一个 RoPE 模块串在整个网络前面。第二类错误是把 MLA、GQA、DSA 和 FlashAttention混为一谈：它们分别主要改变 KV 表示、共享粒度、连接稀疏度和精确注意力 kernel 的 IO；可能组合，但回答的问题不同。

第三类错误是 DSA 先看到未来 token 再做 top-k，或 gather 后的 mask/位置索引错位。第四类错误是只报告理论复杂度而不计 indexer、gather 和小矩阵低利用率。第五类错误是用单一 needle-in-a-haystack 得分证明模型“理解”长文；needle 可能被模板线索破解，无法覆盖多跳、聚合、顺序、干扰和真实语言建模。第六类错误是比较不同硬件、batch、精度或输出长度的 tokens/s，或者只报最好一次结果而不报分位数。

还要警惕时间穿越：截至 2026 年的新预印本和模型卡可用于描述前沿，但要标明发布日期、公开证据和复现状态。不能把 2026 年架构写成早期模型的组成，也不能因官方模型卡说明采用某组件，就推断该组件独自造成全部能力提升。

## 主流工作与证据等级

一级证据包括论文正文、附录、作者代码与可复现实验；二级证据包括官方技术报告和模型卡；三级证据包括高质量博客，适合建立直觉但不承载精确结论。Kimi Linear 公开提出以 KDA 线性层和 MLA 层构成混合栈；Qwen3.5 官方模型卡公开描述 Gated DeltaNet 与 gated attention 的混合设计；DeepSeek-V3.2 报告 DSA，并需要结合官方仓库和实测理解其 kernel 边界。这些案例说明混合已是重要前沿方向，但不能据此宣称所有主流模型都已放弃全注意力。

评测证据也分层：LongBench 提供多任务长上下文基准，RULER 强调随长度增长的可控合成任务，模型报告则可能使用专有数据和服务栈。课程要求交叉使用至少一种受控诊断、一种真实任务和一种系统 benchmark。若数据或实现不公开，结论必须降级为“报告所述”，而不是“已独立验证”。

## Notebook、互动图与 starter

互动图应允许选择层数、混合 pattern、上下文长度、KV dtype、DSA top-k 与 recurrent state 大小，实时展示理论计算、状态内存和注意力覆盖率；同时用醒目标记说明结果不是实测速度。第二个视图画出 query 到被选 token 的稀疏连边，并可切换 causal mask、固定滑窗、随机稀疏和学习索引，帮助定位未来泄漏与证据遗漏。starter 提供统一 mixer 接口、状态容器、DSA gather 骨架和评测记录 schema，核心逻辑留空，由结构测试、数值 oracle 和 benchmark 元数据检查共同核查。

## 实验

实验一在相同层数、参数量和训练 token 下比较全注意力、固定滑窗、3:1 线性—注意力混合，按长度分桶报告语言建模损失和关联召回。实验二对 DSA 改变 top-k，记录相对 dense attention 的输出误差、证据召回率、prefill 时间和峰值内存；必须把 indexer 时间计入。实验三使用 LongBench 类真实任务与 RULER 类受控任务，按证据深度和位置绘制曲线，而不是只报平均分。

实验四模拟在线服务，固定硬件、dtype、batch 与输出长度，分别测量 TTFT、decode tokens/s、p50/p95、峰值显存和最大并发。实验五做状态续接：整段 prefill 与多块 prefill 的最终 logits、KV/state 必须在容差内一致。报告需列出 commit、依赖版本、随机种子、warm-up、重复次数和原始结果；任何跨论文数字比较都要注明不可控变量。

## 验收 rubric

及格要求是画对至少一种混合层调度，区分 MLA、DSA、FlashAttention 与线性层，并给出无未来泄漏的 DSA 小张量测试。良好要求是完成 K=T 对 dense oracle、分块状态续接、长度/证据位置分桶和完整系统指标记录，解释理论复杂度与 wall-clock 的差异。优秀要求是在统一预算下形成质量—效率 Pareto 图，复现至少一个官方架构配置或公开 kernel，逐项列出教学 baseline 与完整算法/生产实现差异，并依据一手来源、官方报告和独立实验分别标注证据等级。

## 一手来源

- [Kimi Linear: An Expressive, Efficient Attention Architecture](https://arxiv.org/abs/2510.26692)
- [Qwen3.5 官方模型卡](https://huggingface.co/Qwen/Qwen3.5-35B-A3B)
- [DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models](https://arxiv.org/abs/2512.02556)
- [LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding](https://arxiv.org/abs/2308.14508)
- [RULER: What's the Real Context Size of Your Long-Context Language Models?](https://arxiv.org/abs/2404.06654)
