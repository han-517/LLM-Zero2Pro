# 第 36 周：容量、Token Dropping 与负载均衡

> 课程定位：第 35 周假设每个 selected assignment 都能进入专家。本周加入真实系统必须面对的
> 有限容量：热门专家不能无限接收 token。你将同时维护 selected、accepted、dropped 三套统计，
> 比较固定容量 dropping 与 dropless block-sparse 路线，并理解负载均衡损失只是训练信号，不是
> 容量保证。

## 1. 学习目标

完成后应能从 token 数、专家数、Top-k 和 capacity factor 算出每专家容量；能区分 token load
和 assignment load；能实现“按 gate 保留前 C 个 assignment”的确定性策略；能解释 dropping
发生后 residual path、MoE 输出和 loss mask 如何处理；能手算 Switch 风格辅助损失；能比较
dropping、padding 与 dropless 三种系统权衡，并用 selected/accepted/dropped 图定位末尾 token
被系统性丢弃的问题。

## 2. 前置知识

需要完成第 35 周 Top-k 路由，理解 N=B×T、E 个专家和每 token 的 k 个 assignment。
本周仍在单机 CPU 上模拟，不要求 CUDA kernel。先打开
[MoE Notebook](../../notebooks/core/09_moe.ipynb) 中 capacity 单元和
[starter 05](../../exercises/starter/05_moe_capacity.py)。注意容量约束作用于 assignment，
不是原始 token：Top-2 时一个 token 可能一个 assignment 被接收、另一个被拒绝。

## 3. 核心直觉

路由器像把包裹送到 E 个仓库。平均每仓应收到 Nk/E 个 assignment，但 Top-k 只看内容分数，
不会自动关心仓库是否拥堵。固定容量方案为每个专家预留 C 个槽；超出的 assignment 要么丢弃，
要么转给备选专家，要么通过额外 padding/动态 kernel 计算。容量因子大可以减少 dropping，
却增加预留内存和空算；容量因子小提高表面利用率，却可能让重要 token 失去专家更新。

负载均衡损失像拥堵价格，尝试在训练期间改变 router 偏好；容量是前向执行的硬约束。即使辅助
损失很小，单个 batch 仍可能溢出；即使本批没有溢出，长期路由也可能只有 token ID 专化而非
语义专化。两者必须分别记录。

## 4. 张量与数据契约

展平后 N 个 token、E 个专家、Top-k。理论 assignment 总数为 Nk。常用容量：

[
C=leftlceil gammarac{Nk}{E}
ight
ceil ,
]

其中 γ 是 capacity factor。selected_load[e] 统计 Top-k 选中专家 e 的数量；
accepted_load[e]=min(selected_load[e],C) 仅在简单 dropping 下成立；若按 gate 排序或有
reroute，必须从实际 dispatch 结果统计。dropped 是被拒绝的 assignment 数，不一定等于
完全失去 MoE 输出的 token 数。

输入和 gate 通常是 [N,k]，accepted mask 也是 [N,k] bool。dispatch 后每专家输入可表示为
[E,C,D] 的 padded dense tensor，或长度不同的列表/块稀疏布局。Padding token 不应参与
selected fraction、balance loss 或 dropping 分母。若一个 token 的全部 assignment 都被拒绝，
MoE 分支常返回零，由 Transformer residual 保留原 x；这不等于该 token 被从语言模型删除。

## 5. 公式推导与算法机制

Switch 风格辅助损失常写为

[
L_{bal}=Esum_{e=1}^{E} f_e P_e,
]

其中 f_e 是实际离散分派到专家 e 的 token/assignment 比例，P_e 是 router 概率在 batch 上的
平均值。若所有专家均衡，二者约为 1/E，损失约为 1。实现时 f 通常由 Top-k index 统计，
本身不可导；梯度主要通过 P 回到 router。该损失鼓励平均概率与实际负载一致，但不能保证每个
microbatch 满足容量，也不等于让专家学习不同知识。

选择容量内 assignment 也有语义。按 token 顺序截断容易产生 drop-towards-the-end：序列后部
更可能被拒绝。按 gate 从高到低保留可减少低置信 assignment，却需要排序，并可能偏向常见
token。随机选择增加方差。任何实现都要把策略写进契约和测试，不能只返回一个 dropped 数字。

Dropless 路线不设置固定 C 丢弃 token，而使用动态/块稀疏 grouped GEMM 处理不同 N_e。它减少
信息损失，但仍受显存峰值、最热门专家、跨设备 buffer 和通信不均限制；“dropless”不等于
“没有容量规划”。

## 6. 手算与数值示例

取 N=8、E=4、k=2、γ=1.25：

[
C=lceil1.25	imes 8	imes2/4
ceil=5.
]

总 assignment 为 16，总槽位为 20。若 selected load=[8,4,3,1]，accepted load 是
[5,4,3,1]，dropped=3，槽位利用率为 13/20=65%，assignment 接收率为 13/16=81.25%。
这两个百分比回答不同问题，不能混写。若有 3 个 assignment 被拒绝但来自 3 个仍有第二专家
被接收的 token，则“完全失去 MoE 分支”的 token 数可能为 0。

再看 γ=1.0，C=4，总槽位16恰等于 assignment 数，但负载不均仍会丢 4 个热门专家 assignment，
同时冷门专家留下空槽。由此可见总容量充足不代表逐专家容量充足。

## 7. 最小代码实现

~~~python
import math

def dispatch_capacity(loads, tokens, experts, top_k, factor):
    capacity = math.ceil(factor * tokens * top_k / experts)
    accepted = [min(load, capacity) for load in loads]
    dropped = sum(loads) - sum(accepted)
    assert sum(loads) == tokens * top_k
    return capacity, accepted, dropped

capacity, accepted, dropped = dispatch_capacity(
    loads=[8, 4, 3, 1],
    tokens=8,
    experts=4,
    top_k=2,
    factor=1.25,
)
assert (capacity, accepted, dropped) == (5, [5, 4, 3, 1], 3)
~~~

这段代码只处理计数，没有决定“哪些 token 被丢”。完整教学实现还要使用 gate 和 token index
生成 accepted mask，并在聚合时保证拒绝项不贡献输出。生产系统会进一步处理排序、buffer、
grouped GEMM 和通信。

## 8. 反例、常见误区与调试

最常见错误是把 C 写成 ceil(γN/E)，在 Top-2 时少算一半 assignment；第二个错误是先 drop
再计算 selected load，导致图上永远看不见拥堵；第三个错误是用 accepted fraction 替代辅助
损失中的路由统计，却没有说明公式变化。还要警惕用 padding 后 [E,C,D] 中的零行计算专家均值，
它会污染归一化或统计。

构造顺序偏置反例：让前半与后半 token 都选择专家0，按遍历顺序截断；比较每个序列位置的
drop rate。如果尾部显著更高，说明策略与 token 顺序耦合。OpenMoE 报告过路由较早固定以及
序列后部更容易被丢的现象，这类作者观察应当作为待复现实验，而不是课程中的普遍定律。

调试时先断言 selected 总和等于有效 token×k，再检查每专家 accepted≤C、dropped≥0、三者守恒，
最后检查全拒绝 token 的 residual 行为与 loss mask。

## 9. 主流工作与实现边界

GShard 和 Switch 使用容量因子与辅助均衡信号，让大规模稀疏专家适配静态编译和设备布局。
Expert Choice 固定每专家 bucket，由专家选择 token，从定义上更容易控制负载，但 token 接收
专家数可变。MegaBlocks 用 block-sparse 运算处理动态不规则专家批次，展示 dropless 训练路线；
其速度数字依赖 GPU kernel 和对比系统，CPU 列表循环不能复现。DeepSeek-V3 在路由分数中引入
动态 bias 做 auxiliary-loss-free balance，同时仍报告很小的 sequence-wise balance loss；
“无辅助损失”不能被简化成完全没有任何均衡信号。

课程本周实现 fixed-capacity dropping 与统计 oracle，不声称实现 MegaBlocks kernel、
expert-choice 或 DeepSeek 的完整训练系统。

## 10. 实验与 Notebook 对照

在[MoE 互动图](../interactive/core-concepts.html#moe)中固定同一批路由，依次改变 γ，先预测
dropped 和空槽。Notebook 中画 selected/accepted 双柱图。实验一验证容量公式；实验二比较
按顺序与按 gate 截断的位置 drop rate；实验三让所有 token 路由同一专家，观察 balance loss
梯度是否把平均概率推向其他专家；实验四模拟 dropless，只报告 N_e、峰值 load 与理想计算，
不伪造 GPU 加速结论。

## 11. 验收标准

- 对 N、E、k、γ 的合法组合，容量与手算一致；非法零值、负值、NaN 被拒绝。
- selected 总和等于有效 assignment，总是满足 accepted[e]≤C。
- dropped=sum(selected)-sum(accepted)，并单独报告完全拒绝 token 数。
- 复现 N=8 示例的 C=5、dropped=3 与两种利用率。
- 画出序列位置 drop rate，能解释截断策略偏差。
- 完成 starter 05 并运行 uv run llm-course exercises check 05。
- 在报告中明确 dropping、padded capacity 和 dropless 的算法/系统边界。

## 一手来源

- [GShard](https://arxiv.org/abs/2006.16668)：Top-2 gating、容量和自动分片。
- [Switch Transformer](https://arxiv.org/abs/2101.03961)：Top-1、capacity factor 与辅助均衡。
- [Expert Choice Routing](https://arxiv.org/abs/2202.09368)：固定 expert bucket 的替代路由。
- [MegaBlocks](https://arxiv.org/abs/2211.15841)：dropless block-sparse MoE 系统。
- [OpenMoE](https://arxiv.org/abs/2402.01739)：开放训练及路由/drop 分析。
- [DeepSeek-V3 技术报告](https://arxiv.org/abs/2412.19437)：auxiliary-loss-free balance 与系统协同。
