# 第 39 周：专家并行、All-to-All 与现代 MoE 系统

> 课程定位：单机 for-loop 能验证路由公式，却没有展示大规模 MoE 的主要成本。本周把专家分布到
>多个设备，完整追踪 dispatch all-to-all、专家计算与 combine all-to-all，计算通信字节和负载矩阵，
>并把 DeepSeekMoE、dropless kernel、通信重叠等现代变体放回同一系统坐标系。

## 1. 学习目标

完成后应能为 P 个 rank 和 E 个专家指定 expert-parallel 映射；能从 Top-k index 构造
source-rank→destination-rank 的 send-count 矩阵；能解释 dispatch 与 combine 为什么通常各需一次
all-to-all；能估算激活、gate、index 的通信量；能区分专家负载不均、设备负载不均和链路拥塞；
能说明 tensor/data/pipeline/context/expert parallel 的正交与耦合；能批判模型报告中“活跃 FLOPs
低”却没有给端到端吞吐、网络拓扑和 batch 分布的结论。

## 2. 前置知识

需要掌握第 35–38 周的 assignment、容量、dropless 与 shared experts。建议先在
[MoE Notebook](../../labs/09_moe.ipynb) 保存一个 [N,k] 的 expert index，再完成
[starter 18](../../labs/starter/18_moe_systems.py) 中通信量估算。本周用 Python 列表模拟通信，
不需要多 GPU；但必须理解 rank、local expert 和 global expert ID 的区别。

## 3. 核心直觉

Dense tensor parallel 常让同一 token 的矩阵计算跨设备分片；expert parallel 则把不同专家放到
不同设备。Router 在源 rank 产生 global expert ID 后，token activation 必须送到拥有该专家的
destination rank。专家完成 MLP 后，结果还要送回源 rank，按原 token 和 slot 聚合。这像两次
邮件分拣：第一次把请求发给专家，第二次把答案寄回查询者。

稀疏计算没有消除通信，只把“执行所有专家”变成“重排并执行少数专家”。Decode batch 太小时，
每个 local expert 得到的 token 很少，GEMM 不饱和；负载过于集中时，最慢 rank 决定全组 step
时间。MoE 的系统效率因此强依赖 batch、sequence packing、网络层级和路由分布。

## 4. 张量与数据契约

设每个 rank 有 N_r 个有效 token，hidden size D，Top-k=k，元素字节 s。Router 输出
global_expert [N_r,k]。若 E 能被 P 整除，简单映射 destination=floor(expert/(E/P))，
local_expert=expert mod (E/P)。实际 dispatch 需要：

- send_counts[src,dst]：src 要发给 dst 的 assignment 数；
- packed_activations：[sum send_counts[src,:],D]；
- metadata：源 token index、slot、local expert、gate；
- recv buffer：按 destination 接收并按 local expert 分组。

所有 rank 的 send_counts 矩阵 shape 是 [P,P]，行和等于本 rank accepted assignment，列和是
目标 rank 接收负载。Combine 阶段沿反向路径返回 [assignment,D] 输出。若在 dispatch 前丢弃，
通信按 accepted 数计算；dropless 则按 selected 数计算，但 buffer 需要处理最大不均衡。

仅激活的双向通信粗略为

[
B_{act}approx 2,N_{accepted}D s,
]

前面的2表示发出输入与返回输出，不是 K/V。若统计整个集群，要说明是否把每条链路在发送端和
接收端重复计数。Gate、index、padding、collective protocol 和多跳网络是额外成本。

## 5. 公式推导与算法机制

令 A_{r,e} 是源 rank r 路由到专家 e 的 assignment 数，专家归属函数 owner(e)。通信矩阵：

[
C_{r,d}=sum_{e:owner(e)=d}A_{r,e}.
]

每个源 rank 计算时间受本地 token 与 router 影响，目标 rank 专家计算时间近似与列和及各 local
expert 的 grouped GEMM shape 有关。Step latency 不是平均负载，而接近最大设备关键路径：

[
t_{step}gtrsimmax_d(t_{recv,d}+t_{expert,d}+t_{sendback,d}).
]

因此 expert-level 均衡与 rank-level 均衡都重要。两个热门专家若恰好在同一 rank，即使专家负载
方差看起来一般，设备仍会成为瓶颈。生产系统可周期性重映射专家、复制热门专家或限制路由域；
这些操作会改变 checkpoint 和一致性约束。

通信重叠尝试把一部分 all-to-all 与其他计算并行，但“隐藏通信”需要存在足够独立计算、分块和
双向流水，不代表网络字节消失。DeepSeek-V3 报告 DualPipe、节点受限路由及通信/计算协同，
这些结论依赖其 H800 集群拓扑，不能从单机教学模拟外推。

## 6. 手算与数值示例

取 P=2、E=4，每 rank 两个 experts：rank0 拥有0/1，rank1拥有2/3。rank0 有4个 token、Top-2，
其 expert indices 为 [[0,2],[1,2],[2,3],[0,3]]。发送到 rank0 的 assignment 有专家0两次、
专家1一次，共3；发送到 rank1 有专家2三次、专家3两次，共5，所以 rank0 的 send row=[3,5]。

若 rank1 自己的 send row=[4,4]，全局 C=[[3,5],[4,4]]。目标 rank0 接收7个，rank1接收9个。
D=4096、BF16 s=2，只算 rank0 本地8个 accepted assignment的双向 activation 通信约
2×8×4096×2=131072 bytes；其中发给自己是否走网络由实现决定。真实 collective 还包含其他
rank、协议和 metadata，不能把这个数当实测带宽。

## 7. 最小代码实现

~~~python
def send_counts(expert_indices, experts, ranks):
    assert experts % ranks == 0
    experts_per_rank = experts // ranks
    counts = [0] * ranks
    for token_slots in expert_indices:
        for expert in token_slots:
            destination = expert // experts_per_rank
            counts[destination] += 1
    return counts

indices_rank0 = [[0, 2], [1, 2], [2, 3], [0, 3]]
assert send_counts(indices_rank0, experts=4, ranks=2) == [3, 5]

def activation_bytes(assignments, hidden, bytes_per_element):
    return 2 * assignments * hidden * bytes_per_element

assert activation_bytes(8, 4096, 2) == 131072
~~~

代码只计算路由账本，不执行 collective。生产实现还需要稳定排序、异步 stream、反向通信、
autograd、失败恢复、拓扑感知和 fused grouped GEMM。

## 8. 反例、常见误区与调试

误区一是认为 k 个专家都在本地，因此通信量只乘1；专家并行的目的恰是让大部分专家分布在其他
rank。误区二是只报告 send_counts 行和，它永远等于本地 assignment，无法看目标拥塞；必须看
完整矩阵列和。误区三是用 all-reduce 解释 all-to-all：all-reduce 对相同 shape 做聚合，all-to-all
交换不同 token 分片，语义和 buffer 都不同。误区四是看到低 GPU utilization 就直接增加专家数；
小 expert GEMM、网络等待和负载不均都可能恶化。

调试先用 global token ID 和 source rank 给每个 assignment 打标签，dispatch 后检查多重集合守恒，
专家输出附加 local expert 标记，再 combine 检查回到正确 token/slot。随后比较 C 的行列和、
最大/平均负载、空专家和 dropped。最后才分析时间线，否则数值错误会伪装成通信问题。

## 9. 主流工作与实现边界

GShard 把条件计算与 XLA 自动分片结合，证明超大规模多语模型可训练。Tutel、DeepSpeed-MoE、
MegaBlocks 和 Megatron-Core 分别从 runtime、并行组、block-sparse kernel 与训练框架完善专家
并行。DeepSeekMoE 改变专家粒度和共享结构；DeepSeek-V3 进一步报告 auxiliary-loss-free balance、
节点受限路由、FP8 与通信重叠。OLMoE 的开放代码与日志提供不同硬件/规模的可检查基线。

这些工作解决层次不同：路由算法、专家结构、kernel、collective 和集群调度不能混成一个“MoE
优化”。本课程只模拟 global/local expert 映射、通信矩阵和字节账，不声称复现 NCCL all-to-all
或论文吞吐。

## 10. 实验与 Notebook 对照

实验一用两 rank 手算 C 并与代码对齐；实验二固定专家负载，改变 expert→rank 映射，观察最大
rank load；实验三扫描 N、D、k、dtype，画 activation bytes 与专家 FLOPs；实验四比较 dropping
前后通信量和质量风险；实验五画 dispatch→expert→combine 时间线，标出可重叠与依赖边。再把
[互动 MoE 图](../interactive/core-concepts.html#moe)中的 selected/accepted load 映射到设备。

## 11. 验收标准

- 正确构造 [P,P] send-count 矩阵，行列和与 assignment 守恒。
- 能从 global expert ID 得到 owner rank 与 local expert ID，并拒绝不可整除配置。
- 手算示例得到 rank0 send row=[3,5] 与131072 bytes。
- 同时报告平均、最大 rank load 和 expert load，不只报告全局均值。
- 画出两次 all-to-all，并解释 combine 如何恢复原 token 顺序。
- 区分算法、kernel、collective、拓扑四个层次的收益证据。
- 完成 starter 18 并运行 uv run llm-course exercises check 18。

## 一手来源

- [GShard](https://arxiv.org/abs/2006.16668)：条件计算与自动分片。
- [MegaBlocks](https://arxiv.org/abs/2211.15841)：block-sparse dropless MoE kernel。
- [DeepSpeed-MoE](https://arxiv.org/abs/2201.05596)：MoE 训练与推理系统。
- [DeepSeekMoE](https://arxiv.org/abs/2401.06066)：细粒度/共享专家结构。
- [DeepSeek-V3 技术报告](https://arxiv.org/abs/2412.19437)：路由、FP8 与通信计算协同。
- [Megatron-Core MoE 文档](https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/moe.html)：官方并行与路由接口。
- [OLMoE](https://arxiv.org/abs/2409.02060)：全开放 MoE 训练与分析。
