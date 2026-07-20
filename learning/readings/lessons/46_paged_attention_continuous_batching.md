# 第 46 周：PagedAttention、连续批处理与 KV 页账本

> 课程定位：量化缩小了权重，但长上下文和高并发服务常被 KV cache 限制。本周借鉴虚拟内存，把每条请求的逻辑 token 位置映射到非连续物理页，并用迭代级调度让请求逐步加入或离开 batch。代码是离线页表模拟器，不包含 GPU attention kernel 或真实异步服务器。

## 1. 学习目标

完成后应能从层数、KV 头数、head dim、dtype 和 token 数计算 KV 字节；能画出逻辑块、物理块与 block table；能区分内部碎片、外部碎片、共享前缀与 copy-on-write；能解释静态 batching、请求级 dynamic batching 和连续 batching 的差别；能模拟 allocate、append、free 与每轮 decode 调度；能说明 PagedAttention 是内存管理/attention 寻址设计，不等同于把磁盘分页直接搬到 GPU。

## 2. 前置知识

需要理解自回归 prefill/decode、每层 K/V shape、因果 attention 和 KV cache。先复习 [推理 Notebook](../../labs/11_inference_serving.ipynb) 中缓存预算，再打开 [服务互动图](../interactive/serving-lab.html) 观察上下文长度与并发。无需掌握 CUDA page table；本周的“page”是服务引擎管理的固定 token block，不是操作系统硬件页错误机制。

## 3. 核心直觉

若每条请求预留最大上下文长度，短请求会占着大片空槽；若要求每条 KV 在物理上连续，增长时容易找不到足够大的连续空间。分页把逻辑序列切成固定 B-token block，物理页可以散落，block table 负责翻译。请求每增加 B 个 token 才申请新页，释放时按页归还，因此外部碎片大幅减少，单请求内部浪费上界小于一个 block。

连续批处理把调度边界从“整条请求结束”缩到“一个模型迭代结束”：每一轮选择当前可运行序列，各生成一个或一小段 token，完成的立即移出，新请求随容量加入。它提高设备利用率，却也引入队列、公平性、抢占和 prefill/decode 干扰；吞吐增加不保证每个请求的 TTFT 都下降。

## 4. 张量与数据契约

设层数 L、每层 KV heads 为 Hkv、head dim D、每元素 s bytes。单 token KV 字节为 `2*L*Hkv*D*s`，2 表示 K 和 V。物理 cache 可抽象成 `[num_pages, 2, L, Hkv, B, D]`，实际 kernel 常重排维度以利向量化。逻辑 block table 为 `[max_sequences,max_blocks]` 的整数表，未分配项用明确 sentinel；slot mapping 把 `(request_id,logical_token)` 变为 `(physical_page,offset)`。

每个请求状态至少含 prompt length、已生成数、最大生成数、block table、到达时间和 finished 标志。页引用计数必须是非负整数：普通请求独占时为1，前缀共享时可大于1；写入共享页前需 copy-on-write。非法边界包括 B≤0、重复释放、页耗尽却静默覆盖、block table 越界，以及请求结束后仍持有引用。

## 5. 公式推导与算法机制

长度 t 的序列需要 `ceil(t/B)` 页，分配容量为 `B*ceil(t/B)`，内部浪费为该容量减 t，故单序列浪费在 `[0,B-1]`。N 条活跃序列的浪费小于 `N(B-1)` token slots；这是容量上界，不等于 GPU 实测显存，因为 allocator、metadata、workspace 与对齐仍有开销。

每轮调度先处理完成/取消并释放页，再根据 token budget、sequence budget 和空闲页准入请求。Prefill 一次可能消耗许多 token，decode 通常每序列一 token；若不设预算，长 prompt 会阻塞 decode。抢占可采用 swap、recompute 或拒绝新请求，各自改变延迟与带宽。attention kernel 读取逻辑位置 i 时，通过 `page=table[i//B]`、`offset=i%B` 找 K/V；非连续只改变寻址，不改变因果 softmax 的数学结果。

连续 batching 的吞吐来自减少 padding 和空泡；PagedAttention 让调度器有更灵活的可用 KV 容量。二者协同但不是同一算法：即使页表完美，错误的调度仍会饿死请求；即使连续调度，连续预分配仍可能因碎片限制并发。

## 6. 手算与数值示例

令 B=4，物理空闲页 `[7,2,9,5]`。请求 A 长5，需要两页，可映射逻辑块0→7、块1→2；请求 B 长3，需要一页，逻辑块0→9。A 的 token4位于页2 offset0，说明逻辑相邻的 token3与token4可在不同物理页。两请求共使用8个 token，分配12个槽，内部浪费4；若旧系统为每请求固定预留8槽，则占16槽、浪费8。

若 L=24、Hkv=8、D=128、BF16 s=2，单 token KV 为 `2×24×8×128×2=98304` bytes，即96 KiB。B=16 的一页为1.5 MiB；4096个活跃 token 仅理论 KV 已约384 MiB。这里没有计权重、激活和 allocator，不能据此把显存填满到100%。

## 7. 最小代码实现

~~~python
class PagePool:
    def __init__(self, num_pages, block_size):
        if num_pages <= 0 or block_size <= 0:
            raise ValueError("positive capacity required")
        self.block_size = block_size
        self.free = list(range(num_pages))
        self.tables = {}

    def append_to_length(self, request, length):
        if length < 0:
            raise ValueError("negative length")
        table = self.tables.setdefault(request, [])
        need = (length + self.block_size - 1) // self.block_size
        while len(table) < need:
            if not self.free:
                raise MemoryError("KV pages exhausted")
            table.append(self.free.pop(0))
        return list(table)

    def slot(self, request, token_index):
        if token_index < 0:
            raise IndexError("negative token")
        table = self.tables[request]
        block, offset = divmod(token_index, self.block_size)
        return table[block], offset

    def release(self, request):
        pages = self.tables.pop(request)
        self.free.extend(pages)

pool = PagePool(num_pages=4, block_size=4)
assert pool.append_to_length("A", 5) == [0, 1]
assert pool.slot("A", 4) == (1, 0)
assert pool.append_to_length("B", 3) == [2]
pool.release("A")
assert sorted(pool.free) == [0, 1, 3]
~~~

模拟器只维护独占页和逻辑翻译；它省略真实 K/V tensor、引用计数、prefix cache、copy-on-write、GPU block copy、并发锁、抢占、chunked prefill 与调度优先级。因此它能验证页守恒，不能复现 vLLM 的 kernel、延迟或论文吞吐。

## 8. 反例、常见误区与调试

误区一是把 block size 越小越好：内部浪费下降，但 block table、调度和寻址开销会上升。误区二是只看“空闲字节”却不检查可用页和预算。误区三是释放共享前缀时直接归还物理页，会让其他请求读到被覆盖的 KV。误区四是把 dynamic batching 当连续 batching；若新请求仍须等待整个旧 batch 结束，就不是迭代级调度。

调试先给每个物理页唯一 poison 值，验证 `(logical block,offset)` 翻译；再检查已分配页与 free list 不相交、并集等于总页、无重复页。用 B-1、B、B+1 长度测试边界；取消请求后检查全部引用回收。调度层记录每轮 admitted/running/preempted/finished 与 token budget，遇到高尾延迟先查队列与长 prefill，而不是只看平均 GPU utilization。

## 9. 主流工作与实现边界

Orca 提出迭代级调度和 selective batching，解决不同请求生成长度不一致造成的空泡；vLLM 的 PagedAttention 将 KV 切成 block 并配合调度、共享和 copy-on-write，论文在其指定模型、硬件和工作负载上报告相对基线的吞吐收益。今天的生产引擎还加入 chunked prefill、prefix caching、优先级、KV offload 和分离式 prefill/decode，但接口和支持矩阵会随版本改变。

本课程不把论文的倍数写成普遍保证。结果取决于输入/输出长度、到达过程、batch、模型、GPU、dtype、网络和 SLO。教学模拟只证明页表不变量与调度直觉，不执行 block-aware attention，也没有证明与未分页 attention 数值等价。

## 10. 实验与 Notebook 对照

在 [推理 Notebook](../../labs/11_inference_serving.ipynb) 先手算单 token KV 字节；到 [服务互动图](../interactive/serving-lab.html) 改变并发、上下文和 page size；在 [starter 20](../../labs/starter/20_inference_systems.py) 完成页分配与连续调度。实验一随机生成序列长度，比较固定最大长度、连续分配与分页的浪费；实验二扫描 B∈{4,8,16,32}，画内部浪费和页表项；实验三模拟长 prompt 与短 decode 混合到达，比较请求级 batching 与每轮准入的完成时间。运行前必须写下预测及公平策略。

## 11. 验收标准

- 对任意 t≥0，页数等于 `ceil(t/B)`，slot 翻译覆盖 `[0,t)` 且不越界。
- allocate/release 后空闲页与已用页无交集、无重复、数量守恒；页耗尽明确报错。
- 手算 A/B 映射、12个分配槽和4个浪费槽正确，KV 字节得到98304/token。
- 能画出一次 prefill、三轮 decode 中新请求加入和完成请求退出的时间线。
- 同时报告平均与 p95/p99 TTFT/TPOT，不用吞吐单指标宣称调度全面更优。
- 完成 starter 20 并运行 `uv run llm-course exercises check 20`。

## 一手来源

- [PagedAttention/vLLM 原论文](https://arxiv.org/abs/2309.06180)：block table、共享与服务评测。
- [vLLM 官方仓库](https://github.com/vllm-project/vllm)：当前实现与功能边界。
- [Orca OSDI 2022 论文页面](https://www.usenix.org/conference/osdi22/presentation/yu)：迭代级调度与 selective batching。
- [vLLM 官方引擎架构文档](https://docs.vllm.ai/en/latest/design/arch_overview.html)：调度器、KV cache manager 与执行器职责。
