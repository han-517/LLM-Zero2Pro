# 阶段九：推理优化、服务评测与毕业项目

## 逐周讲义导航

> 本页是阶段知识地图，用于预习和复盘；完整推导、代码、反例、实验与验收请进入下面的逐周讲义。

- [第 45 周：权重量化与误差](../lessons/45_quantization.md)
- [第 46 周：PagedAttention 与推测解码](../lessons/46_paged_attention_continuous_batching.md)
- [第 47 周：基准设计与公平比较](../lessons/47_speculative_decoding_benchmarks.md)
- [第 48 周：毕业项目与知识答辩](../lessons/48_capstone_defense.md)

> 本章对应第 45–48 周。目标是建立从单算子正确性到在线服务 SLO 的完整测量链，
> 并清楚区分课程模拟器、算法 reference 与生产 runtime。

## 1. Prefill、Decode 与服务指标

Prefill 并行处理 prompt，通常计算密集；Decode 每步增加一个 token，持续读取权重
与 KV/state，常受内存带宽限制。指标必须从请求 arrival 开始，而不是从 GPU 真正
开始计算才计时：

- TTFT：`first_token_time - arrival_time`，包含排队；
- TPOT/ITL：首 token 后相邻输出 token 的平均间隔；只有 1 个输出 token 时无定义；
- E2E latency：`completion_time - arrival_time`；
- request throughput：请求数/观测时间；
- output-token throughput：输出 token/时间；
- total-token throughput：prompt + output token/时间；
- goodput：同时满足 TTFT/TPOT SLO 的请求数/时间。

`RequestTrace` 与 `summarize_serving()` 给出 mean、p50、p95、p99 和三种吞吐，避免
把不同分母的 “tokens/s” 混在一起。

## 2. 量化：先说清量化了什么

课程 `symmetric_quantize()` 是非空有限张量上的 per-tensor 对称 fake quant：

\[
s=\max|x|/q_{max},\quad q=\operatorname{clip}(\operatorname{round}(x/s)),
\quad \hat x=qs
\]

生产方案还要沿三条轴说明：

1. 对象：weight-only、weight + activation、KV Cache；
2. 网格：symmetric/asymmetric，per-tensor/channel/group/token；
3. 方法与硬件：calibration、outlier 处理、GPTQ/AWQ/SmoothQuant/KIVI、INT8/INT4/
   FP8/FP4，以及目标 GPU 是否有对应 kernel。

fake quant 只测重构误差，不会自动带来速度或内存收益。必须再测任务质量、实际峰值
内存、kernel 支持与端到端延迟。

## 3. PagedAttention：逻辑页、共享与 COW

连续 KV buffer 容易因不同序列长度产生预留浪费。分页把逻辑 token block 映射到
固定大小物理页，注意力公式不变。

课程 `PageTable` 是不存真实 K/V 的内存语义模拟器，支持：

- logical token 到 `(physical_page, offset)`；
- 物理页引用计数；
- prefix sharing；
- 对共享 partial page 追加时 copy-on-write；
- 唯一物理页的 internal fragmentation 与利用率。

它不是 PagedAttention kernel，也没有异步调度、GPU block table、换入换出或真实
K/V。生产 prefix cache 通常只缓存完整 block，并要考虑 hash、eviction、tenant
隔离与 cache salt。

## 4. Continuous batching 与 chunked prefill

静态 batching 等整批结束再接新请求，短请求会被长请求拖住。iteration-level/
continuous batching 在每个 decode iteration 加入或移除序列，提高利用率，但调度
策略会影响单请求延迟。长 Prefill 还可能阻塞 Decode；chunked prefill 把 prompt
切片，与 decode token 在预算内混合。

实验应画请求泳道并报告到达率、并发、prompt/output 长度分布、token budget、
TTFT/TPOT p95/p99 与 goodput，而不是只报告离线最大吞吐。

## 5. 推测解码：greedy 控制流与分布保持算法

`greedy_speculative_decode()` 只教学块验证：draft 猜 K 个 token，target 一次返回 K
个贪心预测，遇到首个不匹配就回退。它不保持随机采样分布。

`stochastic_speculative_decode()` 实现标准教学 reference。对候选 `x~q`：

\[
P(accept)=\min(1,p(x)/q(x))
\]

拒绝时从归一化的 `(p-q)_+` 采样；全部 K 个候选接受后，再从 target 的第 `K+1`
个分布采一个 bonus token。`target_distributions(prefix,candidates)` 必须在一次目标
调用返回 K+1 个分布。测试应包含全接受、首个拒绝、bonus 和 Monte Carlo 目标
分布一致性。Medusa、EAGLE、MTP 是不同的多候选/预测头路线，应单独标注。

## 6. 并行、解耦与容量规划

- replica/data parallel：复制模型服务更多请求；
- tensor parallel：拆单层矩阵，增加通信；
- pipeline parallel：拆层，需管理气泡与 microbatch；
- expert parallel：分布 MoE experts，受 all-to-all 影响；
- context/sequence parallel：拆长序列相关计算；
- prefill/decode disaggregation：为两个阶段分别配置计算型与带宽型资源。

选择策略前先确定模型是否单卡容纳、目标 TTFT/TPOT、请求长度分布和通信拓扑。
并行度更高不保证单请求更快。

## 7. 公平基准模板

报告必须包含：硬件与互联、runtime/kernel 版本、模型与量化格式、dtype、batch/
并发、prompt/output 长度分布、预热次数、观测窗口和采样策略。分开测模型加载、
Prefill、Decode、调度排队与 E2E；报告中位数及 p95/p99，并验证输出正确性。

建议实验顺序：

1. fake quant 误差与任务质量；
2. PageTable 共享/COW/碎片模拟；
3. greedy 与 stochastic speculative correctness；
4. static vs continuous batching 离散事件模拟；
5. 固定 SLO 下比较 throughput 与 goodput。

## 8. 毕业项目与边界

比较 Dense、注意力变体与 MoE 时统一 tokenizer、数据切分、训练 token 和评测；同时
报告总参数、活跃参数、估算 FLOPs、验证 loss、TTFT/TPOT、吞吐、内存和失败案例。
详细模板见 `learning/readings/references/capstone.md`。

课程量化、页表和推测解码都是可验证 reference；它们没有生产 runtime 的融合
kernel、网络服务、抢占、容错与多租户安全。最大上下文长度也不等于有效上下文能力。

## 一手来源与官方实现说明

- [PagedAttention / vLLM](https://arxiv.org/abs/2309.06180)
- [Speculative Decoding](https://arxiv.org/abs/2211.17192)
- [Orca iteration-level scheduling](https://www.usenix.org/conference/osdi22/presentation/yu)
- [Sarathi / chunked prefill](https://arxiv.org/abs/2308.16369)
- [DistServe / prefill-decode disaggregation](https://arxiv.org/abs/2401.09670)
- [GPTQ](https://arxiv.org/abs/2210.17323)、[SmoothQuant](https://arxiv.org/abs/2211.10438)、[AWQ](https://arxiv.org/abs/2306.00978)、[KIVI](https://arxiv.org/abs/2402.02750)
- [Medusa](https://arxiv.org/abs/2401.10774)、[EAGLE](https://arxiv.org/abs/2401.15077)
- [vLLM chunked prefill 文档](https://docs.vllm.ai/en/latest/configuration/optimization/)
- [vLLM prefix caching 设计](https://docs.vllm.ai/en/stable/design/prefix_caching/)
- [TensorRT-LLM 量化支持](https://nvidia.github.io/TensorRT-LLM/latest/features/quantization.html)
