# 第 47 周：推测解码、分布校正与公平服务基准

> 课程定位：分页与连续批处理提高并发资源利用率，本周转向单条自回归链的串行瓶颈。小 draft 模型先提议多个 token，大 target 模型并行验证；随机解码必须用修正拒绝采样保持 target 分布。随后把“快”写成可复现的 TTFT、TPOT、吞吐、goodput 与质量契约，而不是一张缺少条件的速度截图。

## 1. 学习目标

完成后应能区分贪心推测与随机推测；能推导 `min(1,p/q)` 接受率及拒绝后的 `[p-q]_+` 修正分布；能解释 draft=target 时为何全部接受、draft 很差时为何可能变慢；能计算 acceptance length 与一次 target verify 的收益；能定义 latency、TTFT、TPOT、ITL、throughput 和 goodput；能设计只改变推测解码开关的配对基准，并保留配置、原始请求级结果和环境证据。

## 2. 前置知识

需要掌握 categorical sampling、softmax、自回归 decode、KV cache 和第46周调度。建议先在 [推理 Notebook](../../notebooks/core/11_inference_serving.ipynb) 看 decode loop，并用 [服务互动图](../interactive/serving-lab.html) 区分 prefill 与 decode。无需训练真正的 draft；二分类数值例足以核对“输出分布不变”，但真实速度必须在支持推测解码的服务引擎与目标硬件上测量。

## 3. 核心直觉

target 对一段候选做一次并行 forward，通常比逐 token 调 target 多次更能利用硬件；draft 的价值不是更准确地替换 target，而是廉价提出 target 多半会接受的连续 token。接受越长，target 每次调用摊销的输出 token 越多；然而 draft forward、verification、缓存管理与拒绝回滚都有成本，batch 很大或 draft 太慢时未必获益。

随机解码不能只比较 argmax。若直接接受 draft 样本，输出将服从 q 而不是目标 p。修正拒绝采样把 q 与 p 重叠的概率质量作为“接受”，拒绝时只从 p 尚未覆盖的剩余质量抽样，从而在精确算术下恢复 p。

## 4. 张量与数据契约

对 gamma 个 draft 位置，draft 概率 `q` shape 为 `[gamma,V]`，target 对候选及下一个位置的概率 `p` 至少为 `[gamma+1,V]`；每行非负、有限且和为1。候选 token 为 `[gamma]` int，均匀随机数为 `[gamma]`。当 `q(x)=0` 时 x 不会由 q 抽到；实现仍需避免除零。拒绝位置后的 draft token 与相关 KV 必须丢弃或回滚。

基准原始记录每请求至少包含 arrival、first_token、finish、input_tokens、output_tokens、成功标志。`TTFT=first_token-arrival`；输出多于1 token 时 `TPOT=(finish-first_token)/(output_tokens-1)`；ITL 是相邻输出 token 间隔。全局 output throughput 是成功输出 token 总数除以墙钟区间；goodput 只计满足预注册 SLO 的请求或 token。所有时间统一为秒或毫秒，聚合时标明 mean、median、p95/p99。

## 5. 公式推导与算法机制

一位置上 draft 先采 `x~q`，以 `a(x)=min(1,p(x)/q(x))` 接受。输出 x 的接受质量为 `q(x)a(x)=min(q(x),p(x))`。总拒绝概率是 `R=1-sum_x min(p(x),q(x))=sum_x max(p(x)-q(x),0)`。拒绝后从 `r(x)=max(p(x)-q(x),0)/R` 抽样，因此最终概率为

`min(q(x),p(x)) + R*r(x) = p(x)`。

若 R=0，则 p=q，不需要修正抽样。多 token 算法从左到右验证；首次拒绝后修正一个 token，并丢弃其后的提议；若 gamma 个全接受，再从 target 的额外位置采一个 token。贪心模式可以比较 draft token 与 target argmax，但“输出相同”还要求 tokenizer、停止条件、数值和 tie-breaking 一致。

成本不能只数 target 调用。令每轮接受 A 个 draft token并额外产生一个 target token，粗略摊销收益与 `(A+1)/(C_target_verify+C_draft+C_overhead)` 有关。verify 并非固定成本，gamma、batch、上下文与 kernel 都会改变它；所以 acceptance rate 高不是速度充分条件。

## 6. 手算与数值示例

词表只有 A/B，target `p=[0.6,0.4]`，draft `q=[0.8,0.2]`。draft 抽到 A 时接受率 `0.6/0.8=0.75`；抽到 B 时接受率1。接受质量分别为0.6和0.2，总拒绝概率0.2。剩余 `[p-q]_+=[0,0.2]`，归一化后拒绝必采 B。因此最终 A 概率为 `0.8×0.75=0.6`，B 概率为 `0.2×1+0.8×0.25=0.4`，恰好恢复 p。

某请求 arrival=0 ms、首 token=120 ms、共输出5 token、finish=200 ms，则 TTFT=120 ms，TPOT=`(200-120)/(5-1)=20 ms`。把总延迟200/5写成40 ms/token 会混合 prefill 与 decode，无法判断推测解码改善了哪一段。

## 7. 最小代码实现

~~~python
def corrected_one_step_distribution(p, q):
    if len(p) != len(q) or not p:
        raise ValueError("same non-empty vocabulary required")
    if any(x < 0 for x in p + q):
        raise ValueError("negative probability")
    if abs(sum(p) - 1.0) > 1e-9 or abs(sum(q) - 1.0) > 1e-9:
        raise ValueError("rows must sum to one")
    accepted = [min(px, qx) for px, qx in zip(p, q)]
    residual = [max(px - qx, 0.0) for px, qx in zip(p, q)]
    reject_prob = sum(residual)
    if reject_prob == 0.0:
        return accepted
    correction = [reject_prob * x / reject_prob for x in residual]
    return [a + c for a, c in zip(accepted, correction)]

p = [0.6, 0.4]
q = [0.8, 0.2]
out = corrected_one_step_distribution(p, q)
assert all(abs(a - b) < 1e-12 for a, b in zip(out, p))

def request_metrics(arrival_ms, first_ms, finish_ms, output_tokens):
    if output_tokens < 1 or not (arrival_ms <= first_ms <= finish_ms):
        raise ValueError("invalid trace")
    ttft = first_ms - arrival_ms
    tpot = 0.0 if output_tokens == 1 else (finish_ms - first_ms) / (output_tokens - 1)
    return ttft, tpot

assert request_metrics(0, 120, 200, 5) == (120, 20.0)
~~~

代码用概率质量代数验证单位置正确性，没有真正采样、连续验证、target forward、KV 回滚或浮点误差测试。生产实现还要处理 top-k/top-p/temperature、EOS、动态 gamma、draft/target tokenizer 兼容、并行调度和统计检验。

## 8. 反例、常见误区与调试

最危险的反例是“拒绝后直接从 p 重采”：已接受质量与重采质量叠加后一般不再是 p。另一个错误是用 `p/q` 不裁到1；概率会越界。只比较一次生成文本也不能证明分布正确，应在小词表做枚举或大量采样检验。draft 与 target tokenization 不兼容会使候选无法逐 token 对齐。

基准误区包括：baseline 用 FP16、实验用量化；不同 prompt/output 长度或停止条件；baseline 贪心、实验随机；只预热一方；把离线 batch 吞吐与在线泊松到达延迟比较；只报平均值隐藏尾延迟；挑选最快一次。调试先验证同 seed/贪心下输出与 token 数，再记录接受长度、拒绝位置和 target/draft 调用；随后固定环境做交替配对运行，保存每请求原始 trace，最后才聚合分位数。

## 9. 主流工作与实现边界

Leviathan 等与 Chen 等在相近时期独立给出保持目标分布的推测解码/采样。后续方法用 n-gram、轻量 head、Medusa/EAGLE 类树或特征预测提高提议效率；当前 vLLM 官方实现支持多种 proposer，但具体模型、backend、并行与采样兼容性随版本变化。HELM 强调多指标、标准化与透明报告；MLPerf 通过场景、质量约束和提交规则约束系统比较。

课程 baseline 只验证一位置残差公式和服务指标账本，不等于完整论文算法，更不是生产服务压测器。论文中的2–3倍收益属于指定模型与硬件；对高并发、短输出或高效 target，draft 开销可能抵消收益。

## 10. 实验与 Notebook 对照

在 [推理 Notebook](../../notebooks/core/11_inference_serving.ipynb) 实现块式候选/验证；在 [服务互动图](../interactive/serving-lab.html) 改变到达率和输出长度；在 [starter 20](../../exercises/starter/20_inference_systems.py) 完成教学版推测解码。先做二词表枚举，再扫描 gamma 和 draft 匹配度，用显式成本模型画预计 speedup。公平基准预注册 target/draft、commit、硬件、dtype、量化、prompt 分布、输出长度、temperature、seed、并发/到达率、预热、重复次数和 SLO；baseline 与实验交替运行，报告 TTFT/TPOT/吞吐/goodput、接受长度及质量一致性。

## 11. 验收标准

- 二词表示例严格恢复 `[0.6,0.4]`，p=q 时拒绝概率为0，概率行均归一化。
- 能解释首次拒绝后的修正与 KV 回滚，不能用“target 再算一遍”替代分布证明。
- 正确计算5-token示例的 TTFT=120 ms、TPOT=20 ms。
- 提交含原始 trace、完整配置、环境/commit、预热和至少三次重复的配对基准。
- 同时报 median 与 p95/p99、throughput 与 SLO-goodput，并验证输出质量/分布契约。
- 完成 starter 20 并运行 `uv run llm-course exercises check 20`。

## 一手来源

- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)：精确推测解码原始工作。
- [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318)：修正拒绝采样与 Chinchilla 实验。
- [vLLM 官方推测解码文档](https://docs.vllm.ai/en/latest/features/spec_decode.html)：当前引擎配置和兼容边界。
- [vLLM 官方 serving benchmark 实现](https://docs.vllm.ai/en/latest/api/vllm/benchmarks/serve/)：TTFT、TPOT 与 goodput 的计算口径。
- [HELM 原论文](https://arxiv.org/abs/2211.09110)与[MLPerf Inference 原论文](https://arxiv.org/abs/1911.02549)：透明、多指标与受约束系统评测。
