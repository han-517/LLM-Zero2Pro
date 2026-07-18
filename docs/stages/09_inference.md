# 阶段九：推理优化、评测与毕业项目

## Prefill 与 Decode

Prefill 并行处理 prompt，通常计算密集；Decode 每步生成一个 token，需要读取大量权重和缓存，常受内存带宽限制。报告性能时至少区分：

- TTFT：首 token 延迟。
- TPOT：后续每 token 时间。
- Throughput：单位时间所有请求生成 token 数。
- Latency：单个请求端到端耗时。

## 量化

对称 per-tensor 量化用一个 scale 把浮点映射到整数：

```text
scale = max(abs(x)) / qmax
q = clamp(round(x / scale), -qmax, qmax)
x_hat = q * scale
```

教学实验比较 int8、int4 的重构误差和理论体积。真实速度还依赖硬件是否有对应低比特内核，文件更小不自动等于运行更快。

## PagedAttention

传统连续 KV Cache 需要预留大块空间，容易碎片化。PagedAttention 像操作系统分页：逻辑序列的 KV 分散在固定大小物理块，通过页表找到。核心价值是内存管理和共享，而不是改变注意力公式。

## 推测解码

小 draft 模型先猜多个 token，大 target 模型一次验证整个候选块。教学 API
`target_verify(prefix, candidates)` 必须一次返回每个候选位置的 target token；若在 Python 循环中逐 token 调 target，就没有减少串行 target 调用。若猜得准，可以用一次 target 前向接受多个 token；若不准，按校正规则拒绝并采样。
严格随机算法保持 target 分布不变；课程贪心版只教学块验证控制流，再阅读随机采样版证明。

## 公平基准

- 预热后重复多次，报告中位数和波动。
- 固定硬件、dtype、batch、序列长度和生成长度。
- 分开测模型加载、Prefill 和 Decode。
- 同时检查输出正确性；错误但快的实现没有意义。

## 毕业项目

比较 Dense、注意力变体和 MoE。统一 tokenizer、数据切分、训练 token 和评测；同时报告总参数、活跃参数、估算 FLOPs、验证 loss、速度、内存和失败案例。详细模板见 `docs/03_capstone.md`。

## 常见误区

- 最大上下文长度不是有效上下文能力。
- tokens/s 不注明 batch 和硬件没有可比性。
- 量化误差小不保证任务准确率完全不变。
- 近似活跃 FLOPs 匹配仍不能消除所有架构差异，报告中必须说明。

