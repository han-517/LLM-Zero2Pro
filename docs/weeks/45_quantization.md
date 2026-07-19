# 第 45 周：权重量化、误差预算与真实加速边界

> 课程定位：前面的课程已经能训练并生成一个小型 Transformer；本周开始回答“模型怎样装得下、跑得动”。我们先把浮点权重映射到有限整数网格，再区分数值误差、存储体积和端到端延迟三种证据。教学代码只做 CPU 伪量化，不宣称复现低比特打包或 GPU kernel。

## 1. 学习目标

完成本周后，你应能独立写出对称 per-tensor 与分组伪量化，解释 scale、zero-point、group size、舍入和饱和；能从参数量计算权重与 scale 的理论体积；能用最大绝对误差、MSE 和下游损失评估量化，而不是只看文件变小；能区分 weight-only、weight-activation 与 KV-cache 量化；能说明 GPTQ、AWQ、SmoothQuant 各自解决的误差来源，以及“4 bit”为什么不自动等于“四倍加速”。

## 2. 前置知识

需要会读线性层 `y = x @ W.T`，理解浮点数、整数范围、均方误差与广播。建议先复习现代解码器中权重矩阵的 shape，并打开 [推理 Notebook](../../notebooks/core/11_inference_serving.ipynb)。不要求预先掌握 Hessian、CUDA 或量化感知训练；二阶补偿和硬件 kernel 会在“主流工作”中给出边界。

## 3. 核心直觉

量化是“选择网格并支付误差”。浮点数先除以网格间距 `s`，舍入到整数格点，再裁剪到可表示范围。网格太稀，舍入误差大；范围太窄，离群值会饱和；若让一个离群值决定整层的 `s`，多数小权重只占少量格点。更细的 per-channel/per-group scale 能贴合局部分布，却增加元数据和 kernel 复杂度。

必须把三层结论分开：伪量化仅说明“重建后的数值怎样”；位打包才决定 checkpoint 体积；支持该格式的 fused kernel 才可能带来速度和显存带宽收益。把整数马上反量化成 Python float 做矩阵乘，本质仍是浮点计算，不能作为 INT4 吞吐证据。

## 4. 张量与数据契约

设线性层权重 `W` 的 shape 为 `[d_out, d_in]`，教学输入使用有限的 Python `float`。有符号 b-bit 对称量化取 `qmin = -2^(b-1)+1`、`qmax = 2^(b-1)-1`，保留对称零点；例如 INT8 为 `[-127,127]`，INT4 为 `[-7,7]`。`scale` 必须为正，整组全零时约定为 `1.0`，避免除零且保持重建为零。

per-tensor 只有一个 scale；per-output-channel 可用 `[d_out,1]`；沿输入维按 G 分组时，scale 逻辑 shape 为 `[d_out, ceil(d_in/G)]`。最后一组允许短于 G，但不得遗漏元素。量化整数必须落在范围内，重建值与 W 同 shape。生产格式还必须声明 scale dtype、整数打包顺序、group axis、对称性、是否含 bias，以及算子支持的激活 dtype；只写“W4A16”仍不足以复现。

## 5. 公式推导与算法机制

对称量化对一组权重取 `a = max_i |w_i|`。若 `a > 0`，令

`s = a / qmax`, `q_i = clip(round(w_i / s), qmin, qmax)`, `w_hat_i = s q_i`。

未裁剪时，最近邻舍入给出 `|w_i-w_hat_i| <= s/2`；但这只是单元素界，线性层输出误差还受激活方向和误差相关性影响。非对称量化用 `w_hat=s(q-z)`，zero-point z 让整数范围覆盖 `[w_min,w_max]`，适合明显偏移的分布；对大致以零为中心的权重，对称格式更简单，也更常利于 kernel。

若共有 N 个权重、每个 b bit、M 个 scale、scale 使用 16 bit，理想存储为 `Nb/8 + 2M` bytes，另加 header、padding 和 zero-point。这个公式不是进程峰值显存：加载时临时 buffer、反量化 workspace、KV cache 与 allocator 都可能占用更多。误差至少报告 `max_abs=max|W-W_hat|` 和 `MSE=mean((W-W_hat)^2)`，最终还要比较固定数据集上的 loss 或任务指标。

## 6. 手算与数值示例

对 `[-1.2,-0.3,0.2,1.0]` 做 INT4 对称量化，`qmax=7`，所以 `s=1.2/7≈0.17143`。舍入后 q 为 `[-7,-2,1,6]`，重建为 `[-1.2,-0.34286,0.17143,1.02857]`。四个绝对误差约为 `[0,0.04286,0.02857,0.02857]`，最大误差约 `0.04286`，不超过 `s/2≈0.08571`。

若 1024×1024 权重使用 INT4、group size 128、每组一个 FP16 scale，理想权重数据为 524288 bytes，scale 数为 8192、占 16384 bytes，总计 540672 bytes；不能直接写成“恰好 0.5 MiB”。若框架实际仍按一个 byte 保存每个 INT4 值，体积会再大一倍，这正是必须检查物理格式的原因。

## 7. 最小代码实现

~~~python
from math import isfinite

def symmetric_fake_quant(values, bits=4):
    if bits < 2:
        raise ValueError("bits must be >= 2")
    if not values or not all(isfinite(x) for x in values):
        raise ValueError("values must be finite and non-empty")
    qmax = (1 << (bits - 1)) - 1
    qmin = -qmax
    max_abs = max(abs(x) for x in values)
    scale = max_abs / qmax if max_abs else 1.0
    q = [max(qmin, min(qmax, round(x / scale))) for x in values]
    restored = [scale * x for x in q]
    mse = sum((a - b) ** 2 for a, b in zip(values, restored)) / len(values)
    return q, restored, scale, mse

w = [-1.2, -0.3, 0.2, 1.0]
q, restored, scale, mse = symmetric_fake_quant(w, bits=4)
assert q == [-7, -2, 1, 6]
assert abs(scale - 1.2 / 7) < 1e-12
assert max(abs(a - b) for a, b in zip(w, restored)) <= scale / 2 + 1e-12
assert 0.0 < mse < 0.002
~~~

这段代码故意不返回压缩文件，也不调用整数 GEMM：它只验证量化网格、饱和与误差。完整实现还需分组 axis、校准集、bit packing、格式序列化、量化线性 kernel、不同硬件后端和端到端精度回归。

## 8. 反例、常见误区与调试

反例一：把 `scale=max_abs/2^b`，会浪费或越过整数端点；先打印 qmin/qmax，再手算最大值应映射到哪一格。反例二：全零组得到 scale=0，后续除零；必须定义全零约定。反例三：先 cast 成整数再计算 scale，小数信息已丢失。反例四：只报告 MSE 很小，却没检查模型 loss；少数敏感通道仍可能破坏注意力或输出头。

调试顺序是：检查有限值和 shape；确认 group 覆盖且最后一组不丢元素；检查所有 q 在范围内；用全零、端点、离群值和非整除 group size 做单测；再测层输出、模型 loss、checkpoint 实际字节和进程峰值。若速度变慢，先确认是否真的走低比特 kernel、是否发生运行时反量化、batch/shape 是否被 kernel 支持，而不是继续调 scale。

## 9. 主流工作与实现边界

GPTQ 使用近似二阶信息逐列量化并补偿后续权重，目标是高精度的 weight-only PTQ；AWQ 根据激活统计保护少量显著权重，并把缩放与部署 kernel 协同；SmoothQuant 把难量化的激活幅值平滑迁移到权重，服务于 W8A8；LLM.int8() 对离群特征采用混合精度路径。这些名字不能互换：校准数据、目标位宽、激活是否量化和硬件算子都不同。

现代推理框架还支持 FP8、KV-cache 量化和多种 INT4 格式，但“支持”随模型、GPU 架构和 kernel 版本变化。课程 baseline 是 per-tensor 对称伪量化；它能教会误差账本，却没有 GPTQ 的 Hessian 补偿、AWQ 搜索、SmoothQuant 校准或任何生产吞吐保证。

## 10. 实验与 Notebook 对照

先在 [推理 Notebook](../../notebooks/core/11_inference_serving.ipynb) 预测 INT8 与 INT4 的误差和理论体积，再运行代码；用 [服务互动图](../interactive/serving-lab.html) 观察权重、KV cache 与并发的不同内存来源；在 [starter 20](../../exercises/starter/20_inference_systems.py) 填写量化函数并核查。实验扫描 bit∈{8,4,3}、group∈{整层,128,32}，固定同一 W，表中同时记录 MSE、max error、理论 bytes 和实际序列化 bytes。扩展实验加入一个离群值，比较 per-tensor 与分组 scale；结论必须标注这是教学伪量化。

## 11. 验收标准

- 全零、正负端点和普通向量均不除零，整数严格落入合法范围。
- 手算例得到 q=`[-7,-2,1,6]`、scale=`1.2/7`，误差界成立。
- 对非整除 group size 的矩阵，每个元素恰好量化一次，shape 不变。
- 同时提交 MSE、最大误差、下游 loss、理论/实际体积；不把伪量化时间当 INT4 kernel 时间。
- 能用一句话分别说明 GPTQ、AWQ、SmoothQuant 的目标，并指出校准和硬件边界。
- 完成 starter 20，并运行 `uv run llm-course exercises check 20`。

## 一手来源

- [GPTQ 原论文](https://arxiv.org/abs/2210.17323)与[作者官方实现](https://github.com/IST-DASLab/gptq)：近似二阶 weight-only PTQ。
- [AWQ 原论文](https://arxiv.org/abs/2306.00978)与[MIT HAN Lab 官方实现](https://github.com/mit-han-lab/llm-awq)：activation-aware 权重量化。
- [SmoothQuant 原论文](https://arxiv.org/abs/2211.10438)与[作者官方实现](https://github.com/mit-han-lab/smoothquant)：平滑激活离群值的 W8A8 PTQ。
- [LLM.int8() 原论文](https://arxiv.org/abs/2208.07339)：离群特征的混合精度分解。
