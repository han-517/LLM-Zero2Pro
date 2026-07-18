# 阶段四：现代 Decoder 组件——为什么今天的模型不像 2017 年原版

## RMSNorm

LayerNorm 会减均值并除标准差；RMSNorm 只按均方根缩放：

```text
RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight
```

它保留均值信息，计算更简单。测试重点是形状、数值稳定和零输入，而不是“输出均值必须为 0”。

## SwiGLU

普通 MLP 先升维、激活、再降维。SwiGLU 用一条分支产生内容，另一条分支像门：

```text
SwiGLU(x) = (SiLU(xW_gate) * xW_up) W_down
```

因为多了一组升维矩阵，公平比较时需要调整隐藏维，不能直接沿用普通 MLP 的 4D。

## RoPE

RoPE 把偶数/奇数维看成二维平面，根据位置旋转。两个向量都旋转后，它们的点积依赖位置差。代码要检查：head dimension 为偶数、cos/sin dtype 与设备一致、旋转不改变向量范数。

在[架构演化图](../interactive/architecture-evolution.html)中沿位置编码时间轴依次查看
RoPE、Position Interpolation、YaRN、LongRoPE、MLA decoupled RoPE、iRoPE 和 partial RoPE。
这些方法有的缩放频率，有的只旋转部分维度，有的交错 NoPE 层，不能统称为“把 `rope_theta` 调大”。

## MQA 与 GQA

MHA 每个 query head 有自己的 K/V；MQA 所有 query head 共享一组 K/V；GQA 让一组 query heads 共享一组 K/V。

```text
KV Cache 元素数约为 2 * layers * sequence * kv_heads * head_dim
```

因此减少 `kv_heads` 能直接降低解码缓存，但 query head 数仍可保持。

## KV Cache

Prefill 一次处理整个 prompt；decode 每次只增加一个 token。缓存每层历史 K/V 后，新 token 只需计算自己的 Q/K/V，再让新 Q 查询全部缓存。正确性验收是“逐 token 缓存 logits 与每次重算完整前缀一致”。

用[KV Cache 交互图](../interactive/core-concepts.html#kv-cache)同时改变 prompt、生成长度、
层数和 KV head 数，区分“减少重复投影”和“增加持久内存”。`TinyGPT.generate()` 默认使用
cache，并提供 `use_cache=False` 作为数值 oracle；学习式绝对位置在滑动窗口后会重新编号，
所以 cache 满时需要重建窗口。完成 `exercises/starter/03_kv_cache_budget.py` 后再做墙钟基准。

## 初始化和训练稳定性

- 记录 loss、梯度范数、激活 RMS、学习率和 token/s。
- 先检查数据与 mask，再盲目调学习率。
- Router、归一化或归约操作在低精度下可能需要 FP32。
- 梯度裁剪处理异常大更新，但不能修复错误目标或数据。

## 常见误区

- KV Cache 减少计算，却增加持久内存。
- GQA 减少持久 K/V head；参考实现按 KV head 分组计算，不使用 `repeat_interleave` 物化重复 K/V。
- RoPE 的长上下文外推不是“把最大长度改大”即可保证。
- 权重衰减与把 L2 项加进 Adam 的梯度并不总是等价。

