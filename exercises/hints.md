# 练习提示

## 基础

- 形状：矩阵乘法保留左侧除最后一维以外的维度，再接上右侧最后一维。
- 链式法则：令中间变量 `u=x*x+3*x`，先求 `dy/du` 和 `du/dx`。
- Softmax：`exp(z-c)` 的公共因子会在分子和分母抵消。

## Transformer

- 因果 mask 应满足 `key_position <= query_position`。
- Cache 路径中，单个新 Query 的绝对位置不是 0，而是当前总 Key 长度减 1。
- GQA 的持久缓存只按 `kv_heads` 计数；为计算重复展开不应真的复制存储。

## MoE

- 容量公式是 `ceil(capacity_factor * tokens * top_k / experts)`。
- Softmax 概率均匀与离散 Top-k 选择均匀是两个命题。
- z-loss 主要抑制 logits 尺度，balance loss 主要影响专家使用分布。

