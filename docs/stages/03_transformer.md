# 阶段三：Transformer——每个 token 都能查阅历史

## 直觉：查询、钥匙和内容

把每个历史 token 想成一张资料卡：

- Query：我现在想找什么？
- Key：这张卡适合被什么问题找到？
- Value：找到后真正取回的内容。

相似度 `QK^T` 决定读哪些卡，Softmax 把相似度变为权重，再对 `V` 加权求和。

## 张量形状

```text
x:      [B, T, D]
q,k,v:  [B, H, T, Dh]   其中 D = H * Dh
score:  [B, H, T, T]
output: [B, H, T, Dh] -> [B, T, D]
```

缩放点积注意力：

```text
Attention(Q,K,V) = softmax(QK^T / sqrt(Dh) + mask) V
```

除以 `sqrt(Dh)` 是为了防止维度变大后点积尺度过大，使 Softmax 过早饱和。

## 因果掩码

生成第 `t` 个 token 时不能看到未来。将 `score[t, j]` 在 `j > t` 时设为负无穷，Softmax 后权重就是 0。最重要的测试不是“mask 长得对”，而是修改未来 token 后，过去位置输出完全不变。

在[因果注意力交互图](../interactive/core-concepts.html#attention)中切换 mask、Query 位置和温度，
观察矩阵每一行的允许区域；然后独立完成 `exercises/starter/02_causal_attention.py`，
用 PyTorch SDPA 和未来 token 扰动作为两个 oracle。

## 从注意力到 GPT

一个 Pre-Norm Decoder Block：

```text
x = x + Attention(Norm(x))
x = x + MLP(Norm(x))
```

残差像信息高速路：子层只需学习对当前表示的修正。多个 Block 后再做 Norm 和词表投影，得到每个位置的 next-token logits。

## 分词

课程实现 Byte-level BPE：初始词表是 256 个字节，反复合并语料中最常见的相邻 token 对。字节级方法不会遇到真正的未知字符，但同一文本可能被拆成不同数量 token。

## 最小训练闭环

1. 文本编码为 token id。
2. 取长度为 `context_length + 1` 的片段。
3. 前 `context_length` 个是输入，后移一位是目标。
4. 计算所有位置交叉熵，反向传播并更新。
5. 验证时关闭梯度，生成时逐步采样。

参考实现位于 `tokenization.py`、`attention.py` 和 `transformer.py`。

## 常见误区

- Attention 权重不是可靠解释；它只是一次前向计算中的混合系数。
- 多头不代表每个头一定学到不同的人类概念。
- 位置 embedding 与 token embedding 相加要求最后一维相同。
- 生成时温度为 0 应使用 argmax，不能直接除以 0。
- 小数据过拟合是正确性测试，不是最终训练策略。

## 阶段验收

从零训练一个 Tiny GPT，使单 batch loss 明显下降；因果性、梯度、保存/加载和生成测试全部通过；能手写核心张量形状。

