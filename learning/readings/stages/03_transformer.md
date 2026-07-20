# 阶段三：Transformer——每个 token 都能查阅历史

## 逐周讲义导航

> 本页是阶段知识地图，用于预习和复盘；完整推导、代码、反例、实验与验收请进入下面的逐周讲义。

- [第 9 周：文本、Unicode、字节与 token](../lessons/09_text_unicode_bytes_tokens.md)
- [第 10 周：从零实现 BPE](../lessons/10_byte_bpe_from_scratch.md)
- [第 11 周：Embedding 与位置信息](../lessons/11_embeddings_and_position.md)
- [第 12 周：缩放点积注意力](../lessons/12_scaled_dot_product_attention.md)
- [第 13 周：因果掩码与多头注意力](../lessons/13_causal_mask_and_multihead_attention.md)
- [第 14 周：Transformer Block 与残差](../lessons/14_transformer_block_and_residual.md)
- [第 15 周：训练并采样 Tiny GPT](../lessons/15_train_and_sample_tiny_gpt.md)

> 学完本章，你应能从张量形状推导 causal self-attention，正确组合 causal/padding
> mask，解释显式 `position_ids`，并用“完整前向 vs. cached decode”验证 Tiny GPT。

## 1. 学习路线与产出

1. 手算一个长度为 3 的缩放点积注意力。
2. 完成 `learning/labs/starter/02_causal_attention.py`，用 PyTorch SDPA 作 oracle。
3. 阅读 `src/llm_from_scratch/attention.py` 的安全 Softmax 与组合 mask。
4. 训练 Tiny GPT 过拟合一个 batch，再比较缓存与非缓存生成。

本章产出是正确、可测试的经典 Decoder 基线；RoPE、GQA 等现代组件在阶段四接入。

## 2. Query、Key、Value 与形状

把每个历史 token 想成一张资料卡：Query 是当前问题，Key 是检索标签，Value
是真正取回的内容。

```text
x:      [B, T, D]
q,k,v:  [B, H, T, Dh]       D = H * Dh
score:  [B, H, Tq, Tk]
output: [B, H, Tq, Dv] -> [B, Tq, D]
```

\[
\operatorname{Attention}(Q,K,V)=
\operatorname{softmax}(QK^\top/\sqrt{D_h}+M)V
\]

除以 `sqrt(Dh)` 是为了让随机初始化时点积尺度不会随维度快速增大，避免
Softmax 过早饱和。多头允许不同子空间并行检索，但不保证每个头自动对应某个
人类可解释概念。

## 3. Mask：不只是画一个下三角

生成位置 `t` 不能读取未来 key。课程的 `causal_mask(Tq, Tk)` 使用 bottom-right
对齐：当 cached decode 中 `Tq < Tk`，新 Query 被视为 Key 序列最后几个位置。

实际 batch 还常有 padding mask：

```text
allowed = causal_allowed AND padding_allowed
```

bool mask 中 `True` 表示可读；浮点 mask 直接加到 logits。一个容易漏测的边界是：
某个 Query 行可能没有任何合法 Key。对一整行有限最小值做 Softmax 会错误地产生
均匀权重；对一整行 `-inf` 直接 Softmax 又会产生 NaN。课程参考实现使用安全
Softmax，让这类行的权重与输出都为零。

必须包含三类测试：

- 修改未来 K/V 后，过去输出不变；
- causal 与 padding mask 组合正确；
- 完全遮蔽行输出为零，并与 PyTorch SDPA 对照。

在[因果注意力交互图](../interactive/core-concepts.html#attention)中观察矩阵只是第一步；
数值 oracle 才是验收依据。

## 4. 位置与显式 `position_ids`

自注意力本身不认识顺序。经典配置把 token embedding 与学习式绝对 position
embedding 相加。`TinyGPT.forward()` 接受共享 `[T]` 或逐样本 `[B,T]` 的
`position_ids`，这对左 padding、拼接样本和 cached decode 都很重要。

默认 cached decode 从已有 cache 长度开始编号。使用绝对位置时 ID 不能超过
`block_size`；使用 RoPE 时位置进入 Q/K 旋转而不是 embedding 相加，详见阶段四。

## 5. 从注意力到 Decoder

Pre-Norm block：

```text
x = x + Attention(Norm(x))
x = x + MLP(Norm(x))
```

残差是信息高速路，子层学习对当前表示的修正。多个 block 后再归一化并投影到
词表，得到每个位置的 next-token logits。`GPTConfig.classic(...)` 提供
LayerNorm、GELU MLP、MHA 和学习式绝对位置的明确基线；阶段四的 modern preset
使用相同训练接口，便于受控比较。

## 6. 分词与最小训练闭环

课程实现 Byte-level BPE：初始词表是 256 个字节，反复合并语料中最常见的相邻
token 对。字节级方法不会遇到真正的未知字符，但不同文本的 token 数量不同。

训练闭环：编码文本；构造错位一位的输入/目标；计算所有位置交叉熵；反向更新；
验证时关闭梯度；生成时逐步采样。小数据过拟合用于验证实现，不代表最终训练策略。

## 7. 代码地图与实现边界

- `tokenization.py`：可读的 Byte BPE 教学实现。
- `attention.py`：显式 attention weights 的参考实现，便于测试，不追求融合 kernel 性能。
- `transformer.py`：CPU 友好的 Tiny GPT；默认仍返回 cache 以兼容旧代码，训练时可传
  `return_caches=False` 避免保存推理缓存。

课程实现不等于生产训练框架：没有分布式并行、融合优化器、FlashAttention kernel
或大规模 checkpoint 管理。

## 8. 实验与验收

1. 对单 batch 训练，确认 loss 显著下降且 embedding 得到梯度。
2. 按 token 前向并拼接 logits，与完整前向比较。
3. 比较 `generate(use_cache=True/False)` 的 greedy 输出。
4. 用全遮蔽行、左 padding 和 `Tq<Tk` 覆盖组合 mask。
5. 记录配置、seed、dtype、shape 与最大误差，而不是只写“看起来一样”。

## 9. 常见误区

- Attention 权重是一次前向中的混合系数，不是可靠解释。
- 温度为 0 应走 argmax，不能除以 0。
- position embedding 与 token embedding 相加要求最后一维相同。
- cache 减少重复投影与历史 token 重算，但增加持久内存。
- “测试通过一个 shape”不能替代因果性、梯度和 oracle 测试。

## 一手来源

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [PyTorch scaled dot product attention API](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)

