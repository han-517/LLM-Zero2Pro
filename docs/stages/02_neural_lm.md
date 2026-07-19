# 阶段二：神经语言模型——把“猜下一个 token”变成可检验任务

## 逐周讲义导航

> 本页是阶段知识地图，用于预习和复盘；完整推导、代码、反例、实验与验收请进入下面的逐周讲义。

- [第 5 周：监督学习、泛化与数据切分](../weeks/05_supervision_generalization.md)
- [第 6 周：MLP、激活函数与优化](../weeks/06_mlp_activations_optimization.md)
- [第 7 周：词向量与神经语言模型](../weeks/07_embeddings_neural_lm.md)
- [第 8 周：RNN、状态与序列瓶颈](../weeks/08_rnn_state_and_sequence_bottleneck.md)

本阶段对应第 5–8 周。Bigram、固定窗口 MLP 和 RNN 不是为了追求现代性能，而是逐步回答三个问题：训练样本怎样构造、共享表示怎样泛化、历史信息怎样进入预测。

## 目标

- 能写出自回归概率分解、平均负对数似然和 perplexity。
- 能在构造重叠窗口前按完整文档切分数据，避免泄漏。
- 能解释 one-hot 矩阵乘法为何等价于 embedding lookup。
- 能训练并比较 Bigram、固定窗口 MLP 和 Elman RNN。
- 能从展开的 RNN 看出串行计算和长距离梯度的困难。
- 能区分训练损失、验证损失、采样质量和真正泛化。

## 前置

先完成阶段一，尤其是张量形状、交叉熵、训练循环和有限差分。开始前应能说明 logits 不是概率，并能让一个两层网络过拟合小 batch。

## 直觉

### 一个序列就是许多共享参数的预测题

文本 `春 天 来 了` 可形成：

```text
春         -> 天
春 天      -> 来
春 天 来   -> 了
```

实际批训练不需要复制三份字符串：输入取 `春 天 来`，目标整体右移为 `天 来 了`，一次前向就为多个位置计算损失。

### 三种模型逐步扩大可见历史

- **Bigram**：只用当前 token 查一张 `[V,V]` 转移 logits 表。
- **固定窗口 MLP**：拼接最近 `C` 个 embedding，能利用顺序，但看不到窗口外历史。
- **Elman RNN**：用同一组参数反复更新隐藏状态，把任意长度前缀压入固定宽度向量。

Transformer 后续会让每个位置直接读取历史位置，但先理解这些限制，才能知道注意力解决了什么、又付出了什么成本。

## 形状

设词表大小 `V`、embedding 宽度 `D`、隐藏宽度 `H`：

```text
token_ids:                    [B, T]
embedding_table:              [V, D]
embedded:                     [B, T, D]

Bigram logits:                [B, T, V]
MLP contexts:                 [N, C]
MLP concatenated embeddings:  [N, C*D]
MLP logits:                   [N, V]

RNN hidden at one step:       [B, H]
RNN all hidden states:        [B, T, H]
RNN logits:                   [B, T, V]
```

one-hot 向量 `[V]` 乘 `E:[V,D]` 会取出 `E` 的一行；`nn.Embedding` 是这个操作的高效、可学习实现。

## 必要公式

### 自回归分解

对 token 序列 `x_1,...,x_T`：

```text
p(x_1:T) = product_t p(x_t | x_<t)
```

训练通常最小化非 padding token 上的平均负对数似然：

```text
NLL = -(1/N) * sum_t log p(x_t | x_<t)
perplexity = exp(NLL)
```

perplexity 的单位依赖 tokenizer；不同词表切分得到的 token 数不同，因此不能脱离 tokenizer 直接横向比较。

### Bigram、MLP 与 RNN

```text
Bigram: logits_t = W[x_t]
MLP:    logits = W_2 tanh(W_1 concat(E[x_1],...,E[x_C]) + b_1) + b_2
RNN:    h_t = tanh(W_x E[x_t] + W_h h_{t-1} + b)
        logits_t = W_o h_t + b_o
```

RNN 通过时间反向传播时，早期梯度会反复乘隐藏状态 Jacobian。其谱尺度长期小于 1 时梯度消失，大于 1 时可能爆炸。LSTM/GRU 用门控和更直接的状态路径缓解问题，但不能消除时间步之间的串行依赖。

## 参考实现

CPU 教学实现位于 [`neural_lm.py`](../../src/llm_from_scratch/neural_lm.py)：

- `split_documents`：按完整文档确定性切分 train/validation/test。
- `make_next_token_windows`、`make_document_windows`：构造固定上下文，不跨文档边界。
- `BigramLanguageModel`：转移表基线。
- `FixedWindowMLP`：embedding 拼接与非线性。
- `ElmanRNNLanguageModel`：显式 Python 时间循环，便于观察状态。

```python
import torch
from llm_from_scratch.neural_lm import FixedWindowMLP, make_next_token_windows

tokens = torch.tensor([0, 1, 2, 3, 4, 5])
contexts, targets = make_next_token_windows(tokens, context_size=2)
model = FixedWindowMLP(vocab_size=6, context_size=2)
logits, loss = model(contexts, targets)
assert logits.shape == (4, 6)
```

训练时提供真实历史 token，生成时模型会读到自己的历史输出；错误可能累积。这常被称为 teacher forcing 与 exposure bias。它不是“训练数据有标签、生成没有标签”这么简单，而是条件前缀分布发生了变化。

## 反例

- **先造窗口再随机切分**：高度重叠的相邻窗口会落入训练集和验证集，验证损失虚低。
- **input 与 target 没右移**：模型学会复制当前 token，loss 也可能很快下降。
- **只比较训练 loss**：更大的模型可能只是记住训练样本。
- **不同 tokenizer 直接比 perplexity**：每个 token 覆盖的文本量不同。
- **把 RNN 说成完全记不住**：它能表示历史，但优化长依赖困难、训练和推理按时间串行。
- **把一次好采样当作指标**：采样有随机性，应同时报告验证 NLL、多个样本和失败案例。

## 实验

运行 [`core/02_neural_language_models.ipynb`](../../notebooks/core/02_neural_language_models.ipynb)，按顺序完成：

1. 观察“先切文档再造窗口”，确认样本不跨文档边界。
2. 在确定的玩具序列上训练 Bigram，并查看转移概率。
3. 用相同数据训练固定窗口 MLP，比较它在需要更长上下文时的失败。
4. 展开 Elman RNN，修改未来 token，验证过去 logits 不变。
5. 记录 RNN 不同时间距离的输入梯度范数，观察消失或放大。
6. 用 `temperature=0` 做确定性贪心生成，再提高温度比较多样性。

推荐附加实验：构造“最后输出必须等于第一个 token”的复制任务。让序列长度超过 MLP 窗口，比较 MLP 与 RNN；报告训练和验证结果，而不是只截图一条生成文本。

## 常见误区

- 切分单位应尽量是独立文档；还要警惕跨 split 的近重复内容。
- embedding 相似性由训练目标和数据共同产生，不保证每一维有人类可解释含义。
- RNN 的隐藏状态不是无损存储；固定宽度表示与优化路径都会形成瓶颈。
- 梯度裁剪限制爆炸梯度的更新幅度，但不会自动修复梯度消失。
- 生成温度为 0 应走 argmax 分支，不能直接除以 0。
- 小数据过拟合是正确性测试，不是最终训练策略。

## 阶段验收

- [ ] 写出 `p(x_1:T)` 的自回归分解和 perplexity 定义。
- [ ] 给定多篇文档，先切分再构造窗口，并证明没有窗口跨文档或 split。
- [ ] 证明 one-hot 乘 embedding 表与索引取行等价。
- [ ] 让 Bigram 和固定窗口 MLP 在玩具数据上明显降低 loss，并比较验证结果。
- [ ] 手写一个 Elman RNN 时间步，确认所有参数获得梯度。
- [ ] 用未来 token 扰动验证 RNN 因果性，并解释 BPTT 的梯度路径。
- [ ] 说出至少三个“训练 loss 下降但实现或实验仍错误”的例子。

## 来源与延伸

一手资料：

- [A Neural Probabilistic Language Model](https://www.jmlr.org/papers/v3/bengio03a.html)
- [Recurrent Neural Network Based Language Model](https://www.fit.vut.cz/research/result/c35937/.en)
- [Learning long-term dependencies with gradient descent is difficult](https://doi.org/10.1109/72.279181)
- [Long Short-Term Memory](https://doi.org/10.1162/neco.1997.9.8.1735)

教学补充：

- [Neural Networks: Zero to Hero](https://github.com/karpathy/nn-zero-to-hero)
- [Stanford CS336: Language Modeling from Scratch](https://cs336.stanford.edu/)
