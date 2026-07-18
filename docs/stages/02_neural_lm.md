# 阶段二：神经语言模型——让“猜下一个词”成为可训练任务

## 直觉

语言模型反复玩同一个游戏：给出前文，猜下一个 token。训练文本 `春 天 来 了` 会变成多组题目：

```text
春 -> 天
春 天 -> 来
春 天 来 -> 了
```

Bigram 模型只看最后一个 token；MLP 可以看固定窗口；RNN 把历史压进一个状态。Transformer 则让当前位置直接读取历史位置。

## 从 one-hot 到 Embedding

若词表大小为 `V`，one-hot 向量长度为 `V`，只有一个位置为 1。它乘上矩阵 `E: [V, d]`，结果就是取出 `E` 的某一行。因此 embedding lookup 是 one-hot 矩阵乘法的高效写法。

```text
token_ids: [batch, time]
embedding_table: [vocab, d_model]
x: [batch, time, d_model]
```

## Bigram、MLP 与 RNN

- Bigram：`logits = table[current_token]`，用于验证数据管道和采样。
- MLP LM：拼接最近若干 embedding，再预测下一个 token。
- RNN：`h_t = tanh(W_x x_t + W_h h_{t-1})`，同一组参数沿时间重复使用。

RNN 的问题不是“完全记不住”，而是训练和推理必须按时间顺序更新，长距离梯度也容易缩小或放大。

## 实验顺序

1. 在 20 个字符的语料上让 Bigram 过拟合。
2. 固定相同词表和训练对，比对 MLP 与 Bigram。
3. 构造需要记住第一个 token 的复制任务，比较固定窗口与 RNN。
4. 记录训练损失和验证损失，观察泛化间隙。

## 常见误区

- 数据切分应在构造重叠窗口前谨慎完成，否则相邻文本会泄漏。
- perplexity 是 `exp(平均负对数似然)`，不同 tokenizer 的 perplexity 不宜直接比较。
- embedding 的相似性是训练目标产生的结果，不保证每一维都有人类可解释含义。
- teacher forcing 训练看到真实历史，生成时看到自己的历史，二者分布不同。

## 阶段验收

能从文本构造 `(input, target)`；能训练 Bigram、MLP 或 RNN；能解释 logits、概率、交叉熵和采样之间的关系，并指出 RNN 的串行瓶颈。

