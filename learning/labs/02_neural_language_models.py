# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown] llm_course_preamble=true
# **课程契约** · 周次 5, 6, 7, 8 · 预计 120 分钟 · Starter 12 · 默认 CPU/离线。

# %% [markdown]
# # 神经语言模型实验室｜Bigram、固定窗口 MLP 与 RNN
#
# 目标：先按文档切分，再构造 next-token 样本；比较三种模型能看到的历史；用因果扰动和梯度曲线核查 RNN。

# %%
import torch

from llm_from_scratch.neural_lm import (
    BigramLanguageModel,
    ElmanRNNLanguageModel,
    FixedWindowMLP,
    make_document_windows,
    make_next_token_windows,
    split_documents,
)

torch.manual_seed(23)

# %% [markdown]
# ## 1. 先切文档，再造窗口
#
# 若先从长文本构造高度重叠的窗口再随机切分，相邻样本会同时出现在训练集和验证集。下面的窗口不会跨越两篇文档。

# %%
documents = [f"document-{index}" for index in range(10)]
split = split_documents(documents, seed=7)
print("train:", split.train)
print("validation:", split.validation)
print("test:", split.test)
assert set(split.train).isdisjoint(split.validation)

contexts, targets = make_document_windows(
    [torch.tensor([0, 1, 2]), torch.tensor([10, 11, 12])], context_size=2
)
print("contexts:\n", contexts, "\ntargets:", targets)
assert torch.equal(contexts, torch.tensor([[0, 1], [10, 11]]))
assert torch.equal(targets, torch.tensor([2, 12]))

# %% [markdown]
# ## 2. Bigram：当前 token 查转移表
#
# 周期序列中，每个 token 的后继是确定的。Bigram 不需要更长历史就能学会。

# %%
sequence = torch.tensor([0, 1, 2, 3] * 20)
bigram_x = sequence[:-1].view(1, -1)
bigram_y = sequence[1:].view(1, -1)
bigram = BigramLanguageModel(vocab_size=4)
optimizer = torch.optim.Adam(bigram.parameters(), lr=0.15)
bigram_losses = []
for _ in range(60):
    _, loss = bigram(bigram_x, bigram_y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    bigram_losses.append(loss.item())
assert bigram_losses[-1] < bigram_losses[0] * 0.1
print("greedy:", bigram.generate(torch.tensor([[0]]), 7, temperature=0).tolist())

# %% [markdown]
# ## 3. 固定窗口 MLP：顺序进入拼接后的 embedding
#
# MLP 的输入始终是 `[N,C]`，因此生成时只能读取最近 `C` 个 token。

# %%
mlp_contexts, mlp_targets = make_next_token_windows(sequence, context_size=2)
mlp = FixedWindowMLP(4, context_size=2, embedding_dim=4, hidden_dim=12)
optimizer = torch.optim.Adam(mlp.parameters(), lr=0.08)
mlp_losses = []
for _ in range(60):
    _, loss = mlp(mlp_contexts, mlp_targets)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    mlp_losses.append(loss.item())
assert mlp_losses[-1] < mlp_losses[0] * 0.1
print("contexts:", mlp_contexts.shape, "targets:", mlp_targets.shape)
print("greedy:", mlp.generate(torch.tensor([[0, 1]]), 6, temperature=0).tolist())

# %% [markdown]
# ## 4. Elman RNN：状态沿时间串行更新
#
# 先做未来 token 扰动：最后一个输入可以改变最后位置，但不能改变过去 logits。再从最后状态反传到每个输入 embedding，画出不同时间距离的梯度范数。

# %%
rnn = ElmanRNNLanguageModel(vocab_size=9, embedding_dim=6, hidden_dim=10)
tokens = torch.tensor([[0, 1, 2, 3, 4, 5]])
before, _, _ = rnn(tokens)
changed = tokens.clone()
changed[0, -1] = 8
after, _, _ = rnn(changed)
torch.testing.assert_close(before[:, :-1], after[:, :-1])
print("通过：修改未来不会改变过去输出")

# %%
embedded = rnn.embedding(tokens).detach().requires_grad_(True)
hidden = torch.zeros(1, rnn.hidden_dim)
for index in range(tokens.shape[1]):
    hidden = torch.tanh(rnn.input_to_hidden(embedded[:, index]) + rnn.hidden_to_hidden(hidden))
hidden.square().sum().backward()
gradient_norms = embedded.grad.norm(dim=-1)[0].detach()
print("input gradient norms:", gradient_norms.tolist())

print("Bigram loss samples:", [round(value, 3) for value in bigram_losses[::20]])
print("MLP loss samples:", [round(value, 3) for value in mlp_losses[::20]])

# %% [markdown]
# ## 练习与核查
#
# 1. 故意把两篇 token 文档先拼接再造窗口，找出跨文档样本。
# 2. 构造一个后继不唯一、必须看两个 token 才能判断的序列，比较 Bigram 与 MLP。
# 3. 把 RNN 的 `hidden_to_hidden` 权重整体乘 0.2 或 1.5，重新画梯度范数。
# 4. 用独立验证文档报告 NLL；不要用训练 loss 或一条好看的采样替代泛化评测。

# %% [markdown] llm_course_enrichment=true
# ## 5. 三类模型到底增加了什么？
#
# | 模型 | 条件信息 | 并行性 | 主要限制 |
# |---|---|---|---|
# | Bigram | 当前 token | 高 | 无法利用更早上下文 |
# | 固定窗口 MLP | 最近 `K` 个 token | 高 | 上下文长度固定 |
# | RNN | 递归隐藏状态 | 时间维串行 | 长程梯度衰减/爆炸 |
#
# 这里比较的是信息路径，不是宣称后者在所有数据上必然更好。

# %% llm_course_enrichment=true
a = [0, 1, 2, 3]
b = [9, 8, 2, 3]
assert a[-1] == b[-1]
print("Bigram 看到的条件相同：", a[-1])
print("窗口 K=3 看到的条件不同：", a[-3:], b[-3:])

# %% [markdown] llm_course_enrichment=true
# ## 6. 误区实验
#
# 训练/验证集必须先按文档或时间切分，再从各自集合造滑动窗口。若先造窗口再随机切分，相邻样本会共享几乎全部 token，验证损失会被泄漏污染。完成 `02_bigram_lm.py`、`12_mlp_lm.py`、`13_rnn_lm.py` 后分别核查。

# %% [markdown] llm_course_enrichment=true
# ## 验收与来源
#
# - [ ] 能画出 BPTT 时间展开图；[ ] 能说明隐藏状态不是“无限无损记忆”；[ ] 能用同一数据切分公平比较三类模型。
# - 来源：[A Neural Probabilistic Language Model](https://www.jmlr.org/papers/v3/bengio03a.html)、[Learning long-term dependencies with gradient descent is difficult](https://ieeexplore.ieee.org/document/279181)。
