# 第 7 周：词向量与神经语言模型

## 课程定位

语言是离散符号，神经网络擅长连续向量。Embedding 把 token id 变成可学习坐标，next-token prediction 则把语料转成统一监督任务。本周从 bigram 计数基线出发，证明 one-hot 乘权重与 embedding lookup 的等价性，再实现固定窗口 MLP 语言模型。它是从“记住相邻计数”走向“相似 token 共享统计强度”的关键一步，也是理解 Transformer 输入、词表输出和权重共享的基础。

## 学习目标

学习者应能构造不跨文档的相邻 token 对；实现平滑 bigram 概率与 NLL；解释 `nn.Embedding` 的形状、稀疏查表语义和梯度更新；从链式概率分解定义自回归语言模型；写出固定窗口 Embedding+MLP 的前向、损失与生成；比较 bigram、固定窗口 MLP 的容量、上下文和失败边界。

## 前置

需要第 4 周 Softmax/NLL、第 5 周文档切分、第 6 周 MLP。约定词表大小 V、上下文长度 C、embedding 宽度 D、隐藏宽度 H。输入 `context_ids:[B,C]` 为 long，范围 `[0,V)`；嵌入表 `E:[V,D]`；查表输出 `[B,C,D]`，展平后 `[B,C*D]`；最终 logits `[B,V]`，目标 `[B]`。

## 自洽直觉

one-hot token `e_i` 只有第 i 位为 1，乘矩阵 `E` 得 `e_iE=E[i]`，也就是直接取第 i 行。查表避免真的构造巨大 one-hot，但数学完全相同。训练时，某 token 出现在输入中，梯度只流向对应行；不同 token 的向量若在预测任务中承担相似作用，会被优化到能让后续网络产生相似输出的位置。向量几何不是预先赋予的语义，而是目标、数据和模型共同产生的可用表示。

Bigram 只用当前 token 预测下一个 token，相当于 V×V 转移表；固定窗口 MLP 同时看最近 C 个 embedding，并通过共享参数在未见过的精确上下文之间泛化。但窗口外信息绝对不可见，且把 C 个位置直接拼接使参数依赖固定 C。

## 张量/数据契约

训练数据必须按文档构造，不能把一篇结尾与下一篇开头配成 bigram。Bigram 计数 `counts:[V,V]` 中 `counts[i,j]` 表示 token i 后紧跟 j 的次数；行归一化得到条件分布。神经 bigram 可令 `nn.Embedding(V,V)` 直接为每个当前 token 存一行 next-token logits。固定窗口模型输入严格 `[B,C]`，targets `[B]`；生成前缀长度至少 C，每步只取最后 C 个 id。padding、BOS/EOS 是否加入词表必须在 tokenizer 契约中明确。

## 公式推导与机制

自回归分解把序列联合概率写为 `P(x_1,…,x_T)=∏_{t=1}^{T}P(x_t|x_{<t})`，负对数似然为 `-Σ_t log P(x_t|x_{<t})`。Bigram 近似将条件缩为 `P(x_t|x_{t-1})`，带加 α 平滑的估计是

`P(j|i)=(N_{ij}+α)/(Σ_k N_{ik}+αV)`。

固定窗口 MLP 则令 `h=tanh([E[x_{t-C}];…;E[x_{t-1}]]W1+b1)`，`logits=hW2+b2`，再 Softmax。Embedding 的反向可理解为 scatter-add：若同一 token 在 batch 出现多次，对该行的梯度贡献必须相加，与第 3 周分支梯度规则完全一致。

## 手算/数值例

语料 token 为 `[0,1,0,2]`，相邻对是 `(0,1),(1,0),(0,2)`，故 `counts[0]=[0,1,1]`。取 α=1、V=3，`P(.|0)=[1/5,2/5,2/5]`；不平滑时未见转移概率为 0，测试 NLL 可能无穷大。若 C=2、D=3、B=4，查表结果 `[4,2,3]`，展平 `[4,6]`，乘 `W1:[6,H]` 得 `[4,H]`，输出 `[4,V]`。把 `flatten(start_dim=1)` 错写成无参数 `flatten()` 会把 batch 也合并，立刻破坏契约。

## 最小可运行代码

```python
import torch
from llm_from_scratch.neural_lm import FixedWindowMLP, make_next_token_windows

tokens = torch.tensor([0, 1, 0, 2, 0, 1, 2, 1], dtype=torch.long)
contexts, targets = make_next_token_windows(tokens, context_size=2)
model = FixedWindowMLP(vocab_size=3, context_size=2,
                       embedding_dim=4, hidden_dim=16)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.05, weight_decay=0.0)
for _ in range(120):
    logits, loss = model(contexts, targets)
    optimizer.zero_grad(set_to_none=True)
    assert loss is not None
    loss.backward()
    optimizer.step()
print(contexts.shape, logits.shape, float(loss.detach()))
sample = model.generate(contexts[:1], 4, temperature=0.0)
print(sample.tolist())
```

## 反例与调试

反例一先拼接所有文档再统计 bigram，凭空学习跨文档转移。反例二将 token id 当连续数值输入线性层，使 id 2 比 id 1“天然大一倍”；id 只是类别地址。反例三在训练输入与目标间忘记右移，模型看到自己要预测的 token，loss 异常低。反例四对 logits 先 softmax 再交叉熵。反例五生成时把每一步整个序列都展平喂给固定 C 模型，shape 逐步增长。反例六从 embedding 邻近关系直接宣称稳定语义；小数据中的旋转、置换和随机性都可能改变坐标。

调试从极小语料开始，打印前五个 `(context,target)`，确认 shift 与文档边界；检查 id 值域；让模型过拟合一个 batch；比较计数 bigram 与神经 bigram NLL；最后检查生成使用最后 C 个位置。若 loss 接近 `log(V)` 且不降，模型相当于均匀猜测，应检查梯度、学习率和标签。

## 主流工作与边界

Bengio 2003 固定窗口神经概率语言模型展示了分布式表示缓解离散组合稀疏；word2vec 进一步用预测目标学习高效词向量。现代 decoder 仍使用 token embedding，并常将输出投影与输入 embedding 权重绑定，但上下文编码由注意力完成。Subword/byte tokenizer 改变“词向量”的基本单位，词义也会分散到多个 token。本周不把 embedding 相似度等同于事实知识，也不覆盖位置编码和长上下文；它们从第 9 周起展开。

## 对应 Notebook、互动图与 starter

运行 `learning/labs/02_neural_language_models.ipynb` 的 bigram 与固定窗口 MLP；打开 `learning/readings/interactive/foundations-lab.html`，在 Bigram/MLP/RNN 间切换观察可见上下文。补完 `learning/labs/starter/12_neural_lm.py` 的 `bigram_counts`，使用 `uv run llm-course exercises check 12` 核查。核心 API 在 `src/llm_from_scratch/neural_lm.py`。

## 实验

实验一手算短语料的 V×V 计数与 α 平滑 NLL。实验二验证 one-hot@E 与 E[id] 数值、梯度等价。实验三用同一切分比较计数 bigram、神经 bigram、C=2 与 C=4 MLP 的 train/val loss和参数量。实验四将某篇文档移到测试侧，确认没有跨界 bigram；改变 temperature 为 0、0.5、1.5，记录生成多样性，但不把玩具流畅度当通用质量。

## 验收 rubric

合格：bigram 计数正确、不跨文档，固定窗口模型 shape 与 loss 正确。良好：能证明 lookup 与 one-hot 乘法等价，推导自回归 NLL，过拟合小数据并生成。优秀：控制上下文长度与参数量比较泛化，诊断 label shift、零概率和生成窗口错误，并准确说明 embedding 几何的边界。把 token id 当连续特征或产生跨文档样本者不通过。

## 一手来源

- Bengio 等《A Neural Probabilistic Language Model》原论文：https://www.jmlr.org/papers/v3/bengio03a.html
- word2vec 高效分布式表示原论文：https://arxiv.org/abs/1301.3781
- PyTorch 官方 `Embedding` 契约：https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html
- PyTorch 官方语言模型示例代码：https://github.com/pytorch/examples/tree/main/word_language_model
