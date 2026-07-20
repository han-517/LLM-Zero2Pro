# 第 8 周：RNN、状态与序列瓶颈

## 课程定位

前几周的固定窗口 MLP 只能看到规定长度的上下文；本周把“窗口”改成随时间更新的状态，完成从静态函数到序列计算的过渡。RNN 并不是今天通用文本 LLM 的主干，但它把三个以后反复出现的问题暴露得最清楚：信息怎样跨 token 传递，梯度怎样跨许多计算步骤返回，以及为什么训练阶段的并行度会决定系统上限。理解它，才能真正说明 Transformer 为何替换循环，而不是只记住“注意力更好”。

## 学习目标

完成本周后，你应能从张量形状写出 Elman RNN 单元并手工展开 BPTT；区分“每步输出”与“最终状态”；用梯度范数诊断长依赖；解释 LSTM/GRU 的门为什么改善但不消除串行依赖；最后用路径长度和可并行性比较固定窗口 MLP、RNN 与自注意力。验收关注可证伪的行为，不以生成几句像样文字替代正确性。

## 前置知识与资产

先掌握矩阵乘法、`tanh` 导数、链式法则、交叉熵和 next-token shift。主实验在 `learning/labs/02_neural_language_models.ipynb`；代码模板是 `learning/labs/starter/12_neural_lm.py`，完成后运行 `uv run llm-course exercises check 12`。互动图统一入口为 `learning/readings/interactive/index.html`，本周用其中的计算图/梯度视图观察时间展开；若互动页面未启动，下面的 CPU 代码足以完成同样的数值核查。

## 自洽直觉

把隐藏状态想成一本容量固定的“滚动摘要”：读到 $x_t$ 时，模型把旧摘要 $h_{t-1}$ 与新证据合成 $h_t$。同一组参数在所有时间复用，所以模型能接受变长序列；代价是第 $t$ 步必须等第 $t-1$ 步结束。更关键的是，远处信息只能沿 $h_1\rightarrow h_2\rightarrow\cdots\rightarrow h_T$ 的单一路径传递。摘要容量、非线性饱和和反复矩阵乘法共同形成瓶颈。LSTM 的 cell state 像一条有门控的直通车道，但时间步仍不能在训练中完全并行。

## 张量/数据契约

本文统一使用 batch-first。token id 为 `LongTensor[B,T]`，范围在 `[0,V)`；embedding 后 $X\in\mathbb{R}^{B\times T\times D}$。单层 RNN 的 $W_{xh}\in\mathbb{R}^{D\times H}$、$W_{hh}\in\mathbb{R}^{H\times H}$、$b_h\in\mathbb{R}^{H}$，初态 $h_0\in\mathbb{R}^{B\times H}$，所有时刻输出堆叠为 $H_{all}\in\mathbb{R}^{B\times T\times H}$。若每步预测下一个 token，投影 $W_{hy}\in\mathbb{R}^{H\times V}$ 得到 `logits[B,T,V]`，训练只比较 `logits[:,:-1]` 与 `ids[:,1:]`。不要把 batch 维和 time 维互换；也不要在文档边界外复用状态，除非任务明确要求流式建模。

## 推导/机制：循环与 BPTT

Elman 单元为

$$a_t=x_tW_{xh}+h_{t-1}W_{hh}+b_h,\qquad h_t=\tanh(a_t).$$

若损失只依赖 $h_T$，则早期状态的梯度包含连乘

$$\frac{\partial L}{\partial h_t}=\frac{\partial L}{\partial h_T}
\prod_{k=t+1}^{T}\left[W_{hh}^{\top}\operatorname{diag}(1-h_k^2)\right].$$

当这些雅可比的谱范数多数小于 1，连乘趋近 0，形成梯度消失；大于 1 时可能爆炸。梯度裁剪只限制爆炸的更新，并不能让已经消失的远程信号回来。LSTM 把核心记忆改为 $c_t=f_t\odot c_{t-1}+i_t\odot g_t$，于是沿 cell 路径有 $\partial c_t/\partial c_{t-1}=f_t$；遗忘门接近 1 时更容易保留梯度。它仍有有限状态、门饱和和逐步计算的边界。

## 手算/数值例

取标量 RNN，$x_t=0$、$h_0=1$、线性化激活，$h_t=0.5h_{t-1}$。三步后 $h_3=0.125$。若 $L=h_3$，则 $\partial L/\partial h_0=0.5^3=0.125$；走 20 步只剩 $0.5^{20}\approx9.54\times10^{-7}$。若循环权重改为 1.5，20 步导数约为 3325，显示爆炸。真实 `tanh` 的导数至多为 1，且在 $|a|$ 大时接近 0，所以权重不大也可能消失。这个例子说明“网络记得长期信息”不是由序列长度本身保证的，而要看实际雅可比链。

## 最小可运行代码

以下代码仅依赖 PyTorch、CPU/offline；它实现单元、展开序列并验证梯度会随时间距离变化。

```python
import torch

torch.manual_seed(0)
B, T, D, H = 2, 6, 3, 4
x = torch.randn(B, T, D, requires_grad=True)
W_xh = torch.randn(D, H) * 0.2
W_hh = torch.eye(H) * 0.5
b = torch.zeros(H)
h = torch.zeros(B, H)
states = []
for t in range(T):
    h = torch.tanh(x[:, t] @ W_xh + h @ W_hh + b)
    states.append(h)
y = torch.stack(states, dim=1)
assert y.shape == (B, T, H)
y[:, -1].sum().backward()
print("grad by time:", x.grad.norm(dim=-1).mean(dim=0).tolist())
```

尝试把 `0.5` 改为 `1.2`，同时把输入缩小，记录早期梯度如何变化。不要据一次随机初始化下结论，至少重复五个种子并画中位数。

## 反例/调试

第一类错误是原地更新状态或把 `h = h.detach()` 放进训练循环，导致 BPTT 被截断；用 `x[:,0].grad` 是否非零核查。第二类是把 padding 也当真实时间步，短样本的最终状态便混入 pad；应使用长度 mask 或 packed sequence。第三类是只看训练损失判断长依赖：复制任务可能靠局部频率“作弊”，应构造首 token 决定末 token、且中间是随机噪声的数据。第四类是看到 NaN 就盲目降低学习率；先打印状态范数与梯度范数，区分前向饱和、梯度爆炸和非法 target。最后，`batch_first=True` 只改变输入输出布局，不改变 PyTorch LSTM 的隐藏状态布局，接口需按官方形状核对。

## 主流工作与边界

RNN/LSTM/GRU 仍适合低延迟流式、小状态持续更新或严格在线信号，但主流大规模文本生成采用 decoder-only Transformer：训练可并行、任意两位置的层内路径更短。早期 seq2seq 把整句压成定长向量，这正是随后 attention 要解除的瓶颈。门控改善优化不等于无限记忆；双向 RNN 可利用未来，但不能直接用于严格自回归生成。也不要把“RNN 参数随长度恒定”误读为计算成本恒定：总 FLOPs 仍随 $T$ 增长，关键差别是临界路径为 $O(T)$。本周只建立替代动机，现代状态空间模型属于后续注意力前沿，不应混作普通 Elman RNN。

## 对应 Notebook、互动图与 starter

按顺序完成 `learning/labs/02_neural_language_models.ipynb` 的 bigram、固定窗口 MLP、RNN 三组对照，再在 `learning/readings/interactive/index.html` 打开核心概念视图，固定同一序列长度观察路径。随后填写 `learning/labs/starter/12_neural_lm.py`；starter 主要核查 bigram 计数和手写层形状，它是进入 RNN 实验的基础契约，而不是完整 RNN 答案。记录所有 shape、seed 与是否跨文档重置状态。

## 实验任务

实验 A：在“第一个 bit 决定最后标签”的合成数据上，令长度为 8、32、128，对固定窗口 MLP 与手写 RNN各训练三次，报告末位准确率、每步 wall time 和最早输入梯度。实验 B：把 $W_{hh}$ 的初始化尺度设为 0.3、0.9、1.3，绘制 `log10(grad_norm)` 随反向距离的曲线，并加/不加梯度裁剪。实验 C：用 `nn.LSTM` 替换 Elman 单元，保证参数预算大致相当；比较的结论必须区分优化稳定性、最终准确率与训练并行度。不得下载数据，合成数据需由固定 seed 生成。

## 验收 rubric

满分 10 分：形状与 next-token shift 全部正确 2 分；能从链式法则解释消失/爆炸而非背定义 2 分；三种长度与多 seed 实验可复现 2 分；反例确实排除局部捷径 1 分；正确区分裁剪、门控与并行性的作用 1 分；starter 公共核查通过 1 分；报告包含失败样例、环境和原始数值 1 分。若跨样本泄漏状态、使用测试集调参或只展示最好 seed，最多 5 分；代码无法在 CPU/offline 运行则不通过。

## 一手来源

- [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215)：定长编码向量的经典 seq2seq 设定与优化观察。
- [Learning Phrase Representations using RNN Encoder–Decoder](https://arxiv.org/abs/1406.1078)：GRU/RNN encoder-decoder 的原始论文。
- [PyTorch `nn.RNN` 官方文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.RNN.html)：方程、参数和精确输入输出形状。
- [PyTorch `nn.LSTM` 官方文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html)：门控方程、packed sequence 与状态契约。
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)：以并行度和路径长度替代循环的原始 Transformer 论证。
