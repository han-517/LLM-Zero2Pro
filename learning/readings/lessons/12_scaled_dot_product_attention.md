# 第 12 周：缩放点积注意力

## 课程定位

本周从“查询—钥匙—内容”的检索直觉推到可计算、可测试的 scaled dot-product attention。它是 Transformer 最核心的路由运算，也是后续 multi-head、causal mask、GQA、KV cache 与 FlashAttention 的共同数学语义。目标不是只会调用 `nn.MultiheadAttention`，而是先用小矩阵逐项算对 score、scale、Softmax 与加权和，再与 PyTorch SDPA 做数值对照。

## 学习目标

你要能说明 Q/K/V 各自职责；从输入投影写出所有 shape；推导为何除以 $\sqrt{d_k}$；稳定实现行 Softmax；手算一行权重与输出；区分数学 attention 与某个高效 kernel；用 atol/rtol 与官方 SDPA 核对前向和梯度；解释 attention 权重不是因果解释或知识来源证明。完成后应独立补完 causal-attention starter 的无 mask 核心。

## 前置知识与资产

需要矩阵乘法、方差、指数与 Softmax，以及第 11 周的位置概念。主实验是 `learning/labs/04_attention_mechanics.ipynb`；互动图为 `learning/readings/interactive/core-concepts.html`；starter 是 `learning/labs/starter/02_causal_attention.py`，第 12 周先完成 QKV、scale 和输出部分，第 13 周再加入严格 causal mask，最终运行 `uv run llm-course exercises check 02`。

## 自洽直觉

对每个目标位置，query 表达“我现在需要什么”；每个上下文位置的 key 表达“我能用什么特征被检索”；二者点积形成相容分数。Softmax 把一行分数变成非负且和为 1 的路由权重，再对 value 做加权和。把 key 和 value 分开很重要：匹配依据与被搬运内容不必相同。attention 不是从数据库取一个唯一答案，而是产生可微的凸组合；权重尖锐还是平滑受 score 尺度影响。

## 张量/数据契约

单头自注意力输入 $X\in\mathbb{R}^{B\times T\times D}$，投影矩阵 $W_Q,W_K\in\mathbb{R}^{D\times d_k}$、$W_V\in\mathbb{R}^{D\times d_v}$，得 $Q,K[B,T,d_k]$ 与 $V[B,T,d_v]$。score 为 `Q @ K.transpose(-2,-1)`，形状 `[B,T_q,T_k]`；自注意力通常 $T_q=T_k=T$，交叉注意力不要求相等。行 Softmax 沿最后一个 key 维。权重 `[B,T_q,T_k]` 乘 $V[B,T_k,d_v]$ 得输出 `[B,T_q,d_v]`。多头接口稍后扩为 `[B,H,T,Dh]`。dtype 必须为浮点，token id 不能直接点积。

## 推导/机制：缩放与稳定 Softmax

定义

$$\operatorname{Attn}(Q,K,V)=
\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V.$$

先令 mask $M=0$。若 $q_i,k_i$ 独立、均值 0、方差 1，则点积 $s=\sum_{i=1}^{d_k}q_i k_i$ 的方差约为 $d_k$，标准差为 $\sqrt{d_k}$。维度增大时，未经缩放的 logits 变大，Softmax 更易饱和，非 winner 项梯度接近 0；除以 $\sqrt{d_k}$ 把典型尺度拉回常数量级。这是初始化条件下的方差论证，不是说训练后每个向量方差严格为 1。

数值稳定 Softmax 对每行先减最大值：

$$p_j=\frac{e^{z_j-m}}{\sum_l e^{z_l-m}},\quad m=\max_l z_l.$$

减同一常数不改变概率，却避免 `exp(1000)` 溢出。输出每行是 values 的凸组合；若所有 value 相同，无论权重怎样输出都相同，不能据此判断 attention 权重实现正确。

## 手算/数值例

取单个 query $q=[1,0]$，keys $k_1=[1,0],k_2=[0,1]$，values $v_1=[2,0],v_2=[0,4]$，$d_k=2$。未缩放 score 为 `[1,0]`，缩放后 `[1/\sqrt2,0]\approx[0.707,0]`。减最大值后指数为 `[1,e^{-0.707}]\approx[1,0.493]`，权重约 `[0.670,0.330]`；输出约为 `[1.340,1.320]`。若忘记 scale，权重是 `[0.731,0.269]`，输出更尖锐。两者都和为 1，所以“行和正确”不足以发现漏除 $\sqrt{d_k}$。

## 最小可运行代码

下面的手写实现与 PyTorch SDPA 在 CPU 上对齐；关闭 dropout，保证核查确定。

```python
import math
import torch
import torch.nn.functional as F

def attention(q, k, v):
    if q.shape[-1] != k.shape[-1] or k.shape[-2] != v.shape[-2]:
        raise ValueError("Q/K head dim 与 K/V sequence 必须匹配")
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    weights = torch.softmax(scores, dim=-1)
    return weights @ v, weights

torch.manual_seed(0)
q = torch.randn(2, 1, 4, 8, requires_grad=True)
k = torch.randn(2, 1, 5, 8, requires_grad=True)
v = torch.randn(2, 1, 5, 6, requires_grad=True)
ours, weights = attention(q, k, v)
ref = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
torch.testing.assert_close(ours, ref, atol=1e-6, rtol=1e-5)
assert torch.allclose(weights.sum(-1), torch.ones(2, 1, 4))
print(ours.shape, weights.shape)
```

官方函数可能选择不同 backend；CPU 通常走 math 实现。浮点运算次序不同会有小误差，应用容差比较而不是 `==`。

## 反例/调试

错误一：Softmax 沿 query 维而非 key 维，列和为 1 却语义错误；断言每个 query 的最后一维和。错误二：scale 用 $\sqrt{D}$ 或 $\sqrt{T}$，多头后尤其隐蔽；必须用单头 `q.shape[-1]`。错误三：比较两个实现时一个启用 dropout，结果随机。错误四：Q/K/V 都设为全零或 value 全相同，错误路由也会“通过”；使用非对称小矩阵和随机梯度测试。错误五：把 attention heatmap 直接解释为 token 对最终决策的贡献；残差、value、后续层与非线性都会改变作用。错误六：半精度下自行 `exp`，先在 float32 验证稳定性再讨论混合精度。

## 主流工作与边界

PyTorch `scaled_dot_product_attention` 把同一数学语义派发到 math、memory-efficient 或 FlashAttention 类 backend；FlashAttention 是精确 attention 的 IO-aware 重排，不是线性近似，也不改变 $O(T^2)$ 的总成对计算。实际系统常用 fused kernel，教学手写矩阵仍是正确性 oracle。长上下文的瓶颈不仅是 FLOPs，还有 score/中间量的内存流量；另一方面，attention 权重稀疏或局部不自动意味着可使用某个稀疏 kernel。当前周不覆盖 causal/multi-head，不能把无 mask 结果用于自回归训练。

## 对应 Notebook、互动图与 starter

在 `learning/labs/04_attention_mechanics.ipynb` 逐项打印 Q、K、score、scaled score、weights、output；打开 `learning/readings/interactive/core-concepts.html` 编辑 Q/K/V，复现本章数值例并观察有无 scale 的熵。随后填写 `learning/labs/starter/02_causal_attention.py` 中与 score、scale、Softmax、value 加权有关的 TODO；保留第 13 周 mask 测试。每次修改先写 shape 注释，避免靠广播碰巧运行。

## 实验任务

实验 A：令 $d_k=4,16,64,256$，随机生成单位方差 Q/K，比较有无 scale 的 score 标准差、平均 attention 熵和 Q 梯度范数，至少五个 seed。实验 B：实现稳定 Softmax 与朴素 Softmax，在 `[1000,999]`、`[-1000,-1001]` 上展示有限性。实验 C：随机使用 $T_q\ne T_k$，与 SDPA 核对前向和三组梯度。实验 D：测量 $T=32,64,128,256$ 的 CPU 时间与权重矩阵元素数；不要用短小 noisy benchmark 宣称硬件复杂度定律，只报告趋势和环境。

## 验收 rubric

满分 10 分：QKV/score/output 形状无误 2 分；scale 方差推导清楚 2 分；手算与代码一致 1 分；稳定 Softmax 和非对称反例有效 1 分；与 SDPA 前向及梯度在容差内 2 分；多 seed 熵实验与边界解释 1 分；starter 核心完成且 CPU/offline 1 分。若 Softmax 维度错、scale 维度错、使用相同 values 掩盖路由错误，或把 FlashAttention 说成近似 attention，则不通过。

## 一手来源

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)：scaled dot-product 与 multi-head attention 的原始定义。
- [PyTorch `scaled_dot_product_attention` 官方文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)：精确 API、dropout、mask 与 backend 行为。
- [FlashAttention 原论文](https://arxiv.org/abs/2205.14135)：IO-aware exact attention 与内存访问分析。
- [FlashAttention 官方代码](https://github.com/Dao-AILab/flash-attention)：kernel 支持范围、接口和数值测试。
- [PyTorch SDPA 官方教程](https://docs.pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html)：backend 选择和基准方法。
