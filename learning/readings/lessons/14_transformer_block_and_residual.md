# 第 14 周：Transformer Block、归一化与残差

## 课程定位

注意力只负责跨位置路由；一个可训练的 decoder block 还需要逐位置 MLP、归一化、残差与 dropout。本周把这些部件组装成 Pre-Norm block，追踪每条支路的 shape 与梯度，并把 2017 年 Post-Norm+ReLU 配方放进 RMSNorm、SwiGLU、RoPE、GQA 的现代演化背景。目标是获得稳定、可堆叠、能端到端反传的模块，而不是堆 API 后只看输出 shape。

## 学习目标

你要能画出并实现 `x + Attention(Norm(x))`、`x + MLP(Norm(x))`；解释残差提供恒等梯度路径；区分 Pre-Norm 与 Post-Norm；手算 LayerNorm/RMSNorm；推导标准 FFN 与 SwiGLU 的参数量；记录深度方向的激活/梯度范数；通过小 batch 过拟合与消融验证组件；说明现代 decoder 的常见配方不是所有模型都必须采用的定理。

## 前置知识与资产

需掌握第 13 周 causal MHA、MLP、自动微分和基础优化。主实验为 `learning/labs/05_tiny_gpt.ipynb`；架构互动图在 `learning/readings/interactive/architecture-evolution.html`，现代组件补充图在 `learning/readings/interactive/architecture-lab.html`。本周 starter 为 `learning/labs/starter/13_tiny_gpt.py`，先完成 causal mask 与 Pre-Norm block，运行 `uv run llm-course exercises check 13`；第 15 周继续完成训练闭环。

## 自洽直觉

attention 让一个 token 收集其他位置的信息，MLP 则在每个位置独立变换通道。残差把子层看作对当前表示的“增量建议”：若新子层尚未学好，主干仍能原样传递 $x$。归一化控制送进子层的尺度，避免层层放大导致优化困难。Pre-Norm 在分支进入 Attention/MLP 前归一化，残差主路保持直接；Post-Norm 在相加后归一化。前者通常更易训练深网络，但最终表示还应做一次 norm。

## 张量/数据契约

block 输入输出均为浮点 `[B,T,D]`，使多个 block 可直接串联。Norm 只沿最后一维 $D$ 统计，不跨 batch/time。causal attention 的 Q/K/V 为 `[B,H,T,Dh]`，输出投影恢复 `[B,T,D]`。标准 FFN：`up[D,F]`、激活、`down[F,D]`；SwiGLU 有 gate 与 up 两个 `D->F` 投影，再逐元素乘并 `F->D`。residual 相加要求 dtype、device、shape 完全一致，禁止依赖不期望的广播。训练时 dropout 只在残差分支，评估与确定性单测必须 `model.eval()`。

## 推导/机制：残差、Norm 与 FFN

Pre-Norm block 可写为

$$u=x+Attn(N_1(x)),\qquad y=u+FFN(N_2(u)).$$

对第一式，$\partial u/\partial x=I+\partial Attn(N_1(x))/\partial x$，恒等项让梯度不必完全依赖深层非线性连乘；它不保证永不爆炸，但提供信息高速路。Post-Norm 则是 $u=N(x+Attn(x))$，梯度每层必须经过 Norm 的雅可比。

LayerNorm 为 $\gamma\odot(x-\mu)/\sqrt{\sigma^2+\epsilon}+\beta$；RMSNorm 不减均值：

$$RMSNorm(x)=g\odot\frac{x}{\sqrt{D^{-1}\sum_i x_i^2+\epsilon}}.$$

标准两层 FFN 忽略 bias 有 $2DF$ 参数；SwiGLU 为

$$FFN(x)=(SiLU(xW_g)\odot xW_u)W_d,$$

有 $3DF$ 参数。因此做公平参数对照时常把 SwiGLU 中间宽度设得小于传统 $4D$，不能固定 $F$ 后宣称提升全来自门控。

## 手算/数值例

取 $x=[1,2,3]$，RMS 为 $\sqrt{(1+4+9)/3}=\sqrt{14/3}\approx2.160$；若 $g=1,\epsilon=0$，RMSNorm 输出约 `[0.463,0.926,1.389]`，其平方均值为 1，但均值不为 0。LayerNorm 的均值为 2，方差为 $2/3$，输出约 `[-1.225,0,1.225]`。两者都控制尺度，却不是同一变换。再看零初始化子层：若 `Attn(N(x))=0`，Pre-Norm 残差输出严格等于 $x$；若误写为 `Attn(N(x)+x)`，输入 shape 仍正确，但恒等路径和语义都已改变。

## 最小可运行代码

以下是 CPU/offline 的最小 Pre-Norm block，使用 PyTorch SDPA 的 causal backend 与普通 GELU FFN。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    def __init__(self, dim=32, heads=4, hidden=64):
        super().__init__()
        if dim % heads:
            raise ValueError("dim 必须能被 heads 整除")
        self.h, self.dh = heads, dim // heads
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.ff = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(),
                                nn.Linear(hidden, dim))

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(self.n1(x)).chunk(3, dim=-1)
        split = lambda z: z.view(b, t, self.h, self.dh).transpose(1, 2)
        a = F.scaled_dot_product_attention(split(q), split(k), split(v),
                                            is_causal=True, dropout_p=0.0)
        a = a.transpose(1, 2).reshape(b, t, d)
        x = x + self.proj(a)
        return x + self.ff(self.n2(x))

x = torch.randn(2, 7, 32, requires_grad=True)
y = Block()(x)
assert y.shape == x.shape
y.square().mean().backward()
assert torch.isfinite(x.grad).all()
print(y.shape, x.grad.norm().item())
```

教学代码关闭 attention dropout；若加入 dropout，参数应由 `self.training` 控制并在 eval 变为 0。

## 反例/调试

错误一：第二个 Norm 仍作用于旧 `x` 而非 attention 后的 `u`；用 forward hook 记录输入可发现。错误二：遗漏最终模型 norm，Pre-Norm 多层残差尺度漂移后直接接 LM head。错误三：原地 `x += branch` 干扰 autograd 或后续复用；保持函数式相加。错误四：比较 Pre/Post-Norm 时初始化、学习率不同，无法归因。错误五：SwiGLU 把 SiLU 用在 `up` 分支或乘法之后，公式已变；为每条支路写 shape。错误六：参数量只数 `requires_grad` 总数却忽略 tied embedding；block 局部与整模统计分开。错误七：只看最终 loss，没有逐层梯度；深层问题可能被小模型掩盖。

## 主流工作与边界

原 Transformer 是 Post-Norm、ReLU FFN；GPT-2 使用改进的 LayerNorm 布局与 GELU。现代 LLaMA 类 decoder 常见 Pre-Norm、RMSNorm、SwiGLU、RoPE，并在后续版本/其他家族中配合 GQA。RMSNorm 省去中心化，SwiGLU 引入门控，RoPE 进入 Q/K 而非 block 残差；这些组件解决不同问题。Pre-Norm 的稳定性优势不意味着 Post-Norm 已无研究价值，也不代表 warmup、初始化和梯度裁剪可以全部删除。稀疏 MoE 通常替换 FFN 子层，但路由、容量和辅助损失属于后续阶段。

## 对应 Notebook、互动图与 starter

在 `learning/labs/05_tiny_gpt.ipynb` 先单独运行一个 block 的前后向和参数统计，再堆叠并画每层梯度范数。用 `learning/readings/interactive/architecture-evolution.html` 对照 2017 Transformer、GPT-2 与现代 decoder 的 Norm/MLP/位置变化，勿把架构时间线当性能排名。填写 `learning/labs/starter/13_tiny_gpt.py` 的 mask 与 Pre-Norm 残差 TODO，运行 `uv run llm-course exercises check 13`；本周产出应独立于完整训练是否收敛。

## 实验任务

实验 A：堆叠 1、4、8 个 block，对相同随机 batch 前反传，记录每层输出 RMS 与 QKV/FFN 权重梯度，比较 Pre/Post-Norm 五个 seed。实验 B：将两个输出投影零初始化，验证整个 block 初始近似恒等，并说明随后梯度是否仍能进入分支。实验 C：在参数预算近似相同下比较 GELU FFN 与 SwiGLU，而不是固定 hidden 宽度；列出公式和实测参数。实验 D：关闭 residual 或 norm，在单 batch next-token 任务上观察 loss、NaN、梯度，但不要将一次小实验推广为大模型结论。

## 验收 rubric

满分 10 分：Pre-Norm 顺序与所有 shape 正确 2 分；残差梯度推导和恒等测试 2 分；LayerNorm/RMSNorm 手算正确 1 分；FFN/SwiGLU 公平参数统计 1 分；多层梯度实验多 seed 1 分；starter 核查通过 1 分；主流配方边界无夸大 1 分；CPU/offline 报告可复现 1 分。若 residual 分支顺序错、Norm 跨 batch/time、SwiGLU 公式错或只给最终 loss，则不通过。

## 一手来源

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)：原始 Post-Norm、residual 与 position-wise FFN。
- [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745)：Pre-LN/Post-LN 初始化梯度分析。
- [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)：RMSNorm 定义与效率实验。
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)：GEGLU/SwiGLU 等 gated FFN 原始比较。
- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)：RMSNorm、SwiGLU、RoPE 的现代 decoder 配方实例。
