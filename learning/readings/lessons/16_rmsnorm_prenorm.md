# 第 16 周：Pre-Norm 与 RMSNorm——先搭好残差高速路

## 课程定位

前十五周已经完成 Decoder-only Transformer 的最小闭环。本周开始把“能训练”的经典结构改造成更接近公开现代模型的结构。第一步不是堆新模块，而是理解归一化放在残差分支前后时，梯度如何穿过几十层网络。RMSNorm 是本周可独立实现的部件，Pre-Norm 则是组织 block 的方式；二者经常一起出现，但不是同一个概念。本周交付物是经典 Post-Norm、Pre-LayerNorm、Pre-RMSNorm 的同预算对照，而不是宣称一个小模型实验足以决定所有规模的最优结构。

## 学习目标

- 从残差 Jacobian 说明 Pre-Norm 为什么保留恒等梯度路径。
- 写出 RMSNorm 的公式、形状、dtype 与 epsilon 契约，并与参考公式逐元素核对。
- 区分“不减均值”“均值被保留”“输出均值不为零”三种不同表述。
- 通过激活 RMS、梯度范数和非有限值定位深层训练故障。

## 前置

需要掌握链式法则、LayerNorm、残差连接和 PyTorch autograd。先复习第三阶段的 `x = x + sublayer(norm(x))`，并能解释 `[B,T,D]` 中每个轴。若仍把 batch 归约与 feature 归约混淆，应先在形状 notebook 中对 `dim=-1` 做手算。

## 直觉

Post-Norm 写作 `y = Norm(x + F(x))`，梯度必须穿过 Norm 才能回到较早层。Pre-Norm 写作 `y = x + F(Norm(x))`，无论子层当前学得好不好，残差项都提供一条导数为 1 的路径。它不保证梯度永不爆炸，也不意味着任意深度都无需初始化设计；它只是让“至少有一条直接通路”成为结构不变量。

RMSNorm 可以想成只校准向量的整体能量，不把向量搬到零均值超平面。它不执行均值中心化，因此输出均值不受约束；“保留均值信息”若被理解为输出均值与输入均值相等就是错误的，因为缩放仍会改变均值。

## 张量/数据契约

输入 `x` 为浮点张量 `[..., D]`，参数 `weight` 为 `[D]`，归约只沿最后一维。`D>=1`，`eps>0`，输出 shape、device、dtype 与输入一致。平方和均值在 FP16 中可能溢出或损失精度，教学实现先转 FP32 完成平方、均值和倒平方根，再乘 FP32 权重并转回输入 dtype。零向量输出应为零且有限。参数、激活和梯度的统计必须分别记录，不能把一个全局标量当作层级诊断。

## 推导与机制

令 `rms(x)=sqrt(D^{-1} sum_i x_i^2 + eps)`，则

\[
\operatorname{RMSNorm}(x)_i=w_i\frac{x_i}{\operatorname{rms}(x)}.
\]

若忽略 `eps` 并把输入整体乘正数 `a`，归一化结果对尺度近似不变；若乘负数，方向整体翻转。RMSNorm 没有 LayerNorm 的 `x-mean(x)`。对 Pre-Norm block `y=x+F(N(x))`，其 Jacobian 是 `I + J_F J_N`；恒等项 `I` 是直接梯度路径。Post-Norm 的 Jacobian 是 `J_N(I+J_F)`，所有路径都左乘归一化 Jacobian。这个推导解释结构差异，但不能单独预测最终质量，因为优化器、初始化、残差缩放与训练长度都会改变结果。

## 数值例

取 `x=[1,2,3,4]`、`eps=0`、`w=1`。均方为 `(1+4+9+16)/4=7.5`，RMS 为约 `2.7386`，输出约 `[0.3651,0.7303,1.0954,1.4606]`。输出均方为 1，但均值约 `0.9129`，既不是 0，也不等于输入均值 2.5。若输入全零，不能直接除以 0，`eps` 使分母有限。

## 最小代码

```python
import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        work = x.float()
        scale = torch.rsqrt(work.square().mean(dim=-1, keepdim=True) + self.eps)
        return (work * scale * self.weight.float()).to(x.dtype)


x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
y = RMSNorm(4, eps=1e-12)(x)
assert y.shape == x.shape and torch.isfinite(y).all()
torch.testing.assert_close(y.square().mean(dim=-1), torch.ones(1))
```
这是可读 reference，不是 fused RMSNorm kernel。生产实现会合并读写、减少中间张量，并与张量并行、autocast 和编译器交互；不能用这段 Python 的墙钟速度评价 RMSNorm 的系统收益。

## 反例与调试

最常见错误是沿 `dim=0` 归约，把不同样本耦合起来；改变 batch 中另一个样本后当前样本输出也变化即可抓住它。第二个错误是在 FP16 中先平方再转 FP32，大值已经溢出，转换无法补救。第三个错误是把 `sqrt(mean(x**2))` 写成 `mean(abs(x))`，二者形状相同却数值不同。深层模型出现 loss spike 时，先记录每层输入/输出 RMS、全局梯度范数、学习率和是否出现 NaN，再决定是否裁剪；gradient clipping 只能限制更新，不能修复错误 mask 或目标。

## 主流工作与证据等级

RMSNorm 原论文给出理论与多任务实验，属于基础证据。LLaMA、OLMo、OpenELM 等公开技术报告展示它在现代 Decoder 中的采用，属于公开模型采用证据，不等于所有架构上的因果优越性。Pre-Norm 的分析论文和大量公开配方支持其稳定性价值，但深层网络还会使用残差缩放、DeepNorm 或其他策略。课堂结论应写成“常见且有公开采用”，不能写成“LayerNorm 已被淘汰”。

## Notebook、互动图与 starter

使用 `learning/labs/06_modern_decoder.ipynb` 记录 LayerNorm/RMSNorm 的输出均值、RMS 和梯度；在 `learning/readings/interactive/training-and-alignment.html` 观察训练内存与数值策略；完成 starter `09` 中 RMSNorm 部分。互动图负责形成直觉，starter 与 PyTorch 公式 oracle 才负责验收。

## 实验

固定 seed、数据、参数预算和学习率，分别训练 2、8、16 层的 Post-LN、Pre-LN、Pre-RMSNorm Tiny GPT。每 20 step 记录各层激活 RMS、梯度范数、loss 与非有限值。再做一个故意错误的 batch 维归约作为负对照。报告不得只给最终 loss，还要说明哪一层最先偏离以及不同深度是否改变结论。

## 验收 rubric

- 40%：公式、shape、零输入、dtype 与梯度测试全部通过。
- 25%：能用 Jacobian 的恒等项解释 Pre-Norm，而不是背结论。
- 20%：实验固定变量并报告层级诊断量。
- 15%：明确 reference 与 fused kernel、公开采用与因果证据的边界。

## 一手来源

- [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745)
- [OLMo: Accelerating the Science of Language Models](https://arxiv.org/abs/2402.00838)
