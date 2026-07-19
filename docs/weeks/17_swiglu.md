# 第 17 周：SwiGLU——在相同参数预算下比较门控 MLP

## 课程定位

Attention 负责 token 间混合，MLP 负责每个 token 内的通道变换。本周把经典 `Linear→GELU→Linear` 替换为 SwiGLU，并重点训练“预算意识”：多一个升维投影后，若仍使用 `4D` 隐藏宽度，就不是公平替换。学习目标是实现、核算和消融，而不是把某个激活函数名字当作模型能力的充分解释。

## 学习目标

- 推导普通 MLP 与 SwiGLU 的主导参数量，并求近似等参数隐藏宽度。
- 解释 gate、content、down projection 的 shape 契约和逐元素乘法。
- 用小数值例分辨 `SiLU(gate)*up` 与 `SiLU(gate*up)`。
- 检查激活分布、饱和、死门和初始化，而不只比较最终 loss。

## 前置

需要熟悉线性层、GELU、Sigmoid、SiLU 与广播规则，并能计算矩阵 `[B,T,D] @ [D,H] -> [B,T,H]` 的参数量和 MAC。先完成 RMSNorm 周，保证比较中的归一化位置保持固定。

## 直觉

普通 MLP 为每个 token 生成一组隐藏特征再降维。GLU 家族增加一条门分支：一条分支提供内容，另一条分支决定哪些通道当前值得通过。SwiGLU 用 SiLU 作为门的非线性，它不是 0 到 1 的硬概率门；SiLU 在负区间也可能给出小负值，所以“开/关”只是帮助理解的比喻。

门控增加表达路径，也增加矩阵。公平问题由此产生：若 baseline 隐藏宽度为 `4D`，两矩阵主导参数约 `8D²`；SwiGLU 有 gate、up、down 三矩阵，隐藏宽度 `H` 时约 `3DH`。令 `3DH≈8D²` 得 `H≈8D/3`，工程中再按 8、64 或 256 对齐。

## 张量/数据契约

输入 `x:[B,T,D]`。`gate(x)` 与 `up(x)` 都为 `[B,T,H]`，二者必须完全同形状才能逐元素乘；`down` 把 `[B,T,H]` 还原为 `[B,T,D]`。所有权重 device/dtype 一致，bias 是否启用必须与整体架构预算一同记录。参数比较要注明 embedding、attention 是否计入；只比较 MLP 子层与比较全模型会得到不同百分比。

## 推导与机制

SwiGLU 写作

\[
y=W_d\left(\operatorname{SiLU}(xW_g)\odot(xW_u)\right).
\]

其中 `SiLU(z)=z sigmoid(z)`。若忽略 bias，普通 `D→4D→D` MLP 参数为 `4D²+4D²=8D²`；SwiGLU 参数为 `DH+DH+HD=3DH`。取 `H=8D/3` 时主导项一致。MAC 也近似按同样矩阵元素数缩放，但激活函数和逐元素乘增加少量开销。参数相等不意味着 wall-clock 相等，矩阵对齐、融合 kernel 和硬件利用率会影响速度。

## 数值例

令 gate pre-activation 为 `[-2,0,2]`，content 为 `[3,3,3]`。SiLU 约为 `[-0.2384,0,1.7616]`，乘 content 后约 `[-0.7152,0,5.2848]`。门不是非负概率：第一个通道传递负值。若误写成 `SiLU(gate*content)`，得到的是对 `[-6,0,6]` 做 SiLU，数值与梯度都不同。对 `D=768`，普通 `H=3072`；等参数 SwiGLU 的理论 `H=2048`，恰为 `8D/3`。

## 最小代码

```python
import torch
from torch import nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, bias: bool = False):
        super().__init__()
        self.gate = nn.Linear(d_model, hidden_dim, bias=bias)
        self.up = nn.Linear(d_model, hidden_dim, bias=bias)
        self.down = nn.Linear(hidden_dim, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.silu(self.gate(x)) * self.up(x)
        return self.down(hidden)


x = torch.randn(2, 3, 4, requires_grad=True)
y = SwiGLU(d_model=4, hidden_dim=8)(x)
assert y.shape == x.shape
y.square().mean().backward()
assert x.grad is not None and torch.isfinite(x.grad).all()
```
这是教学模块。生产实现可能把 gate/up 合并成一次更宽的 GEMM，再使用 fused activation-and-multiply，并让张量并行按隐藏轴切分。不能因 Python 中有三次 `Linear` 就推断生产 kernel 必然慢三倍。

## 反例与调试

第一类错误是把逐元素乘写成矩阵乘，shape 可能在小维度碰巧合法却改变语义。第二类是 baseline 用 `4D`、SwiGLU 也用 `4D` 后宣布后者更好，却没有报告参数多了约 50%。第三类是 gate 与 up 共用同一权重，退化成特殊自门控而非标准 SwiGLU。若激活 RMS 持续增大，检查初始化、RMSNorm 和 down projection；若大量通道长期接近零，画 gate pre-activation 与 SiLU 后分布，而不是立即增大学习率。

## 主流工作与证据等级

GLU Variants 论文在 Transformer 上系统比较多个门控变体，是基础实验证据。PaLM、LLaMA 系列、OLMo 与多种公开现代模型采用 SwiGLU，属于公开配方证据。不同报告的隐藏宽度还受对齐与参数预算影响，不能把 `8D/3` 当成不可更改的数学定律。2026 年仍有模型使用 GELU 或其他 MLP；“主流”表示常见，不表示唯一正确。

## Notebook、互动图与 starter

在 `notebooks/core/06_modern_decoder.ipynb` 对比 GELU MLP 与 SwiGLU 的参数、MAC、激活直方图和单步梯度；完成 starter `09` 的门控部分。`docs/interactive/architecture-evolution.html` 用于观察公开模型的组件组合，但参数核算必须由代码输出验证。

## 实验

构造三个模型：`GELU H=4D`、`SwiGLU H=8D/3`、故意不公平的 `SwiGLU H=4D`。固定 attention、norm、数据顺序、训练 token 和 seed。报告总参数、MLP 参数、每 step token/s、峰值内存、验证 loss，并保存 gate 分布。至少重复三个 seed；若差异小于 seed 波动，应写“未观察到稳定差异”。

## 验收 rubric

- 35%：实现和公式 oracle 正确，shape 与梯度测试完整。
- 25%：参数预算推导与代码统计一致。
- 25%：公平消融含速度、内存、质量和多 seed。
- 15%：能解释教学 Python、融合 kernel 与公开模型证据的边界。

## 一手来源

- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)
- [PaLM: Scaling Language Modeling with Pathways](https://arxiv.org/abs/2204.02311)
- [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288)
- [OLMo: Accelerating the Science of Language Models](https://arxiv.org/abs/2402.00838)
