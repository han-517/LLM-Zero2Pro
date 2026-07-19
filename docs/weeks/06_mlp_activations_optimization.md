# 第 6 周：MLP、激活函数与优化

## 课程定位

MLP 是 Transformer 中每个 token 上最主要的非线性计算，也是理解深度网络优化的最小场景。本周从“两个线性层为何仍是一个线性层”推导激活函数的必要性，比较 tanh、ReLU、GELU 与 SiLU 的梯度行为，并把初始化、学习率、梯度流和 AdamW 放进同一训练闭环。目标不是追逐某个激活的排行榜，而是能够让一个小网络可靠地过拟合小数据并解释失败。

## 学习目标

学习者应能写出两层 MLP 的前向与形状；证明无激活的线性层组合仍是线性映射；解释饱和、dead ReLU 与平滑门控；根据 fan-in 选择合理初始化尺度；用 SGD 或 AdamW 完成训练，记录 loss 与梯度范数；区分 L2 penalty 与 AdamW 解耦 weight decay，并完成小数据过拟合验收。

## 前置

需要矩阵乘法、链式法则、交叉熵和数据切分。约定输入 `X:[B,Din]`，第一层 `W1:[Din,H]`、`b1:[H]`，隐藏 `h=φ(XW1+b1):[B,H]`，输出 `logits=hW2+b2:[B,V]`。所有参数为浮点且参与梯度，目标为 long `[B]`。训练诊断至少包含 loss、准确率、每层梯度范数和参数是否有限。

## 自洽直觉

若没有激活，`(XW1+b1)W2+b2 = X(W1W2)+(b1W2+b2)`，无论堆多少层都可折叠成一次线性变换，无法表示 XOR 等弯曲决策边界。激活逐坐标改变斜率，让不同输入落入不同局部线性区域。tanh 输出有界且零中心，但大绝对值处梯度趋零；ReLU 正半轴梯度 1、负半轴 0，计算简单却可能永久关死；GELU/SiLU 平滑地用输入自身作门控，现代 Transformer 常用 GELU 或门控 SiLU 变体。

## 张量/数据契约

分类 MLP 的 `x` 是 `[B,Din]` float，`targets` 是 `[B]` long；语言模型 FFN 则把同一变换独立应用到 `[B,T,D]` 的每个位置，输出形状仍 `[B,T,D]`，不在时间轴混合。初始化需让每层激活与梯度方差不过快放大/缩小；bias 常从零开始。优化器只接收 `requires_grad` 参数，step 顺序是 forward→loss→zero_grad→backward→可选裁剪→step。weight decay 是否应用于 bias/normalization 需要明确参数组。

## 公式推导与机制

两层网络 `z1=XW1+b1, h=φ(z1), z2=hW2+b2`。反传有 `∂L/∂W2=h^T(∂L/∂z2)`，`∂L/∂z1=(∂L/∂z2)W2^T ⊙ φ'(z1)`，`∂L/∂W1=X^T(∂L/∂z1)`。激活导数直接控制梯度通道：tanh 导数 `1-tanh²(z)`，|z| 大时接近 0；ReLU 导数在 z>0 为 1、z<0 为 0。若权重尺度过大，tanh 一开始就饱和；过小则信号和梯度层层缩小。

Adam 维护一、二阶矩估计并按坐标缩放更新。将 L2 项梯度 `λθ` 混入 Adam 会被二阶矩归一化；AdamW 则在梯度更新之外直接衰减参数，因此对自适应优化器二者不等价。小实验中优化器不是免调参按钮，学习率仍是首要超参数。

## 手算/数值例

XOR 四点 `(0,0),(0,1),(1,0),(1,1)` 的标签 `0,1,1,0` 无法由单条直线完全分开；含两个以上隐藏单元和非线性可形成分段边界。再取 tanh 输入 z=0，导数为 1；z=3 时 `tanh(3)≈0.995`，导数约 0.0099，梯度几乎缩小百倍。ReLU 若某单元对所有样本 z=-2，输出和导数均为 0，后续仅靠该数据梯度不能复活；降低学习率、改善初始化或使用 SiLU/LeakyReLU 可缓解，但不是万能修复。

## 最小可运行代码

```python
import torch
from torch.nn import functional as F

torch.manual_seed(7)
x = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
y = torch.tensor([0, 1, 1, 0])
w1 = torch.nn.Linear(2, 8)
w2 = torch.nn.Linear(8, 2)
optimizer = torch.optim.AdamW([*w1.parameters(), *w2.parameters()], lr=0.03,
                              weight_decay=0.0)
for step in range(500):
    logits = w2(torch.tanh(w1(x)))
    loss = F.cross_entropy(logits, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if step in {0, 499}:
        print(step, float(loss.detach()), float(w1.weight.grad.norm().detach()))
    optimizer.step()
assert (logits.argmax(-1) == y).all()
```

## 反例与调试

反例一用 `nn.Sequential` 堆层却不写 shape，误把最后维或 batch 展平。反例二忘记激活，增加深度却不增加表达力。反例三学习率太大，loss 震荡并出现非有限参数；先把数据缩到四个样本、降低学习率并监控第一处 NaN。反例四 ReLU 全零，打印正激活比例与每层梯度范数。反例五每步在 `optimizer.step()` 后才清梯度，虽然下一轮可能仍工作，但诊断混乱；固定标准顺序。反例六把“Adam 的 weight_decay”不加说明地叫 L2 正则，忽略解耦语义。

若小数据都不能过拟合，优先怀疑代码、标签、容量和优化，而不是泛化。先关闭 dropout/weight decay，固定单 batch，验证 loss 能逼近零；再逐项恢复正则化。对激活比较必须使用相同 seed、初始化尺度、步数与学习率 sweep，否则单次曲线没有结论力。

## 主流工作与边界

经典 Transformer 使用 ReLU，BERT/GPT 系广泛使用 GELU，许多现代 decoder 使用 SwiGLU 等门控 FFN；门控层会增加投影并常调整隐藏宽度以匹配参数量。AdamW 是大模型训练常用基线，但 Lion、Adafactor、Muon 等优化器各有系统权衡，不能从玩具 XOR 推断规模化优劣。本周只训练浅层小网络，不覆盖归一化、残差、混合精度和分布式优化；这些机制会改变梯度尺度但不改变本周链式法则。

## 对应 Notebook、互动图与 starter

运行 `notebooks/core/02_neural_language_models.ipynb` 的 MLP、激活与梯度流部分；打开 `docs/interactive/foundations-lab.html` 比较上下文模型。实现 `exercises/starter/12_neural_lm.py` 前先完成本周无 `nn.Sequential` 的 MLP；参考实现接口位于 `src/llm_from_scratch/neural_lm.py`，不要复制答案。

## 实验

实验一用 tanh MLP 过拟合 XOR，画 loss 与梯度范数。实验二保持初始化和优化器不变，比较 ReLU、tanh、GELU、SiLU 的收敛和零/饱和比例。实验三把初始权重乘 0.01、1、20，观察信号与梯度。实验四比较 Adam 中把 `λ||θ||²` 加入 loss 和 AdamW 的解耦衰减，说明更新式差别，不要求玩具结果谁必然更优。

## 验收 rubric

合格：不用 `nn.Sequential` 写出两层 MLP 并 100% 拟合 XOR。良好：推导两层反传形状，解释无非线性可折叠、饱和与 dead ReLU，并用曲线佐证。优秀：控制变量比较激活/初始化/优化器，能区分 L2 与 AdamW，诊断 NaN 和零梯度。只展示最终准确率、不记录梯度或不能解释激活作用者不通过。

## 一手来源

- 深度稀疏 Rectifier 网络原论文：https://proceedings.mlr.press/v15/glorot11a.html
- GELU 原论文：https://arxiv.org/abs/1606.08415
- Swish/SiLU 系统研究原论文：https://arxiv.org/abs/1710.05941
- Adam 原论文：https://arxiv.org/abs/1412.6980
- AdamW 解耦权重衰减原论文：https://arxiv.org/abs/1711.05101
