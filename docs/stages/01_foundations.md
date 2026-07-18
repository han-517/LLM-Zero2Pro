# 阶段一：数学与 PyTorch——只学会真正用到的部分

本阶段对应第 1–4 周。目标不是先学完一本数学教材，而是建立三种调试习惯：给轴命名、用最小反例检验直觉、用数值结果核对公式。

## 目标

- 能追踪 LLM 常见张量的 batch、time、feature 轴。
- 能解释链式法则为何沿路径相乘、在分支汇合处相加。
- 能用中心有限差分核对标量梯度。
- 能从 logits、Softmax、负对数似然走到交叉熵。
- 能写出带 `zero_grad → backward → step` 的最小 PyTorch 训练循环。

## 前置

需要会写 Python 函数、循环和类，并能运行仓库测试。不要求熟练微积分证明。环境还未通过时先看[环境搭建](../00_environment.md)，不要在 Notebook 内临时安装另一套 Python。

## 直觉

### 张量是带轴含义的数据盒子

标量是一个数，向量是一列数，矩阵是一张二维表。张量把这个想法推广到更多轴。LLM 最常见的隐藏状态是：

```text
x: [batch, time, feature]
```

`[2, 4, 8]` 表示同时处理 2 个序列，每个序列有 4 个 token，每个 token 用 8 个数描述。只写数字而不写轴名，很容易把能广播的代码误当成语义正确的代码。

### 导数是局部敏感度

若损失 `L` 对参数 `w` 的导数为正，轻微增大 `w` 会让 `L` 增大，所以梯度下降沿相反方向更新：

```text
w <- w - learning_rate * dL/dw
```

链式法则不只是“一路相乘”。一条路径上的局部导数相乘；同一变量通过多条路径影响结果时，各路径贡献相加。表达式 `y = x*x + 3*x` 中，`x` 出现三次，所以反向实现必须使用 `+=`。

### 概率来自相对分数

模型最后输出 logits；它们可以为负，也不需要和为 1。Softmax 只关心相对差距。给所有 logits 加同一个常数不会改变概率，而温度会缩放差距。

## 形状

矩阵乘法让每个输出特征汇总所有输入特征：

```text
x:       [B, T, D_in]
W:       [D_in, D_out]
x @ W:   [B, T, D_out]
```

被求和的是相邻的 `D_in`，其余轴保留。多头注意力会反复使用同一规则，因此现在就养成写形状的习惯。

广播则是在不复制数据的前提下把长度为 1 或缺失的轴视为重复。例如：

```text
x:       [B, T, D]
bias:             [D]
x + bias:[B, T, D]
```

但 `[B,T,1] + [D]` 能运行，只说明广播规则允许，不说明两个轴在语义上应该相加。还要同时检查 `dtype` 和 `device`；形状相同不代表精度与硬件位置相同。

## 必要公式

### 链式法则与分支求和

若 `u=f(x)`、`y=g(u)`：

```text
dy/dx = dy/du * du/dx
```

若 `y=g(f1(x), f2(x))`，则：

```text
dy/dx = dy/df1 * df1/dx + dy/df2 * df2/dx
```

自动微分先对计算图做拓扑排序，再从输出向输入传播这些局部规则。反向模式特别适合“一个标量 loss、许多参数”的训练场景。

### 中心有限差分

对足够小但不至于被浮点舍入吞掉的 `epsilon`：

```text
df/dx ≈ (f(x + epsilon) - f(x - epsilon)) / (2 * epsilon)
```

它速度慢、只适合调试，但能作为解析梯度和自动微分之外的独立 oracle。比较时使用容差，不要求位级相等。

### Softmax、负对数似然与交叉熵

稳定 Softmax 先减最大值：

```text
p_i = exp(z_i - max(z)) / sum_j exp(z_j - max(z))
```

对类别索引目标 `y`，单样本负对数似然为：

```text
L = -log p_y
```

对 one-hot 目标，它等价于交叉熵 `-sum_i y_i log p_i`。PyTorch 的 `cross_entropy` 直接接收未归一化 logits 与整数类别索引，默认对样本取平均；不要先手动 Softmax，因为融合的 log-softmax + NLL 计算更稳定。

## 参考实现

标量计算图位于 [`autograd.py`](../../src/llm_from_scratch/autograd.py)。叶子梯度会像 PyTorch 参数梯度一样跨多次 `backward()` 累加；中间节点每次重新计算。开始新实验时，对输出调用 `zero_grad()` 清理整张可达图。

```python
from llm_from_scratch.autograd import Value

x = Value(2.0, label="x")
shared = x * x
y = shared + 3 * shared
y.backward()
assert x.grad == 16.0

y.backward()       # 叶子梯度累加，中间梯度不会重复污染
assert x.grad == 32.0
y.zero_grad()      # 清理整张图
```

PyTorch 最小训练循环：

```python
import torch

torch.manual_seed(7)
x = torch.randn(4, 3)
target = torch.tensor([0, 1, 0, 1])
model = torch.nn.Sequential(torch.nn.Linear(3, 8), torch.nn.Tanh(), torch.nn.Linear(8, 2))
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for _ in range(50):
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits, target)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
```

## 反例

- **只看能否运行**：交换两个同长度轴，广播和矩阵乘法可能仍能运行，但含义已错。
- **直接 `exp(1000)`**：会溢出；稳定 Softmax 先减最大值。
- **把链式法则说成只相乘**：共享子表达式会漏掉分支贡献。
- **忘记清梯度**：PyTorch 叶子参数梯度默认累加；训练步之间必须清理。
- **有限差分 epsilon 极小**：浮点相消会让数值梯度反而更差。
- **loss 下降等于模型有用**：小数据过拟合只证明训练闭环能工作，不证明泛化。

## 实验

1. 运行 [`00_shapes_and_autograd.ipynb`](../../notebooks/00_shapes_and_autograd.ipynb)，完成形状、广播、分支求和、有限差分和两层网络过拟合。
2. 打开 [Softmax 交互图](../interactive/core-concepts.html#softmax)，先预测共同平移和温度的效果，再拖动滑块。
3. 独立完成 [`01_stable_softmax.py`](../../exercises/starter/01_stable_softmax.py)，用大 logits、平移不变性和概率和核查。
4. 改变 MLP 的学习率、去掉非线性，各记录一个失败现象，不要只保留成功曲线。

## 常见误区

- `tensor.grad` 累加的是叶子张量梯度；非叶张量默认不保留 `.grad`。
- `reshape/view/transpose` 改变的是轴布局；矩阵含义必须随之核对。
- 数学推导中的求和、batch 平均与代码的 `reduction` 可能不同。
- 随机种子改善复现，但不同平台、dtype 和内核仍可能有小数值差异。
- 数值相近不等于公式正确；至少准备一个能区分错误实现的最小反例。

## 阶段验收

- [ ] 为三段张量代码写出每一步轴名和形状，并指出矩阵乘法求和轴。
- [ ] 手算一个含共享子表达式的梯度，解释为何分支贡献相加。
- [ ] 用中心有限差分核对 `tanh(x*x + 3*x)` 的梯度。
- [ ] 实现稳定 Softmax，并通过极端 logits 与平移不变性核查。
- [ ] 让两层网络在一个小 batch 上把 loss 降到初始值的 10% 以下。
- [ ] 能解释为什么上述过拟合仍不能证明泛化、数据切分或因果性正确。

## 来源与延伸

- [PyTorch：Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html)
- [PyTorch：Autograd mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html)
- [PyTorch：CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
- [micrograd：标量自动微分参考](https://github.com/karpathy/micrograd)

