# 第 3 周：导数、链式法则与自动微分

## 课程定位

训练神经网络就是反复回答一个问题：参数稍微改变，损失如何改变？本周不把 `loss.backward()` 当魔法，而是从局部导数、计算图、链式法则和分支梯度累加出发，实现一个标量反向传播核心。掌握它以后，梯度爆炸、被 detach、原地修改和没有清零等问题才能从机制上解释；Transformer 只是计算图更大，反传原则没有改变。

## 学习目标

学习者应能把导数解释为局部敏感度；用有限差分核验解析梯度；为由加法、乘法和非线性组成的图写局部反传；解释反向拓扑顺序以及同一节点多条路径的梯度为何必须相加；区分叶子张量、计算图中间值、`requires_grad`、`no_grad` 和 `detach`；完成 starter 11 并用 PyTorch 结果交叉验证。

## 前置

需要第 2 周的标量、向量和形状语言，以及一元函数斜率概念。约定损失 `L` 是标量，伴随量 `\bar v=∂L/∂v`。自动微分不是符号化简，也不是有限差分：前向执行实际数值并记录操作，反向应用每个操作的局部导数，通过链式法则组成精确到浮点误差的梯度。

## 自洽直觉

想象水从损失节点沿图反向流动。每条边按局部导数缩放水量；一个节点若流向损失有两条路线，两股水回到该节点时必须相加。乘法 `z=xy` 的局部规则是 `∂z/∂x=y`、`∂z/∂y=x`；加法把上游梯度原样分给两端；`tanh` 乘以 `1-tanh²`。反向模式特别适合“很多参数到一个标量损失”，一次反传便能得到所有参数梯度，这正是神经网络训练的形状。

## 张量/数据契约

标量教学引擎中每个 Value 保存 `data: float`、`grad: float`、父节点集合和局部 backward 闭包。前向图必须是有向无环图；反传起点是标量输出并置 `grad=1`。在 PyTorch 中，参数通常为浮点叶子张量且 `requires_grad=True`，输入 token id 是整数、不可求导；`loss` 必须可缩成标量，参数 `.grad` 与参数同 shape/dtype/device。批量损失采用 mean 还是 sum 会成比例改变梯度，必须写入契约。

## 公式推导与机制

令 `m=xy`、`a=m+x`、`f=a²`。局部导数为 `∂f/∂a=2a`，`∂a/∂m=1`，`∂a/∂x=1`，`∂m/∂x=y`，`∂m/∂y=x`。因此

`∂f/∂x = (∂f/∂a)(∂a/∂x) + (∂f/∂a)(∂a/∂m)(∂m/∂x) = 2a(1+y)`，

`∂f/∂y = (∂f/∂a)(∂a/∂m)(∂m/∂y)=2ax`。

第一个式子的加号正是分支累计。实现时先深度优先建立拓扑序，再逆序调用局部 backward；若按前向顺序，节点可能在所有下游贡献到齐之前就传播，结果不完整。有限差分 `(f(x+ε)-f(x-ε))/(2ε)` 是二阶精度的核查工具，但 ε 太大会有截断误差，太小会有浮点消减误差。

## 手算/数值例

取 `x=2,y=-0.5`，得到 `m=-1,a=1,f=1`。于是 `∂f/∂x=2*1*(1-0.5)=1`，`∂f/∂y=2*1*2=4`。若错误地只沿 `m=xy` 路径回传，会得到 `∂f/∂x=2a*y=-1`，符号都错。取 ε=`1e-5`，中心差分应接近 `(1,4)`；将 ε 降到 `1e-16` 时，双精度也可能因为 `f(x+ε)` 与 `f(x-ε)` 舍入成同一数而返回 0。

## 最小可运行代码

```python
import torch

x = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
y = torch.tensor(-0.5, dtype=torch.float64, requires_grad=True)
a = x * y + x
loss = a.square()
loss.backward()
print(loss.item(), x.grad.item(), y.grad.item())  # 1.0, 1.0, 4.0

def f(x_value: float, y_value: float) -> float:
    return (x_value * y_value + x_value) ** 2

eps = 1e-5
fd_x = (f(2 + eps, -0.5) - f(2 - eps, -0.5)) / (2 * eps)
assert abs(fd_x - x.grad.item()) < 1e-8
```

## 反例与调试

反例一是赋值 `x.grad = local_grad` 而不是 `+=`，共享节点会丢失支路。反例二是在多次训练迭代间不清梯度，PyTorch 默认累加，于是第二步包含第一步历史。反例三是用 `.item()`、NumPy 或新建 `torch.tensor(old_tensor)` 参与后续损失，图被切断。反例四是在 backward 所需值上做原地修改，版本计数器会报错或破坏梯度。反例五是用 float32 做极小 ε 的 gradcheck，数值噪声掩盖错误；官方 `gradcheck` 默认假设双精度。

调试顺序应是：用极小图和 float64；打印 `requires_grad`、`is_leaf`、`grad_fn`；检查 loss reduction；用有限差分只核查少量坐标；最后才扩大模型。若梯度为 None，先问参数是否真的参与 loss；若为零，检查激活饱和或 mask；若 NaN，沿图注册 hook 找到第一个非有限节点。

## 主流工作与边界

PyTorch 使用动态 define-by-run 图与反向模式自动微分；JAX 倾向函数变换和编译，核心链式法则相同。前向模式适合少输入多输出，反向模式适合多参数单损失；高阶梯度需保留/创建图，内存开销更大。checkpointing 通过重算前向换显存，不改变数学梯度。混合精度、分布式归约和自定义 CUDA backward 会增加数值与同步边界，本周只验证标量和小型 CPU 图。

## 对应 Notebook、互动图与 starter

运行 `learning/labs/01_shapes_and_autograd.ipynb` 的 finite difference 与分支图单元；打开 `learning/readings/interactive/foundations-lab.html` 调整 x、y 观察两支梯度。实现 `learning/labs/starter/11_autograd.py` 的 `branch_gradients`，用 `uv run llm-course exercises check 11` 核查。可对照 `src/llm_from_scratch/autograd.py`，但应先独立完成 starter。

## 实验

实验一手算上述分支图并用中心差分核验。实验二把 `f` 改为 `tanh(xy+x)`，扫描 ε 从 `1e-1` 到 `1e-12`，画解析误差的 U 形趋势。实验三故意把梯度累加改为覆盖，构造 `z=x*x+x` 让测试失败。实验四连续两次 backward，比较清零与不清零。记录每个失败的“预期、实际、根因、修复”。

## 验收 rubric

合格：解析梯度与有限差分在容差内一致，starter 通过。良好：能画出拓扑顺序并解释分支使用 `+=`、多轮训练需清梯度。优秀：能诊断 detach、原地修改、错误 reduction 与 ε 选择，并说明反向模式为何适合神经网络。只会调用 `backward()`、不能手算分支图者不通过。

## 一手来源

- PyTorch 官方 Autograd mechanics：https://docs.pytorch.org/docs/stable/notes/autograd.html
- PyTorch 官方 `gradcheck` 契约：https://docs.pytorch.org/docs/stable/generated/torch.autograd.gradcheck.html
- micrograd 官方代码与教学实现：https://github.com/karpathy/micrograd
- PyTorch 官方自动微分入门：https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html
