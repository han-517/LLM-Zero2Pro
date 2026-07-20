# 第 2 周：向量、矩阵与张量形状

## 课程定位

LLM 代码中最昂贵的一类错误不是语法错误，而是“形状合法但语义错位”：程序能跑、损失也会下降，却把 batch、time、head 或 feature 混在一起。本周把张量视为带名字的轴，而不是一串神秘数字。后续注意力的 `QK^T`、多头拆分、KV cache 和 MoE 路由都依赖这套形状语言；若本周不能在纸上追踪形状，后面只能靠试错。

## 学习目标

学习者应能区分标量、向量、矩阵与高阶张量；解释点积、矩阵乘法、逐元素运算和广播的不同；为 `[B,T,D]` 每个轴写出业务含义；在不运行代码前预测 `reshape`、`transpose`、`matmul` 和广播后的形状；用断言保护数据契约，并知道连续性、view 与复制之间的边界。

## 前置

需要基本 Python 索引与高中代数。约定大写 `B` 为 batch 大小，`T` 为序列长度，`D` 为特征宽度，`V` 为词表大小。数学下标从 1 写起，Python 索引从 0 开始；讲义中的“最后一维”是 `-1`。任何例子先写轴名，再写具体大小，例如 `tokens: [B=2,T=3]`，不要只写 `[2,3]`。

## 自洽直觉

向量可看成一行有序特征；矩阵是一批向量，矩阵乘法是“对共享轴做成对乘积并求和”。张量只是继续增加索引轴。`x[b,t,d]` 表示第 `b` 个样本、第 `t` 个位置、第 `d` 个特征。线性层权重 `W[d,o]` 把每个位置的 `D` 维向量映射为 `O` 维，因此 `x @ W` 保留 `[B,T]` 两轴，只把末轴 `D` 替换成 `O`。广播不是复制的语义承诺，而是把缺失或大小为 1 的轴视作可扩展；它方便，也容易把错位掩盖。

## 张量/数据契约

以 token 嵌入为例：`token_ids` 必须是整数张量 `[B,T]`，每个值在 `[0,V)`；嵌入表 `E` 是浮点 `[V,D]`；查表后 `x=E[token_ids]` 是 `[B,T,D]`。掩码若表达“每个样本每个位置是否有效”，应为布尔 `[B,T]`，应用到特征时需显式升为 `[B,T,1]`。线性层 `W:[D,O]`、`b:[O]` 输出 `[B,T,O]`，其中 `b` 沿 B、T 广播。契约不仅写 shape，还要写 dtype、device、值域和轴的语义。

## 公式推导与机制

对 `X∈R^{B×T×D}` 和 `W∈R^{D×O}`，输出满足

`Y_{b,t,o}=Σ_{d=1}^{D} X_{b,t,d} W_{d,o}+b_o`。

求和轴 D 必须匹配；B、T 没被求和，所以原样保留。点积是 `D→1` 的特例；逐元素乘法则要求形状相同或可广播，不会对 D 求和。广播从尾轴对齐：`[B,T,D] + [D]` 合法，`[B,T,D] + [T]` 只有在 `T==D` 时可能“碰巧合法”，但语义通常错误。多头重排常从 `[B,T,H*Dh]` reshape 为 `[B,T,H,Dh]`，再 transpose 为 `[B,H,T,Dh]`；reshape 改分组，transpose 改轴顺序，两者不可互换。

## 手算/数值例

令 `X=[[1,2,3],[4,5,6]]`，形状 `[2,3]`，`W=[[1,0],[0,1],[1,1]]`，形状 `[3,2]`。第一行输出是 `[1*1+2*0+3*1, 1*0+2*1+3*1]=[4,5]`，第二行是 `[10,11]`，所以结果 `[2,2]`。若加 `b=[10,-1]`，它按行广播为 `[[10,-1],[10,-1]]`，输出 `[[14,4],[20,10]]`。再考虑 `[B=2,T=3,D=4]` 加 `[D=4]`：偏置对六个位置共享；若误把位置偏置 `[T=3]` 直接相加，会因末轴 4 与 3 不匹配而报错，这种报错反而保护了语义。

## 最小可运行代码

```python
import torch

B, T, D, O = 2, 3, 4, 5
token_ids = torch.tensor([[0, 1, 2], [2, 1, 0]], dtype=torch.long)
embedding = torch.arange(3 * D, dtype=torch.float32).reshape(3, D)
x = embedding[token_ids]                    # [B,T,D]
w = torch.randn(D, O, generator=torch.Generator().manual_seed(7))
bias = torch.zeros(O)
y = x @ w + bias                            # [B,T,O]
mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool)
y_masked = y.masked_fill(~mask[..., None], 0.0)
assert x.shape == (B, T, D)
assert y_masked.shape == (B, T, O)
print(y_masked[0, 2])                       # 被遮蔽位置全为 0
```

## 反例与调试

反例一：`x.reshape(B,-1)` 能运行，却把 time 和 feature 合并，后续层再也不知道 token 边界。反例二：`softmax(dim=1)` 用在 `[B,T,V]` 上会让不同时间位置竞争，而 next-token 分类应沿 `V` 即 `dim=-1`。反例三：`transpose` 后直接 `view` 可能因内存不连续失败；优先使用 `reshape`，或明确 `.contiguous().view(...)` 并理解复制成本。反例四：两个轴恰好同为 32，错误 transpose 仍保持相同数字形状，单靠 `shape` 测试查不出；应构造每个轴值模式不同的小张量，核对索引语义。

调试时先把表达式拆开，为每一步写 `assert tensor.shape == ...`；再检查 dtype（索引应为 long、掩码应为 bool）、值域和有限性。不要靠不断添加 `unsqueeze` 直到报错消失，因为那只能满足广播规则，不能证明轴含义正确。

## 主流工作与边界

PyTorch、NumPy 与 JAX 都采用类似的尾轴广播，但命名张量支持和编译器约束不同。现代模型大量使用 `einops`/einsum 表达轴变换，优点是公式接近，代价是错误的轴标记仍可能产生合法结果。类型系统工具可把 shape 写进注解，却无法替代运行时值域检查。本周只讨论稠密张量；稀疏张量、分布式分片、量化 dtype 和布局优化会在系统阶段出现。

## 对应 Notebook、互动图与 starter

运行 `learning/labs/01_shapes_and_autograd.ipynb` 的形状单元，并打开 `learning/readings/interactive/core-concepts.html` 与 `learning/readings/interactive/foundations-lab.html`。本周没有独立填空模板；第 3 周的 `learning/labs/starter/11_autograd.py` 会立即复用本周形状与分支语义。阶段导航见 `learning/readings/stages/01_foundations.md`。

## 实验

先为三段含 embedding、mask、linear 的代码手写逐步形状表，再运行验证。随后构造 B、T、D 互不相等的输入，分别故意交换 time/feature、修改 softmax 轴、删除 mask 的 singleton 轴，记录报错或静默错误。最后用 `torch.einsum('btd,do->bto', x, w)` 重写线性映射，与 `x @ w` 比较最大绝对误差。

## 验收 rubric

合格：能正确预测所有形状，并实现线性映射和 mask。良好：能从指标下标推导矩阵乘法，解释广播从尾轴对齐，使用不相等轴大小暴露错位。优秀：能同时说明 shape、dtype、device、值域和轴语义，并设计一个“数值形状相同但语义错误”的测试。只展示代码运行成功、无法解释求和轴或 softmax 轴者不通过。

## 一手来源

- PyTorch 官方广播语义：https://docs.pytorch.org/docs/stable/notes/broadcasting.html
- PyTorch 官方 Tensor view 与连续性说明：https://docs.pytorch.org/docs/stable/tensor_view.html
- PyTorch 官方 `einsum` 契约：https://docs.pytorch.org/docs/stable/generated/torch.einsum.html
- NumPy 官方广播规则：https://numpy.org/doc/stable/user/basics.broadcasting.html
