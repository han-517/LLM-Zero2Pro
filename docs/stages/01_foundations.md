# 阶段一：数学与 PyTorch——只学会真正用到的部分

## 直觉：张量是带形状的数据盒子

标量是一个数，向量是一列数，矩阵是一张表，张量只是更高维的表。LLM 中最常见的输入形状是：

```text
[batch, time, feature]
```

例如 `[2, 4, 8]` 表示同时处理 2 句话，每句 4 个 token，每个 token 用 8 个数字描述。

矩阵乘法不是“所有数字相乘”，而是让每个输出特征询问所有输入特征：

```text
x: [batch, time, d_in]
W: [d_in, d_out]
x @ W: [batch, time, d_out]
```

中间的 `d_in` 被求和，两边留下来。

## 导数：参数轻轻动一下，结果会怎样

导数是局部敏感度。若损失 `L` 对参数 `w` 的导数为正，增大 `w` 会让损失增大，因此梯度下降会让 `w` 变小：

```text
w <- w - learning_rate * dL/dw
```

链式法则只是把一串局部影响相乘。自动微分保存计算图，从结果向输入反向应用这些局部规则。`src/llm_from_scratch/autograd.py` 提供一个标量实现，用于看清 PyTorch 隐藏的工作。

## 概率与交叉熵

模型最后输出 logits，它们不是概率。稳定 Softmax 先减去最大值：

```text
p_i = exp(z_i - max(z)) / sum_j exp(z_j - max(z))
```

若正确 token 的概率是 `p`，损失为 `-log(p)`。模型越确信正确答案，损失越接近 0。实现训练时直接使用 `torch.nn.functional.cross_entropy`，不要先手动 Softmax，因为融合实现更稳定。

先打开[Softmax 交互图](../interactive/core-concepts.html#softmax)：分别改变共同平移和温度，
确认“平移不改变概率”“温度改变分布尖锐程度”后，再完成 `exercises/starter/01_stable_softmax.py`。

## 最小实验

```python
import torch

torch.manual_seed(7)
x = torch.randn(4, 3)
w = torch.randn(3, 2, requires_grad=True)
target = torch.tensor([0, 1, 0, 1])
loss = torch.nn.functional.cross_entropy(x @ w, target)
loss.backward()
print(loss.item(), w.grad.shape)
```

手动标注：`x @ w` 的形状是 `[4, 2]`，代表 4 个样本各有 2 个类别分数。

## 常见误区

- 广播能运行不代表语义正确；始终写出形状。
- `tensor.grad` 是累加的，每次训练步要清零。
- loss 下降只说明训练目标变好，不自动代表泛化或有用。
- 数学推导中的 batch 平均、求和与代码默认设置可能不同。

## 阶段验收

能解释矩阵乘法、链式法则、Softmax 和交叉熵；能完成一个两层网络的单 batch 过拟合，并用有限差分核对至少一个梯度。

