# 第 32 周：线性注意力——正特征映射与归一化状态

## 课程定位

稀疏注意力减少连接，本周用特征映射和结合律把历史压进固定大小状态。课程实现的是 `ELU+1` 正特征映射下的因果归一化线性注意力，并要求 parallel-prefix 与 recurrent decode 数值一致。它不是 Softmax attention 的精确重排；表达能力、状态冲突和数值行为都不同。

## 学习目标

- 写出包含特征映射、状态和归一化分母的完整公式。
- 实现 parallel cumulative-sum 与 recurrent state 两条路径并做 oracle。
- 分析状态大小、每 token 计算和与序列长度的关系。
- 构造键冲突与精确检索反例，避免只比较语言模型平均 loss。

## 前置

需要掌握矩阵结合律、outer product、prefix sum、causal attention 和浮点归约。应知道 `(QK^T)V=Q(K^TV)` 只有在线性运算下成立，Softmax 位于中间时不能直接移动括号。

## 直觉

Softmax attention 像保留全部历史卡片并按 Query 重新查找；线性 attention 把每张卡片的 key-value 外积累加进一本固定大小账本。新 Query 读取账本，不再逐卡扫描。账本省空间，却会让相似 key 的内容叠在同一状态里。归一化向量 `z` 记录“写入了多少 key 质量”，防止输出尺度随长度任意增长。

## 张量/数据契约

`Q,K:[B,H,T,Dk]`、`V:[B,H,T,Dv]`，正特征映射后 shape 不变。状态 `S:[B,H,Dk,Dv]`，normalizer `z:[B,H,Dk]`。每一步先写当前 `phi(k_t)v_t^T` 与 `phi(k_t)`，再读当前 Query，符合包含自身的 causal 语义。`eps>0`，输出 `[B,H,T,Dv]`。分段 decode 接收并返回 `(S,z)` 时必须与完整序列一致。

## 推导与机制

选择 `phi(x)=ELU(x)+1>0`：

\[
S_t=S_{t-1}+\phi(k_t)v_t^\top,
\quad z_t=z_{t-1}+\phi(k_t),
\]

\[
y_t=\frac{\phi(q_t)^\top S_t}
{\phi(q_t)^\top z_t+\epsilon}.
\]

parallel 形式先计算每个时间的 outer product，再沿 T 做 cumulative sum；recurrent 形式逐步更新同一状态。状态元素数与 T 无关，为 `BHDkDv+BHDk`；Prefill 工作随 T 线性，但训练并行效率取决于 scan/kernel。该核不等于 `exp(q·k)`，所以不是精确 Softmax。

## 数值例

取标量特征 `phi(q)=phi(k)=1`，V 依次为 2、4、8。状态和 `S_t` 为 2、6、14，`z_t` 为 1、2、3，输出为 2、3、14/3≈4.667，即历史均值。若省略分母，输出会是 2、6、14 随长度增长。若两个不同语义 token 得到同一 key，它们只能在状态中求和，Query 无法像完整 attention 那样精确选择其中一个位置。

## 最小代码

```python
def causal_linear_recurrent(q, k, v, eps=1e-6):
    q, k = F.elu(q) + 1, F.elu(k) + 1
    state = q.new_zeros(*q.shape[:2], q.shape[-1], v.shape[-1])
    normalizer = q.new_zeros(*q.shape[:2], q.shape[-1])
    outputs = []
    for t in range(q.shape[2]):
        state += torch.einsum("bhd,bhv->bhdv", k[:, :, t], v[:, :, t])
        normalizer += k[:, :, t]
        num = torch.einsum("bhd,bhdv->bhv", q[:, :, t], state)
        den = torch.einsum("bhd,bhd->bh", q[:, :, t], normalizer)[..., None]
        outputs.append(num / den.clamp_min(eps))
    return torch.stack(outputs, dim=2)
```

这是可读 recurrent reference。生产训练使用并行/分块 scan、融合归约和专门反向；Python loop 的速度不能代表线性 attention 的硬件表现。

## 反例与调试

只写 `Q(K^TV)` 而省略正特征和分母是最常见概念错误。parallel 路径若用全序列总和而非 prefix，会读取未来。write/read 顺序不同会决定是否包含当前 token，必须与 causal baseline 对齐。`eps` 过大改变短序列输出，过小在低精度中可能失稳。测试未来 V 扰动、分段 state 续算、parallel/recurrent 梯度和相同 key 冲突。

## 主流工作与证据等级

Linear Transformers 与 Performer 提供核化/随机特征等基础路线，方法并不相同。RetNet、Mamba/DeltaNet/Kimi Linear 将固定状态、衰减或更新规则进一步工程化，属于后续架构证据。本周 ELU+1 是教学选择，不等于当前所有主流 linear attention。论文的线性复杂度需与实际 scan kernel、状态宽度和短序列开销共同评估。

## Notebook、互动图与 starter

在 `docs/interactive/architecture-lab.html` 选择 Linear state；使用 `notebooks/core/08_attention_frontiers.ipynb` 对照 dense、parallel、recurrent 和状态内存；完成 starter `16` 的线性注意力部分。互动图展示固定状态，不展示冲突，实验必须加入 associative recall。

## 实验

随机小张量覆盖多个 T/D，比较 parallel/recurrent 输出和梯度。用未来扰动验证因果，用两段续算验证 state API。训练局部语言建模和多 key associative recall，比较 full attention；报告 loss、按距离 recall、state bytes 与墙钟。Python reference 只报告 correctness，性能另用可用官方 kernel。

## 验收 rubric

- 35%：parallel/recurrent、因果、分段 state 数值一致。
- 25%：公式包含正特征、S、z、分母与 eps。
- 25%：冲突/检索反例和资源测量完整。
- 15%：教学 ELU+1 与 Performer/生产 scan 边界清楚。

## 一手来源

- [Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention](https://arxiv.org/abs/2006.16236)
- [Performer](https://arxiv.org/abs/2009.14794)
- [Retentive Network](https://arxiv.org/abs/2307.08621)
- [Kimi Linear](https://arxiv.org/abs/2510.26692)
