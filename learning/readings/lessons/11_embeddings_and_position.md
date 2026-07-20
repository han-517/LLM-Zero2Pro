# 第 11 周：Embedding 与位置信息

## 课程定位

token id 只是标签，embedding lookup 把它变成可学习向量；但仅靠内容向量的 self-attention 对输入排列具有等变性，无法区分“狗咬人”和“人咬狗”的顺序关系。本周先理解 token embedding 与输出 logits 的接口，再用绝对位置向量破除对称性，并把它放进 RoPE、ALiBi 等现代方案的演化坐标。重点是“位置怎样进入计算”，不是把每种公式都当可互换装饰。

## 学习目标

你应能证明不加位置的 self-attention 在置换下等变；实现 learned absolute embedding 与 sinusoidal encoding；写清 token、position、hidden 的形状和广播；构造只有位置才能解决的反例；区分绝对位置、相对 bias 和旋转 Q/K；说明训练长度外推不是公式看起来连续就自动成立。还要会检查 padding 的 `position_ids`、缓存解码的 offset 和超出位置表范围。

## 前置知识与资产

先掌握 embedding lookup、矩阵乘法、sin/cos 和 self-attention 的粗略流程；注意力细节在第 12 周推导。主实验为 `learning/labs/04_attention_mechanics.ipynb`，互动图是 `learning/readings/interactive/core-concepts.html` 与 `learning/readings/interactive/architecture-evolution.html`。本周没有必填 starter；不要把第 7 号 RoPE starter 当作绝对位置作业，它属于后续现代 decoder 深化。

## 自洽直觉

embedding 表像一本可学习字典：id 只选择一行，同一个 token 在不同位置初始内容相同。self-attention 只比较这些向量时，若同时打乱所有行，输出也只会按相同方式打乱；模型知道“有哪些内容”，却没有固定的第几个。给每一行加一个位置向量，相当于在内容之外贴上座位号。RoPE 则不把座位号加到 value，而是按位置旋转 query/key，使它们的点积显式依赖相对距离；ALiBi 直接在注意力 score 上加距离惩罚。三者注入的位置和归纳偏置不同。

## 张量/数据契约

`token_ids` 为 `LongTensor[B,T]`，`token_table` 为 `[V,D]`，lookup 得 $X_{tok}[B,T,D]$。绝对位置 id 通常为 `[B,T]` 或可广播的 `[1,T]`，取 `position_table[P,D]` 后逐元素相加：$X=X_{tok}+X_{pos}$，输出仍为 `[B,T,D]`。要求 `0 <= position_id < P`；padding 位置是否增长必须明示。sinusoidal encoding 可预计算 `[P,D]`，不含参数，设备和 dtype 应与输入一致。自回归 cache 解码新 token 时，位置应从已缓存长度 `offset` 开始，不能每步重置为 0。

## 推导/机制：置换等变与正弦位置

忽略 mask，令置换矩阵为 $P$，输入换为 $PX$。线性投影给出 $Q'=PQ,K'=PK,V'=PV$，于是

$$\operatorname{softmax}(Q'K'^T)V'
=\operatorname{softmax}(P QK^T P^T)PV
=P\operatorname{softmax}(QK^T)V.$$

Softmax 按行应用，所以输出只随输入同行置换；池化后甚至可能完全不变。绝对位置使 $PX+E_{pos}$ 不等于 $P(X+E_{pos})$，破坏该对称性。

原 Transformer 使用

$$PE(pos,2i)=\sin(pos/10000^{2i/D}),\quad
PE(pos,2i+1)=\cos(pos/10000^{2i/D}).$$

不同维度提供不同频率，且 $\sin(a+b),\cos(a+b)$ 可由位置 $a$ 的 sin/cos 线性组合表示相对位移。但这只是表示性质，不是训练外长度泛化的保证。learned table 更直接，却有固定最大索引；RoPE/ALiBi 改变 attention score 的位置关系，也各有外推限制。

## 手算/数值例

取两个相同内容 token，$x_1=x_2=[1,0]$。若 Q/K/V 投影都是单位阵且无位置，两行 score 相同，attention 权重均为 `[0.5,0.5]`，交换两 token 没有可观察差别。加位置 $p_1=[0,1]$、$p_2=[0,-1]$ 后，输入变 `[1,1]` 与 `[1,-1]`，点积矩阵为 `[[2,0],[0,2]]`，每个 query 更偏向自己的位置。再看一维 learned position table：训练只分配 `P=4` 行，长度 5 的输入访问索引 4 会越界；这不是调大 mask 能解决的问题。

## 最小可运行代码

以下实现正弦位置并验证形状，CPU/offline 可复制运行。

```python
import math
import torch

def sinusoidal_positions(length: int, dim: int) -> torch.Tensor:
    if dim % 2:
        raise ValueError("dim 必须为偶数")
    pos = torch.arange(length, dtype=torch.float32)[:, None]
    inv = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
    pe = torch.empty(length, dim)
    pe[:, 0::2] = torch.sin(pos * inv)
    pe[:, 1::2] = torch.cos(pos * inv)
    return pe

B, T, V, D = 2, 5, 11, 6
ids = torch.randint(V, (B, T))
table = torch.nn.Embedding(V, D)
x = table(ids) + sinusoidal_positions(T, D)[None, :, :]
assert x.shape == (B, T, D)
print(x[0, :, :2])
```

把位置行一起置换与只置换 token 两种情况分开比较，才能检验“固定座位号”是否生效。

## 反例/调试

最常见错误是 `position_ids=torch.arange(T)` 在 cached decode 每步都得到 0，导致缓存与完整前向不一致；用相同 prompt 对比两条路径的 logits。第二是把 padding 的位置处理和 attention mask 混为一谈：mask 决定能否被看见，position id 决定它带什么坐标。第三是用重复 token 做置换实验却比较集合平均值，平均本来就会抹掉顺序；应比较逐位置 logits 或设计 AB/BA 分类。第四是奇数 `D` 直接切片写 sin/cos 造成 shape 不匹配。第五是宣称 sinusoidal 或 RoPE “无限上下文”；数值频率、训练分布、attention 模式和缓存容量都会限制实际外推。

## 主流工作与边界

原 Transformer 使用固定正弦位置，BERT/GPT-2 一类模型常见 learned absolute embedding；现代 decoder 广泛采用 RoPE，把绝对旋转转化为 Q/K 点积中的相对位移；ALiBi 用按距离线性 bias 强化近邻并研究长度外推。相对位置方案不应只按榜单选：要同时考虑预训练长度、插值/缩放、KV cache、数值精度和任务是否需要精确位置。位置编码也不能单独修复 tokenizer 边界、数据顺序错误或 causal mask 泄漏。本核心周先把绝对方案做对，RoPE 的完整实现放在现代 decoder 阶段。

## 对应 Notebook、互动图与 starter

在 `learning/labs/04_attention_mechanics.ipynb` 完成“打乱 token”与绝对位置相加实验；打开 `learning/readings/interactive/core-concepts.html`，固定 Q/K/V 后只改变位置相关输入，观察 score/heatmap；再用 `learning/readings/interactive/architecture-evolution.html` 对照 absolute、relative、RoPE、ALiBi 的注入点。本周路线没有必填 starter，产出是一个能证明无位置时置换等变的最小反例及其数值结果。

## 实验任务

实验 A：实现无位置单头 self-attention，生成随机置换 $P$，数值验证 `attn(PX) == P attn(X)`，误差阈值 `1e-6`。实验 B：加入 learned absolute embedding 后重复，指出等式在哪一步失效。实验 C：在 AB/BA 合成分类上训练同一模型的无位置、sinusoidal、learned 三版，每版多 seed，比较准确率。实验 D：用长度 8 训练、长度 16 测试；learned table 明确选择报错或扩表策略，不能静默截断，再说明结果为何不足以证明通用外推。

## 验收 rubric

满分 10 分：置换等变推导完整 2 分；所有 shape、offset 与 padding 契约正确 2 分；手算和代码能复现 1 分；AB/BA 反例排除内容捷径 1 分；三种位置方法实验公平且多 seed 1 分；能说明长度外推边界 1 分；Notebook/互动图证据齐全 1 分；报告失败情况与误差阈值 1 分。若无位置模型的数据泄漏标签位置、cache 每步重置位置或把越界索引截断后不披露，则不通过。

## 一手来源

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)：正弦位置编码及 Transformer 原始结构。
- [BERT 原论文](https://arxiv.org/abs/1810.04805)：learned position embedding 的代表性模型设定。
- [OpenAI GPT-2 官方 `model.py`](https://github.com/openai/gpt-2/blob/master/src/model.py)：token/position embedding 相加的真实实现。
- [RoFormer / RoPE 原论文](https://arxiv.org/abs/2104.09864)：旋转位置进入 Q/K 点积的推导。
- [ALiBi 原论文](https://arxiv.org/abs/2108.12409)：线性距离 bias 与长度外推实验。
