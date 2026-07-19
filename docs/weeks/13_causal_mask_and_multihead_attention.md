# 第 13 周：因果掩码与多头注意力

## 课程定位

第 12 周的每个位置能看见所有 key，适合双向编码，却会让自回归训练偷看答案。本周加入 causal mask，并把多个低维 attention head 并行后拼接。这里的正确性不是“loss 会下降”，而是一个强不变量：改变未来 token，过去位置的 logits 必须保持不变。完成后你将拥有 decoder self-attention 的完整核心，并能理解 MHA 到 MQA/GQA 的系统演化。

## 学习目标

你应能构造 $T_q\times T_k$ 的因果允许矩阵，处理 cache offset；区分 bool mask 在不同 PyTorch API 中的相反语义；证明未来扰动不影响过去；写出 MHA 的投影、reshape、transpose、concat 与输出投影；计算参数量和 KV 张量尺寸；说明每个 head 并不被保证学到可命名语言学功能；理解 MQA/GQA 共享 K/V 的质量—带宽折中。

## 前置知识与资产

需完成第 12 周，掌握 `[B,H,T,Dh]` attention 与 stable Softmax。主实验仍为 `notebooks/core/04_attention_mechanics.ipynb`，互动图 `docs/interactive/core-concepts.html` 可切换 mask；starter 是 `exercises/starter/02_causal_attention.py`，完成后运行 `uv run llm-course exercises check 02`。现代 RoPE/GQA starter 在后续周，不要在本周把共享 K/V 混入基准 MHA。

## 自洽直觉

训练时整段目标序列一次送入模型是为了并行，但第 $i$ 行 query 只允许读取位置 $j\le i$。mask 像在 score 矩阵上封住上三角；被禁止项在 Softmax 前变为 $-\infty$，概率才严格为 0。多头则像用几套不同的检索坐标：每头有独立 Q/K/V 子空间，输出拼接后再由 $W_O$ 混合。头数增加不改变总 hidden 宽度时，每头维度会变小，并非免费增加容量。

## 张量/数据契约

输入 $X[B,T,D]$，要求 `D % H == 0`，$D_h=D/H$。一次线性层可生成 `qkv[B,T,3D]`，再拆成三份并 reshape/transponse 为 `[B,H,T,Dh]`。score/allowed 均可广播到 `[B,H,Tq,Tk]`；对完整训练 `Tq=Tk=T, offset=0`，允许 `j<=i`。cached decode 中已有 `offset=Tk-Tq` 个旧 key，新 query 的绝对位置是 `offset+i`，允许 `j<=offset+i`。head 输出 `[B,H,T,Dh]` transpose 后必须 `contiguous()` 或用 `reshape` 合为 `[B,T,D]`，最后乘 $W_O[D,D]`。

## 推导/机制：mask 与多头

因果 attention 为

$$A_{ij}=\operatorname{softmax}_j\left(
\frac{q_i k_j^T}{\sqrt{D_h}}+M_{ij}\right),\qquad
M_{ij}=\begin{cases}0,&j\le i\\-\infty,&j>i.\end{cases}$$

于是禁止项指数为 0。不能在 Softmax 后简单乘 0 而不重新归一化，否则行和小于 1；更不能用有限的 `-1e2` 就宣称对所有 dtype/score 都严格遮蔽。

多头定义

$$head_r=Attn(XW_Q^{(r)},XW_K^{(r)},XW_V^{(r)}),\quad
MHA(X)=Concat(head_1,\ldots,head_H)W_O.$$

若 Q/K/V/O 都是无 bias 的 $D\times D$ 总投影，参数量为 $4D^2$，与头数无关。KV cache 每层的 K/V 元素数为 $2BHTD_h=2BTD$；MQA 把 KV 头降至 1，GQA 使用 $H_{kv}$ 介于 1 与 $H$ 之间以降低解码带宽。

## 手算/数值例

对 $T=3$，允许矩阵为 `[[1,0,0],[1,1,0],[1,1,1]]`。若第一行未 mask 的 score 是 `[1,100,100]`，正确 mask 后权重严格 `[1,0,0]`；若 Softmax 后才乘 mask，原 Softmax 第一项约为 0，输出也约为 0，而正确结果应是 $v_0$。取 $D=8,H=2,D_h=4$，每个 Q/K/V 总投影仍各含 64 个权重，$W_O$ 64 个，总计 256（忽略 bias），不是把单头 256 再乘 2。

## 最小可运行代码

下面构造支持非方阵/offset 的 allowed mask，并用未来扰动验证 PyTorch SDPA。注意 SDPA 的 bool `True` 表示允许参与。

```python
import torch
import torch.nn.functional as F

def causal_allowed(tq: int, tk: int, device=None):
    if tq > tk:
        raise ValueError("cached self-attention 要求 tq <= tk")
    offset = tk - tq
    q_pos = offset + torch.arange(tq, device=device)[:, None]
    k_pos = torch.arange(tk, device=device)[None, :]
    return k_pos <= q_pos

torch.manual_seed(0)
B, H, T, Dh = 1, 2, 5, 4
q = torch.randn(B, H, T, Dh)
k = torch.randn(B, H, T, Dh)
v = torch.randn(B, H, T, Dh)
mask = causal_allowed(T, T)
y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                    dropout_p=0.0)
v_changed = v.clone(); v_changed[:, :, 3:] += 1000
y2 = F.scaled_dot_product_attention(q, k, v_changed, attn_mask=mask,
                                     dropout_p=0.0)
torch.testing.assert_close(y[:, :, :3], y2[:, :, :3])
print(mask.int(), y.shape)
```

该扰动只改 value，完整模型测试还应改变未来 token 后重算 Q/K/V，从最终 logits 检验端到端不变量。

## 反例/调试

错误一：`nn.MultiheadAttention` 的 `key_padding_mask=True` 表示屏蔽，而 SDPA 的 bool `attn_mask=True` 表示允许；迁移时必须核对官方接口。错误二：用 `torch.tril` 方阵处理 cache 的 `Tq=1,Tk>1`，只允许 key 0，应该加 offset。错误三：某行全部被 mask，手写 Softmax 会出现 `-inf - (-inf)` 和 NaN；定义接口拒绝这种输入或显式处理。错误四：transpose 后用 `view` 得到错乱数据；先 contiguous 或 reshape。错误五：未来扰动测试只比较位置 0，而 BOS 本来就只能看自己；应比较所有受保护前缀。错误六：dropout 开启导致两次输出不同，被误判为泄漏；测试模式设为 0。

## 主流工作与边界

标准 MHA 仍是数学基线，但 decoder 推理常受 KV cache 读取带宽限制。MQA 让所有 query heads 共享一套 K/V，最快但可能损失质量；GQA 使用若干 KV groups，在质量与吞吐间折中，已成为许多现代 decoder 的常见部件。FlashAttention 改变精确 MHA 的 IO 调度；它与 GQA 解决的问题不同，可组合使用。局部/稀疏 mask 通过减少可见 pair 约束感受野，但 causal 只规定方向，不规定局部性。任何高效实现都应先通过未来扰动、全遮蔽行策略和 reference 对齐。

## 对应 Notebook、互动图与 starter

在 `notebooks/core/04_attention_mechanics.ipynb` 完成 full-vs-causal heatmap、未来 token 扰动和多头 reshape；在 `docs/interactive/core-concepts.html` 逐格切换 mask，确认每行归一化只覆盖允许 key。然后补完 `exercises/starter/02_causal_attention.py`，运行 `uv run llm-course exercises check 02`。公共核查通过后，再给自己的实现添加 `Tq=1,Tk=5` 与全遮蔽错误策略测试。

## 实验任务

实验 A：对随机 Tiny MHA 比较完整前向两次：第二次只改变位置 `r:` 的 token，断言位置 `<r` 的 logits 在 `1e-6` 内一致，而后缀允许变化。实验 B：枚举 `Tq,Tk=(1,5),(2,5),(5,5)` 打印 allowed mask，手算最后两个例子。实验 C：固定 $D=64$，使用 1、2、4、8 heads，报告参数量不变、单头维度变化和 CPU 时间；不要把速度噪声当理论保证。实验 D：计算 MHA/MQA/GQA 在 `B=1,T=4096,Dh=128,Hq=32,Hkv={32,8,1}` 下每层 fp16 KV bytes，并解释这只是缓存体积，不是端到端延迟。

## 验收 rubric

满分 10 分：因果 mask 与 cache offset 正确 2 分；未来扰动端到端通过 2 分；MHA reshape/concat 与参数量正确 2 分；API bool 语义和全遮蔽策略明确 1 分；MHA/MQA/GQA 边界解释准确 1 分；starter 核查通过 1 分；实验可复现且不夸大 benchmark 1 分。若 Softmax 后 mask、不重新归一化、方阵 mask 错用于 cache，或未来前缀变化，则不通过。

## 一手来源

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)：causal decoder mask 与 multi-head attention 原始定义。
- [PyTorch SDPA 官方文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)：bool mask 语义、`is_causal` 与 GQA 参数。
- [PyTorch `MultiheadAttention` 官方文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html)：MHA 输入输出、padding mask 与 fastpath 条件。
- [Fast Transformer Decoding: One Write-Head Is All You Need](https://arxiv.org/abs/1911.02150)：MQA 与解码带宽动机。
- [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245)：GQA 定义与从 MHA checkpoint uptraining 的原始工作。
