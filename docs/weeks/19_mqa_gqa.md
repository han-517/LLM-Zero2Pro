# 第 19 周：MQA 与 GQA——共享 K/V，保留 Query 容量

## 课程定位

多头注意力让每个 Query head 拥有自己的 K、V；自回归解码时，这些历史 K/V 会在每层长期驻留。本周研究 MQA 与 GQA 如何减少 KV heads，并用数值等价实验确认“共享”不等于随意平均。它连接结构设计和推理内存，是下一周 KV Cache 的必要前置。

## 学习目标

- 画出 MHA、MQA、GQA 的 Query-to-KV head 映射。
- 推导参数量、KV Cache 字节数和分组计算 shape。
- 写出不物化 `repeat_interleave` 的 GQA reference，并与显式展开 oracle 对照。
- 识别质量、带宽、并行实现和 kernel 支持之间的权衡。

## 前置

需要熟悉缩放点积注意力、head split/merge、broadcast/einsum 与 RoPE。应能从 `[B,T,D]` 推出 Q 为 `[B,Hq,T,Dh]`，并理解 K/V 在写入 cache 前已经完成位置变换。

## 直觉

生成新 token 时，模型只产生少量新 Query，却要读取所有历史 K/V。MHA 给每个 Query head 一份独立 K/V，表达自由但缓存大；MQA 让所有 Query heads 共用一组 K/V，最省缓存；GQA 把 Query heads 分组，每组共享一个 KV head，在二者之间折中。Query projection 仍保留 `Hq` 个 heads，所以共享的是被检索的历史表示，不是把所有 Query 也合并。

## 张量/数据契约

`Q:[B,Hq,Tq,Dh]`，`K:[B,Hkv,Tk,Dh]`，`V:[B,Hkv,Tk,Dv]`。必须满足 `Hq>=1`、`Hkv>=1`、`Hq % Hkv == 0`。每个 KV head 服务 `G=Hq/Hkv` 个连续 Query heads；课程实现 reshape 为 `[B,Hkv,G,Tq,Dh]`。因果与 padding mask 要能广播到 `[B,Hkv,G,Tq,Tk]`。MQA 是 `Hkv=1` 的合法边界，不能因 Python 的 `or` 默认值把 0 静默解释成 `Hq`。

## 推导与机制

忽略 bias，Q projection 参数约 `D·Hq·Dh=D²`；K/V 参数由 MHA 的各 `D²` 变为各 `D·Hkv·Dh`。每层缓存字节：

\[
B\,T\,2\,H_{kv}\,D_h\,s,
\]

其中 2 表示 K 和 V，`s` 是每元素字节。全模型乘层数 `L`。GQA 分数可写为

\[
S_{b,h,g,q,k}=Q_{b,h,g,q,:}K_{b,h,k,:}^{\top}/\sqrt{D_h}.
\]

显式重复 K/V 得到与分组 einsum 相同的数学结果，但会在教学代码中制造不必要的中间张量。生产 kernel 可以在索引层共享，不必真的复制。

## 数值例

设 `B=1,L=32,T=8192,Hq=32,Dh=128,FP16`。MHA 每层 K/V 为 `8192×2×32×128×2` 字节，约 128 MiB，全模型约 4 GiB。GQA 取 `Hkv=8` 后为四分之一，约 1 GiB；MQA 取 1 后约 128 MiB。这个估算不含 allocator 对齐、page metadata、batch padding 和其他激活，报告时必须写清假设。

## 最小代码

```python
def grouped_scores(q, k):
    # q [B,Hq,Tq,D], k [B,Hkv,Tk,D]
    b, hq, tq, d = q.shape
    hkv, tk = k.shape[1], k.shape[2]
    if hkv < 1 or hq % hkv:
        raise ValueError("Hq 必须能被非零 Hkv 整除")
    grouped_q = q.reshape(b, hkv, hq // hkv, tq, d)
    return torch.einsum("bhgqd,bhkd->bhgqk", grouped_q, k) / d**0.5
```

这是清晰 reference。生产实现还要融合 mask、online softmax、value 聚合、RoPE、cache layout 和张量并行；显式 `repeat_interleave` 只应作为小张量 oracle，不应进入真实性能路径。

## 反例与调试

若 reshape 前的 head 排列与分组假设不同，输出 shape 正确但 Query 会读错 KV head。用每个 KV head 填不同常数可快速定位。若 cached logits 与完整前向不一致，检查 K 是否在写 cache 前应用了正确 position ID 的 RoPE。若 mask 为 `[B,1,Tq,Tk]`，在五维分组分数上要显式插入 group 轴。全遮蔽行仍应归零。性能实验若先物化重复 K/V，会掩盖 GQA 的内存优势。

## 主流工作与证据等级

MQA 原论文首先系统展示共享 K/V 的快速增量解码；GQA 论文研究从 MHA checkpoint 转换与质量/速度折中，属于基础实验。Llama 2、Qwen、Gemma、DeepSeek 等公开报告采用 GQA 或相关共享策略，属于公开模型证据。不同模型的 `Hkv/Hq` 比例依规模、训练预算和 kernel 而异，不能把 1:4 当作通用最优。

## Notebook、互动图与 starter

在 `docs/interactive/architecture-lab.html` 切换 MHA/MQA/GQA 并改变序列长度；在 `notebooks/core/06_modern_decoder.ipynb` 输出缓存 shape 与字节数；完成 `exercises/starter/08_grouped_query_attention.py`。互动估算默认一层一 batch，只用于直觉，正式交付必须调用通用预算函数。

## 实验

先用随机小张量比较分组实现与显式重复，覆盖 `Hkv=Hq、Hkv=Hq/2、Hkv=1`、full/cached、bool mask 与全遮蔽行。再训练相同宽度的 MHA/GQA/MQA Tiny GPT，报告参数、验证 loss、cache bytes 和逐 token latency。CPU 墙钟只作为课程环境数据，不外推到 GPU fused kernel。

## 验收 rubric

- 35%：所有 head 比例、mask 和 cache 数值 oracle 通过。
- 25%：缓存公式包含 B、L、T、dtype，并与实际张量字节吻合。
- 25%：公平报告质量、内存和延迟，不只给压缩比。
- 15%：明确分组 reference、展开 oracle 与生产 kernel 的边界。

## 一手来源

- [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150)
- [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245)
- [Llama 2 Technical Report](https://arxiv.org/abs/2307.09288)
- [Gemma 2 Model Card](https://ai.google.dev/gemma/docs/model_card_2)
