# 第 23 周：过滤、去重、混合与 Packing

## 课程定位

本周把治理 manifest 变成可测的数据变换。过滤决定保留哪些分布，去重控制重复和记忆，混合决定来源权重，packing 决定 token 如何进入固定长度训练块。这四步彼此影响：先后顺序或边界 mask 错误，会让同一语料产生不同训练目标。

## 学习目标

- 区分精确、近似、文档级与子串级去重。
- 用 MinHash/LSH 直觉解释阈值召回与误删的权衡。
- 计算来源混合后的实际 token 比例并记录上/下采样。
- 实现无跨文档泄漏的 packing、loss mask 和 position ID 契约。

## 前置

需要会 n-gram、Jaccard 相似度、哈希、EOS、causal mask 与 batch collate。应理解“拼进同一张量”不等于“允许跨文档 attention”，也不等于“所有 token 都计算 loss”。

## 直觉

精确去重像删除完全相同的复印件；近似去重像识别只改了页眉或少量字符的版本。阈值越激进，重复越少，但可能误删合法引用、模板语言或低资源文本。packing 像把不同长度货物装进固定集装箱：提高填充率，却必须用边界标记防止前一文档成为后一文档的上下文。混合权重则决定每种货物被看到多少次。

## 张量/数据契约

tokenized 文档是变长 `list[int]`，末尾显式 EOS。packed batch 至少返回 `input_ids:[B,T]、targets:[B,T]、loss_mask:[B,T]、segment_ids:[B,T]、position_ids:[B,T]`。若禁止跨文档 attention，还需 block-diagonal bool mask `[B,1,T,T]`；若采用 EOS 连续流训练，则必须在报告中明确允许跨边界上下文。padding 位置 loss mask 为 0。去重输出要保留代表文档 ID 与被合并成员列表。

## 推导与机制

文档 shingle 集合 A、B 的 Jaccard 为 `|A∩B|/|A∪B|`。MinHash 通过多个随机哈希最小值近似该相似度，LSH 用 banding 降低候选比较量；它是概率候选器，不是相似度真值。来源采样若权重 `w_s`，归一化概率为 `p_s=w_s/Σw`，但实际 token 比例还受文档长度和耗尽/重复策略影响。Packing 利用率为 `非 padding token/(B·T)`，必须与跨段污染率同时报告。

## 数值例

文档 A 的 3-gram 集为 `{abc,bcd,cde,def}`，B 为 `{abc,bcd,cde,xyz}`，交集 3、并集 5，Jaccard 为 0.6。阈值 0.8 不会合并，0.5 会进入近重复簇。长度 `[6,5,3]` 的文档装入 `T=8`：朴素 padding 用 24 槽、14 token，利用率 58.3%；合理 packing 可用两个块 16 槽，利用率 87.5%，但第二文档起点必须重置 position 或提供清楚的连续位置策略。

## 最小代码

```python
def pack_documents(documents, block_size, eos_id):
    blocks, current, segments = [], [], []
    segment = 0
    for doc in documents:
        tokens = [*doc, eos_id]
        while tokens:
            take = min(block_size - len(current), len(tokens))
            current.extend(tokens[:take])
            segments.extend([segment] * take)
            tokens = tokens[take:]
            if len(current) == block_size:
                blocks.append((current, segments))
                current, segments = [], []
        segment += 1
    return blocks, (current, segments)
```

这只是顺序装箱 baseline，未做最优 bin packing、分布式确定性 sharding 或 block-diagonal kernel。生产管道需要并行 reader、checkpointable iterator、样本顺序恢复和高效 packed attention。

## 反例与调试

在 split 之后分别去重会让近重复跨 train/eval；应在保留评测隔离规则下做全局污染检查。小写化后精确去重改变了“精确”的定义，应作为独立规范化规则。Packing 常见 off-by-one 是 targets 跨 EOS 指向下一文档；画 `input/target/segment/loss_mask` 四行即可发现。只统计文档混合比例会被长度差欺骗，应在实际 emitted tokens 上计数。

## 主流工作与证据等级

Deduplicating Training Data 论文给出重复、记忆和评测重叠实验，是基础证据。FineWeb 与 Dolma 公布多阶段过滤/去重消融，是开放数据工程证据。具体 MinHash 参数和质量阈值依语言、领域和目标，公开案例不能直接当作通用最优。框架提供 sequence packing 并不自动保证 segment mask 正确，必须查看官方实现契约。

## Notebook、互动图与 starter

在 `docs/interactive/training-and-alignment.html` 观察过滤保留率；在 `notebooks/core/07_pretraining_systems.ipynb` 可视化近重复簇、来源 token 比和 packing 利用率；完成 starter `14` 的去重与数据 pipeline。互动漏斗不展示误删样本，正式报告必须附抽检。

## 实验

构造包含完全重复、模板改写、合法引用和低资源文本的小语料。比较 exact、字符 n-gram Jaccard 与 MinHash 候选，人工标注 100 对计算 precision/recall。然后比较 padding、连续 EOS packing、segment-isolated packing 的利用率和 loss；用故意跨段可见的 mask 作为污染反例。

## 验收 rubric

- 30%：去重定义、簇成员和保留策略可审计。
- 25%：packing 的 input/target/mask/segment/position 契约正确。
- 25%：报告 token 级混合与利用率、误删/漏删抽检。
- 20%：说明小规模 baseline 与互联网规模索引、packed kernel 边界。

## 一手来源

- [Deduplicating Training Data Makes Language Models Better](https://arxiv.org/abs/2107.06499)
- [The FineWeb Datasets](https://arxiv.org/abs/2406.17557)
- [Dolma: an Open Corpus of Three Trillion Tokens](https://arxiv.org/abs/2402.00159)
- [DataComp-LM](https://arxiv.org/abs/2406.11794)
