# 第 10 周：从零实现 Byte-level BPE

## 课程定位

第 9 周的 byte tokenizer 永不 OOV，却把常见词也拆成许多 token；本周用 Byte Pair Encoding（BPE）学习一组可复现的相邻 pair 合并，在“256 个 byte 的完备覆盖”和“常见片段的紧凑表示”之间搭桥。重点不是调用现成库，而是把训练、编码、解码、特殊 token 与模型版本都做成显式算法。后续模型的序列长度、词表大小、embedding 参数和数据吞吐都受这里的决定约束。

## 学习目标

你要能从 byte 序列统计相邻 pair，按确定规则选择 winner，应用一次不重叠合并并保存 rank；解释训练阶段与编码阶段的差别；证明 byte 基础上的支持域没有普通文本 OOV；对中英、emoji、空格和随机 bytes 做往返；计算 tokenizer fertility 与压缩率；指出 BPE、WordPiece、Unigram、纯 byte 的目标与边界。最终独立补完 starter，不把参考实现复制进去。

## 前置知识与资产

先完成第 9 周，能区分 Unicode 与 byte，并理解 Python tuple/dict。主实验是 `learning/labs/03_tokenization_and_bpe.ipynb`；模板为 `learning/labs/starter/06_byte_bpe.py`，核查命令 `uv run llm-course exercises check 06`。互动入口 `learning/readings/interactive/index.html` 用于逐步观察 pair 频数和 merge；本章代码只用标准库，CPU/offline 可执行。

## 自洽直觉

把每个 byte 当初始积木。若语料里 `t h` 经常相邻，就新增一个代表 `th` 的积木；下一轮又可能把 `th e` 合为 `the`。训练产生的是有顺序的 merge 规则，而非一次分词结果。编码新文本时，应从初始 byte pieces 开始，反复应用“当前可合并 pair 中 rank 最小者”，而不是重新统计新文本的最高频 pair。这样同一模型文件对任意输入给出确定切分。解码只需把每个 piece 展开回原始 bytes 并拼接。

## 张量/数据契约

训练语料先变成若干独立 `list[bytes]` 或 byte-id 序列；不得跨文档、跨样本统计边界 pair，除非显式插入 EOS。基础 token id `0..255` 对应单 byte；第 $m$ 次 merge 新增 id `256+m`，piece 表保存 `id -> bytes`，merge 表保存 `(left_id,right_id) -> (new_id,rank)`。`encode(str)->list[int]` 的 id 范围是 `[0,V)`；`decode(ids)->str` 必须校验未知 id，并在 bytes 拼接后严格 UTF-8 解码。special token 建议从独立保留区分配，绝不参与普通 merge。下游 batch 才转成 `LongTensor[B,T]`。

## 推导/机制：训练、编码与复杂度

设当前语料为序列集合 $S$，pair 频数

$$c(a,b)=\sum_{s\in S}\sum_{i=1}^{|s|-1}\mathbf{1}[s_i=a\land s_{i+1}=b].$$

每轮选 $p^*=\arg\max_p c(p)$，但 `argmax` 并不规定平局；为可复现，应使用键 `(-count, left_id, right_id)` 的最小项。应用 merge 时从左到右扫描：命中 pair 就输出新 id 并前进 2，否则复制左项并前进 1。`[a,a,a]` 合并 `(a,a)` 只能得到 `[aa,a]`，不能让中间 `a` 同时参与两次。

编码时所有规则已有 rank。朴素实现每轮扫描所有 pair，找可用的最小 rank 并合并，适合教学；生产 tokenizer 会用 heap、缓存或专用库加速。无论优化怎样，规范结果应一致。词表大小约为 `256 + merges + special_count`，但某些 merge 可能因预分词边界而从不跨越空格。

## 手算/数值例

语料只有 `[a,b,a,b]` 与 `[a,b,c]`。初始 pair 计数：`(a,b)=3`、`(b,a)=1`、`(b,c)=1`，第一轮合并 `ab=X`，序列变为 `[X,X]` 与 `[X,c]`。此时 `(X,X)=1`、`(X,c)=1`；若按 id 字典序平局，winner 由明确的 pair key 决定。假设第二轮合并 `(X,c)=Y`，新输入 `abcab` 初始为 `[a,b,c,a,b]`，先按 rank 0 得 `[X,c,X]`，再按 rank 1 得 `[Y,X]`。若错误地在新输入重算频数，可能得到不同切分，破坏模型契约。

## 最小可运行代码

下面只演示一次确定合并与解码底座，完整多轮训练留给 starter。

```python
from collections import Counter

def pair_counts(sequences):
    counts = Counter()
    for seq in sequences:
        counts.update(zip(seq, seq[1:]))
    return counts

def merge_once(seq, pair, new_id):
    out, i = [], 0
    while i < len(seq):
        if i + 1 < len(seq) and (seq[i], seq[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(seq[i])
            i += 1
    return out

docs = [list(b"abab"), list(b"abc")]
counts = pair_counts(docs)
winner = min(counts, key=lambda p: (-counts[p], p[0], p[1]))
merged = [merge_once(s, winner, 256) for s in docs]
assert winner == (ord("a"), ord("b"))
print(counts, merged)
```

请额外写 `expand(id)` 或直接维护 `piece_bytes[id]`，并断言所有合法字符串在 encode/decode 后 byte 完全一致，而不只是显示结果相似。

## 反例/调试

常见 bug 一是用 `Counter.most_common(1)` 却未规定平局，语料顺序变化会导致模型哈希变化；打乱文档顺序测试 merge 表应相同。二是扫描命中后只前进 1，产生重叠 merge。三是训练每轮只更新局部计数却漏掉邻居 pair；教学实现可先全量重算保证正确，再优化。四是用字符串 piece 拼接，组合字符与特殊 token 容易混淆；内部保持 bytes。五是把 merge 次数当最终词表大小却忽略 special token 与重复/停止条件。六是 `decode` 对未知 id 静默输出 replacement，应明确抛错。七是跨文档合并了末 byte 与下一文档首 byte，制造永远不应出现的 piece。

## 主流工作与边界

Sennrich 等把 BPE 用于开放词表 NMT；GPT-2 的 byte-level BPE 用 byte 映射加正则预分词，具体行为不等同于“在整份 bytes 上任意跨边界 merge”。SentencePiece 支持 BPE 与基于概率模型的 Unigram，后者从候选词表出发删减并可进行采样分词。WordPiece 常以似然提升而非纯频数选择合并。现代 LLM 依然大量使用 BPE/SentencePiece 类 tokenizer，但词表设计是系统折中：较大词表缩短序列，却增大 embedding 与 LM head，且低资源语言的 bytes/token 可能明显偏高。Byte fallback 解决覆盖，不保证公平、语义合理或对注入攻击安全。

## 对应 Notebook、互动图与 starter

先在 `learning/labs/03_tokenization_and_bpe.ipynb` 观察每轮 winner 与语料长度，再用 `learning/readings/interactive/index.html` 的 tokenizer/BPE 视图手工改变语料，尤其检查平局和重叠。然后只编辑 `learning/labs/starter/06_byte_bpe.py` 的 TODO：pair 统计、确定 winner、单次应用和编码逻辑。运行 `uv run llm-course exercises check 06`；通过后再与参考模块比较，不能用硬编码公共样例过关。

## 实验任务

实验 A：对同一离线语料训练 0、20、100 次 merge，报告词表大小、平均 bytes/token、最大序列长度及训练时间。实验 B：反转文档顺序、改变 dict 插入顺序，验证 merge 表逐项相同；若不同，给出首个分叉轮次。实验 C：用训练中未见的中文、emoji、拼写错误和长数字做严格往返，比较字符、byte 与 BPE 长度。实验 D：人为允许/禁止跨空格 merge，展示对压缩和词边界的影响；结论必须注明预分词策略，不能把一种实现推广成全部 BPE。

## 验收 rubric

满分 10 分：训练/编码算法区分正确 2 分；非重叠 merge 和 rank 规则正确 2 分；文档顺序变化仍确定 1 分；复杂 Unicode 严格往返且未知 id 报错 2 分；指标同时报告词表成本与序列收益 1 分；starter 核查通过 1 分；能比较 BPE/WordPiece/Unigram 的目标而不混称 1 分。若跨文档统计、使用 lossy decode 或把新文本频数用于编码，直接不通过。

## 一手来源

- [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909)：将 BPE 引入子词建模的原始论文。
- [Sennrich `subword-nmt` 官方实现](https://github.com/rsennrich/subword-nmt)：merge 与应用规则的参考实现。
- [OpenAI GPT-2 官方 encoder](https://github.com/openai/gpt-2/blob/master/src/encoder.py)：byte-level BPE、rank 和预分词的真实代码。
- [SentencePiece 原论文](https://arxiv.org/abs/1808.06226)：BPE/Unigram、原始句子训练与可逆分词。
- [SentencePiece 官方仓库](https://github.com/google/sentencepiece)：模型文件、normalization 与 subword regularization 的实现边界。
