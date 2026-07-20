# 第 28 周：可靠评测、去污染与故障诊断

## 课程定位

Scaling Laws 回答“在给定假设下训练 loss 如何随规模变化”，本周独立回答“模型究竟学会了什么，以及分数是否可信”。可靠评测是一条数据与软件管道：样本版本、prompt、chat template、tokenizer、解码、答案解析、随机性、污染和统计方法都会改变读数。本周把能力、实现故障、训练泄漏和随机波动分开诊断，为后续第 29–34 周的新架构比较建立统一尺子。

## 学习目标

完成后应能设计可重复的评测卡；给出随机/多数类/旧 checkpoint 基线；核查 prompt、stop token、padding、答案 parser 与样本计数；用精确、n-gram 和近重复方法筛查训练—评测污染并保存命中证据；按领域、长度、语言和证据位置分桶；报告置信区间和配对比较；区分“题目支持的主张”和超出评测范围的宣传；为 loss spike、异常高分、回归和长上下文失败建立一次一变量诊断顺序。

## 前置

需要理解 train/validation/test 划分、交叉熵、perplexity、准确率、bootstrap、随机种子和 tokenizer。应完成第 22–23 周的数据 lineage、去重与 packing，知道训练语料中的评测题、答案或轻微改写都可能导致污染。还需掌握 cached decode 与 attention mask，否则生成差异可能来自实现而非能力。

## 直觉

把 benchmark 看作测量仪器而不是排行榜。仪器若刻度、温度或样本发生变化，模型没变也会得到不同数字。异常高分首先应怀疑泄漏、模板提示、答案 parser 和缓存；异常低分首先应检查 tokenization、截断、stop、padding 和任务格式。去污染筛查像烟雾报警器：命中表示需要审查，不自动证明模型记住答案；未命中也不能证明完全无泄漏。

## 张量/数据契约

每条结果至少记录 `example_id`、benchmark 与 commit/version、split、原始文本 hash、prompt template hash、few-shot 示例及顺序、tokenizer revision、模型 checkpoint、dtype、设备、最大输入/输出长度、解码参数、seed、原始输出、解析结果和错误类型。聚合结果保存分子、分母与被排除样本，而不是只存百分比。

污染表记录训练快照、评测快照、规范化规则、匹配方法、阈值、命中片段、题干/答案类型、人工决定和审查者。perplexity 必须给出有效 token 数、loss mask 与 tokenizer；不同 tokenizer 的 per-token perplexity不可直接排名，可在固定字节文本上补充 bits-per-byte。

## 推导与机制

二项准确率 $\hat p=c/n$ 的近似标准误为 $\sqrt{\hat p(1-\hat p)/n}$，但小样本或极端概率更适合 Wilson 区间。比较两个模型应对同一题目的成败做配对 bootstrap，保留题目难度相关性。多次尝试 prompt 或 checkpoint 后只报告最好分数会产生多重比较偏差，必须披露选择过程或使用独立保留集。

精确污染用规范化 hash 查找完全相同内容；n-gram 查找局部重叠；MinHash/LSH 或语义检索筛查轻微改写。任何规范化都会改变匹配定义，因此应同时保留原始片段。去污染的决策可以删除样本、单独报告 contaminated/uncontaminated 分数，或降低证据等级，但不能静默丢弃不利样本。

故障诊断遵循从契约到统计的顺序：先用极小手工样本验证 prompt 与 parser，再用固定 checkpoint 验证确定性，随后核查 mask/position/cache，最后才分析模型能力。一次只改一个变量，并保存变更前后原始输出。

## 数值例

100 道二分类题答对 60 道，近似标准误为 `sqrt(.6*.4/100)=4.9%`；61% 与 60% 没有足够证据说明能力不同。若 10 个被训练语料精确命中的样本全部答对，其余 90 题答对 50 题，总分 60%，去除命中后只有 55.6%，应同时报告两者和筛查方法，而非选择更好看的数字。

长上下文任务若 32K 平均 80%，但证据位于末尾时 95%、开头时 45%，平均分掩盖了位置退化。正确做法是按长度与证据位置联合分桶，并给每桶样本数和区间。

## 最小代码

```python
import random

def paired_bootstrap(a, b, rounds=2000, seed=0):
    if len(a) != len(b) or not a:
        raise ValueError("paired, non-empty results required")
    rng = random.Random(seed)
    n = len(a)
    diffs = []
    for _ in range(rounds):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(sum(a[i] - b[i] for i in idx) / n)
    diffs.sort()
    return diffs[int(.025 * rounds)], diffs[int(.975 * rounds)]

def exact_contamination(train_texts, eval_texts):
    train = {" ".join(x.split()) for x in train_texts}
    return [i for i, x in enumerate(eval_texts) if " ".join(x.split()) in train]
```

这是教学 baseline。空白规范化可能误并或漏匹配，bootstrap 未处理层级抽样，真实管道还要支持版本化 artifacts、近重复候选和人工复核。代码输出是审计线索，不是“污染已证明”的法律或因果结论。

## 反例与调试

答案 parser 失败时默认错误会把格式问题当能力问题，默认正确更会制造虚高；必须保留原始输出和错误类。只查题干全文会漏答案、解析和轻微改写泄漏。随机基线若没有按真实标签分布构造，可能低估简单策略。比较 API 模型时 prompt、采样、日期和后端版本不固定，差异不可归因。只报平均分会隐藏语言、长度或群体退化；只跑 needle-in-a-haystack 不能证明长文推理。

训练故障方面，gradient clipping 能压住症状却不能修复错误 token、越界 label 或全 mask batch。恢复 checkpoint 后若 sampler/RNG 没恢复，曲线变化不一定是模型 bug。评测缓存若 key 不含模板、模型 revision 和解码参数，会返回过期结果。

## 主流工作与证据等级

HELM 强调场景、指标与透明报告，PALOMA 对语言模型 fit、数据顺序、污染与跨 tokenizer 比较给出规范；lm-evaluation-harness 提供可审计的开放实现入口；RULER 与 LongBench 展示长上下文需要受控诊断和真实任务共同评估。公开论文和固定 artifacts 是较强证据，官方模型报告需注明“作者报告”，在线排行榜若版本和 prompt 不透明只能作为线索。

## Notebook、互动图与 starter

在 `learning/labs/07_pretraining_systems.ipynb` 中展示原始输出、parser 决策、污染命中和置信区间。互动图允许拖动样本数、真实准确率和污染比例，观察区间与总分如何变化；第二视图按上下文长度和证据位置画热力图。starter 留出 prompt hash、污染匹配、分桶聚合和 paired bootstrap 空缺；核查器用手工 oracle 验证样本分母、mask 与固定 seed。

## 实验

实验一对同一 checkpoint 使用两个等义 prompt 与两个 parser，比较原始输出和最终得分。实验二向训练候选注入精确复制、局部复制和改写样本，测量三种筛查器的召回与误报。实验三构造一次 cache key 缺少 tokenizer revision 的故障并定位。实验四对两个相近 checkpoint 做配对 bootstrap，说明“平均高 1 分”何时不足以宣称提升。所有实验保存失败样本而非只留汇总表。

## 验收 rubric

及格要求是评测卡、随机基线、原始输出和样本分母完整，能发现手工 parser/mask 故障。良好要求是同时做精确与近重复污染筛查、配对区间和多维分桶。优秀要求是建立可重放流水线，报告污染前后结果与人工复核，覆盖能力、稳健性、校准和系统指标，并明确区分论文证据、作者报告与本地实验推断。

## 一手来源

- [HELM: Holistic Evaluation of Language Models](https://arxiv.org/abs/2211.09110)
- [PALOMA: A Benchmark for Language Model Fit](https://arxiv.org/abs/2312.10523)
- [Language Model Evaluation Harness 官方仓库](https://github.com/EleutherAI/lm-evaluation-harness)
- [LongBench](https://arxiv.org/abs/2308.14508)
- [RULER](https://arxiv.org/abs/2404.06654)
