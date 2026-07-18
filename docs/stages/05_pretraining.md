# 阶段五：预训练——模型之外，数据、系统和测量同样重要

这一阶段不追求在个人电脑上复制一次商业级训练，而是建立一条可以缩尺验证、可以审计、
可以恢复的训练链路。完成后，你应能回答四个问题：数据从哪里来、每个 token 经过了什么、
训练资源花在哪里、评测结果在什么条件下成立。

## 六周学习顺序

1. 数据来源、治理和 data card。
2. 过滤、去重、污染检查、混合和 packing。
3. AdamW、warmup、学习率计划和故障诊断。
4. FLOPs、内存、混合精度与分布式切分。
5. Scaling Laws 与计算最优。
6. 可重复评测、消融和训练报告。

每周都遵循“先写预测 → 运行缩尺实验 → 检查数值不变量 → 记录反例”的顺序。规模变小后，
吞吐或质量的绝对数字不能外推；可以复现的是公式、趋势和测量方法。

## 1. 数据不是一个文本文件

一条可审计管道至少记录：

- 来源 URL/数据集、抓取或发布快照、许可状态和获取时间。
- 原始内容哈希、处理代码版本、规则版本和随机种子。
- 每一步输入/输出文档数与 token 数、拒绝原因和抽检样本。
- 语言与脚本、领域、时间、地域等分布在处理前后的变化。
- 训练、验证、评测和保留审计集的生成方式。

“网上可访问”不自动等于“可用于训练”。许可、版权、robots、个人信息和删除请求是不同问题；
课程只讲工程记录方法，不提供法律结论。PII、密钥、恶意代码、有害内容和偏见也要分别处理，
不能用一个模糊的“质量分”代替。

开放案例可以对照 [Dolma](https://arxiv.org/abs/2402.00159)、
[DataComp-LM](https://arxiv.org/abs/2406.11794) 和
[FineWeb](https://arxiv.org/abs/2406.17557)。它们的重要价值不仅是数据规模，还包括处理过程、
消融和公开工具。

### 过滤器也有偏差

规则或模型过滤器会改变训练分布。例如“教育质量”分类器可能更偏好某种文体或高资源语言。
因此必须同时报告保留率和分布差异，并抽检 false positive/false negative。合成数据还要记录生成
模型、prompt、采样参数和验证器；不能把模型生成文本的来源写成“无来源”。

## 2. 去重与污染不是同一个任务

精确去重只移除字节或字符串完全相同的文档。`exact_deduplicate` 保留第一次出现的文档并返回
重复项原始下标，故意不做小写化或空白归一化。任何规范化都会改变“相同”的定义，应作为独立
规则测量误删率。

近似去重寻找网页模板、局部复制或轻微编辑，常用 MinHash/LSH 或 n-gram 指纹。阈值太松会保留
大量重复，太严会删除合法引用、法律文本或常用代码。原始研究见
[Deduplicating Training Data Makes Language Models Better](https://arxiv.org/abs/2107.06499)。

污染检查是拿训练候选数据和评测材料比较。`detect_ngram_contamination` 是词级 n-gram 高召回
筛查器：命中表示“需要审查”，不等于已经证明模型会记忆答案。真实流程还要保存命中片段、匹配
类型和人工决定。只做 train/test 文档级切分不够，因为评测片段可能嵌在更长网页里。

## 3. Tokenizer、混合与 packing

分词器决定实际训练单位。数据卡至少报告：

- 词表和训练语料版本、特殊 token、byte fallback。
- 不同语言/领域的 tokens-per-character 或 fertility。
- BOS/EOS、文档边界、padding 和截断规则。
- 与旧模型比较时是否使用同一 tokenizer。

`deterministic_mixture_sample` 用固定 seed 按权重抽样并保留来源标签，便于核对实际混合比例。
它是有放回教学实现；大规模训练通常使用流式 shard、每来源独立游标和可恢复 sampler state。
只保存配置权重而不保存实际样本来源，无法重放训练。

`pack_documents` 把多个 token 文档装进定长块：

```text
doc A tokens → EOS → doc B tokens → EOS → PAD
label mask:  首 token=0；文档内后续 token/EOS=1；PAD=0
```

新文档首 token 不作为 label，避免模型学习“上一文档 EOS 后必然接下一文档开头”的人工边界。
其他系统也可能允许这种转移；关键不是背一个答案，而是写清数据契约并让 mask 测试覆盖它。

## 4. AdamW、warmup 与训练稳定性

AdamW 将权重衰减与自适应梯度更新解耦：

```text
theta <- (1 - lr * weight_decay) * theta
m     <- beta1 * m + (1-beta1) * grad
v     <- beta2 * v + (1-beta2) * grad^2
theta <- theta - lr * m_hat / (sqrt(v_hat) + eps)
```

`adamw_step_` 用单参数公式和 PyTorch 单步 oracle 对齐。真实训练还要决定哪些参数不做衰减，
例如 norm scale 或 bias；这属于配方，不是 AdamW 公式本身。

`warmup_cosine_lr` 的步数定义是 `0..total_steps`：warmup 从 0 线性到 `max_lr`，随后 cosine
下降到 `min_lr`。日志必须记录 optimizer step，而不是把 microbatch 次数误当 step。

建议诊断顺序：

1. 固定一个 batch，确认模型能过拟合。
2. 检查 target shift、mask、文档边界、token 范围和损失分母。
3. 检查 NaN/Inf、梯度/激活范数和 loss spike 前的数据来源。
4. 检查 global batch、gradient accumulation、学习率和初始化。
5. 验证 checkpoint 是否恢复模型、优化器、scheduler、sampler 和 RNG 状态。
6. 最后才扩大模型或更换优化器。

## 5. 资源账本与分布式训练

至少分别估算：

- 参数权重；混合精度下还可能有 FP32 master weights。
- 梯度。
- Adam 一阶/二阶矩。
- 激活和临时 kernel workspace。
- 通信 buffer、碎片和 allocator 峰值。

`training_memory_ledger` 返回每设备的显式账本，并可分别模拟参数、梯度、优化器状态是否分片。
它不包含框架临时内存，因此是预算起点而不是 OOM 保证。

需要区分：

- FLOPs：做多少计算。
- 显存：状态能否装下。
- 带宽/延迟：数据搬运需要多久。
- MFU：实际模型 FLOPs 相对硬件峰值的利用率。

常见切分：

| 方法 | 主要切分对象 | 典型通信 | 主要目的 |
|---|---|---|---|
| DDP | batch；模型复制 | gradient all-reduce | 提高吞吐 |
| ZeRO/FSDP | optimizer、gradient、parameter | all-gather/reduce-scatter | 降低每卡状态内存 |
| Tensor Parallel | 层内矩阵/head | 高频 all-reduce/all-gather | 单层放不下 |
| Pipeline Parallel | 层 | stage 间激活；有 bubble | 模型纵向切分 |
| Sequence/Context Parallel | 序列 | ring/all-to-all 等 | 长序列激活/attention |

ZeRO 的原始分析见 [ZeRO](https://arxiv.org/abs/1910.02054)，层内模型并行见
[Megatron-LM](https://arxiv.org/abs/1909.08053)。具体 API 会变化，实践时以
[PyTorch FSDP2 文档](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html)
为准。算法能分片不代表在任意拓扑上更快；必须测量通信、bubble 和端到端 step time。

## 6. Scaling Laws：拟合关系，不是自然常数

一维幂律可以写成：

```text
L(x) = E + A * x^(-alpha)
```

计算最优分析通常同时考虑非 embedding 参数 `N` 和训练 token `D`：

```text
L(N, D) = E + A / N^alpha + B / D^beta
C approximately 6 * N * D
```

`6ND` 是 dense Transformer 训练 FLOPs 的常用粗略量级，不是对所有架构、序列长度和实现都
精确。应区分 [Kaplan 等人的 Scaling Laws](https://arxiv.org/abs/2001.08361) 与
[Chinchilla](https://arxiv.org/abs/2203.15556) 的计算最优结论。数据质量、架构、优化器和拟合
范围改变后，系数会改变；部署还可能需要考虑长期推理成本，见
[Beyond Chinchilla-Optimal](https://arxiv.org/abs/2401.00448)。

缩尺报告必须包含观测范围、每点种子、拟合方法、残差、置信区间和外推区域。不能从三次小模型
训练得到“某个固定 tokens/parameter 适用于所有模型”的结论。

## 7. 评测与消融

`evaluate_token_logits` 输出 loss、perplexity、token accuracy 和有效 token 数。per-token
perplexity 依赖 tokenizer，不能直接比较不同词表；必要时固定 tokenizer，或报告 bits per byte。
[PALOMA](https://arxiv.org/abs/2312.10523) 对污染、训练顺序和跨 tokenizer 比较给出了可操作规范。

任务评测还要固定并记录：

- 数据集和评测脚本版本、prompt/chat template、few-shot 示例和顺序。
- 解码参数、随机种子、样本数、置信区间和失败样本。
- 训练数据截止时间、污染检查方式和被排除样本。
- 能力、校准、稳健性、公平、安全、吞吐和显存；不要压成一个没有解释的总分。

一次一变量消融只有在数据、训练 token、tokenizer、模型预算、优化器、seed 和评测脚本等其余
条件可比时才成立。若同时改变数据质量与数量，应把结论写成端到端配方比较，而不是单组件因果。

## 阶段代码入口与验收

代码位于 `src/llm_from_scratch/training.py`，测试位于 `tests/test_training.py`。验收时应能：

1. 解释精确去重与 n-gram 污染筛查的边界，并展示被移除下标。
2. 用来源标签核对确定性混合比例，说明 sampler state 如何恢复。
3. 画出 packed tokens 与 label mask，证明文档首 token 和 padding 不计损失。
4. 让教学 AdamW 与 PyTorch 单步一致，画出 warmup-cosine 曲线。
5. 写出显存账本假设，比较 DDP 与状态分片，而不声称估算就是峰值。
6. 报告 loss/perplexity/accuracy 的样本数和 tokenizer，不跨 tokenizer 误比 perplexity。
7. 对 scaling 拟合报告残差、置信区间和外推限制。

推荐进一步参考 [Stanford CS336](https://cs336.stanford.edu/spring2025/index.html)，但课程网页或
博客用于帮助理解，技术结论仍应回到原论文、公开代码和可复现实验。
