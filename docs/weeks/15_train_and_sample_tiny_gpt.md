# 第 15 周：训练、保存并采样 Tiny GPT

## 课程定位

本周把原始文本、tokenizer、causal decoder、next-token loss、优化、checkpoint 和采样接成第一个完整闭环。Tiny GPT 的价值是做“可控缩尺实验”：先证明 shape、因果性、shift、保存加载与生成算法正确，再为后续现代 decoder 和大规模预训练建立可靠基线。小模型生成不流畅很正常；验收重心是可复现的工程证据，不是挑一段最好看的文本。

## 学习目标

你应能构造不跨文档泄漏的 `(input,target)`；把 `[B,T,V]` logits 与右移 token 对齐；让单 batch 过拟合；记录训练/验证 loss 与梯度有限性；保存配置、tokenizer、模型、优化器和随机状态；加载后复现 logits；实现 greedy、temperature 与 top-k sampling；解释温度为 0、超上下文、EOS、随机 seed 和 cache 开关的边界；用失败案例而非主观文风评估模型。

## 前置知识与资产

必须完成第 9–14 周，尤其是 tokenizer 往返、causal 未来扰动和 Pre-Norm block。主实验为 `notebooks/core/05_tiny_gpt.ipynb`；互动入口 `docs/interactive/index.html`，架构流用 `docs/interactive/architecture-evolution.html`。starter 为 `exercises/starter/13_tiny_gpt.py`，核查 `uv run llm-course exercises check 13`。所有默认实验使用仓库内/代码内小语料、CPU/offline，不下载预训练权重。

## 自洽直觉

自回归语言模型只学一个任务：给定前缀，预测下一个 token 的条件分布。训练把整条序列并行展开，因果 mask 保证每个位置只用前缀；生成则一次选一个 token，再把它接回前缀。两条路径必须共享相同 tokenizer、位置规则、权重和 mask。单 batch 过拟合像单元测试：如果足够容量仍记不住几十个 token，先查 shift、mask、学习率或 dropout，别立即扩模型。

## 张量/数据契约

文档 tokenize 后得到 ids；在单文档内切长度 `T+1` 的块，`x=chunk[:-1]`、`target=chunk[1:]`，批量为 `LongTensor[B,T]`。模型输出 `logits[B,T,V]`，交叉熵输入可 reshape 为 `[B*T,V]`，target 为 `[B*T]`，范围 `[0,V)`；padding 需用明确 `ignore_index` 且 attention mask 一致。训练模式默认不创建 KV cache，以免保存每层中间状态；生成可用 cache，但 cached 与 full logits 应对齐。checkpoint 至少保存模型 config、`state_dict`、tokenizer 标识和 step，加载使用同一词表大小与 block size。

## 推导/机制：目标函数与采样

对 token 序列 $x_{1:T}$，自回归分解

$$p(x_{1:T})=\prod_{t=1}^{T}p(x_t\mid x_{<t}),\qquad
L=-\frac{1}{N}\sum_{n=1}^{N}\log p(y_n\mid context_n).$$

`logits[:,t]` 预测 `ids[:,t+1]`，不是当前位置本身。若均匀猜测 $V$ 类，期望 cross-entropy 为 $\log V$；明显低于它只是 sanity signal，不等于泛化。

采样温度 $\tau>0$：$p_i=softmax(z_i/\tau)$；$\tau<1$ 更尖锐，$\tau>1$ 更平坦。top-k 先保留最大 k 个 logits，其余设为 $-\infty$，再归一化采样。`temperature=0` 数学上不可除，应明确定义为 greedy `argmax`，而不是加任意 epsilon。采样改变输出分布，不修复模型知识或训练缺陷。

## 手算/数值例

若词表三类 logits 为 `[2,1,0]`，温度 1 的未归一化权重约 `[7.389,2.718,1]`，概率约 `[0.665,0.245,0.090]`；温度 0.5 等价 logits `[4,2,0]`，概率约 `[0.867,0.117,0.016]`。top-k=2 会删除第三类，再得到约 `[0.731,0.269,0]`。训练 shift 例：ids `[BOS,a,b,EOS]` 的 inputs 为 `[BOS,a,b]`，targets 为 `[a,b,EOS]`；若 targets 未右移，模型能通过复制当前 embedding 形成虚假的低损失。

## 最小可运行代码

下面只实现正确的 loss 对齐与可复现 top-k 采样，可直接用于任何返回 logits 的 Tiny GPT。

```python
import torch
import torch.nn.functional as F

def next_token_loss(logits, token_ids):
    if logits.shape[:2] != token_ids.shape:
        raise ValueError("logits 前两维必须等于 token_ids")
    pred = logits[:, :-1, :].contiguous()
    target = token_ids[:, 1:].contiguous()
    return F.cross_entropy(pred.view(-1, pred.size(-1)), target.view(-1))

def sample_logits(logits, temperature=1.0, top_k=None, generator=None):
    if temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)
    if temperature < 0:
        raise ValueError("temperature 必须非负")
    z = logits / temperature
    if top_k is not None:
        if not 1 <= top_k <= z.size(-1):
            raise ValueError("top_k 超出词表")
        cutoff = torch.topk(z, top_k, dim=-1).values[..., -1, None]
        z = z.masked_fill(z < cutoff, float("-inf"))
    return torch.multinomial(torch.softmax(z, -1), 1, generator=generator)

g = torch.Generator().manual_seed(7)
z = torch.tensor([[2.0, 1.0, 0.0]])
print(sample_logits(z, temperature=1.0, top_k=2, generator=g))
```

top-k 边界并列时可能保留超过语义上预期的同值项；若需要严格 k 个 index，应基于 `topk.indices` scatter mask，并为并列规定策略。

## 反例/调试

错误一：未 shift target，loss 异常快速下降；打印一条输入/标签对逐位检查。错误二：把整个语料先拼接再随机切 train/val，相邻片段高度重复造成泄漏；先按文档分割再 chunk。错误三：训练时模型创建/复用 cache，旧 batch 状态污染新 batch且内存增长；训练显式 `use_cache=False`。错误四：保存只有权重没有 tokenizer/config，加载后 id 语义错。错误五：加载往返只比较生成文本，采样随机性掩盖差异；应在 eval 模式比较固定输入 logits。错误六：top-k 后忘记重新 Softmax，或 temperature 对 probabilities 再除。错误七：生成超过 block size 时静默裁剪却沿用错误 position offset；明确滑窗策略及模型实际可见上下文。

## 主流工作与边界

GPT-2 代表 learned absolute position、decoder-only、自回归预训练的经典配方；现代 LLM 常换成 RoPE、RMSNorm、SwiGLU、GQA，并使用更大且精心治理的数据、分布式优化与高效 attention。架构正确不是规模能力的充分条件：数据量/质量、tokenizer、计算预算和评测决定结果。经验 scaling laws 描述特定范围内 loss 随模型、数据、compute 的趋势，不保证任意小语料扩参都有收益，也不是跳过数据切分的理由。Tiny GPT 不用于事实可靠性、安全关键任务或与闭源大模型直接做能力对标。

## 对应 Notebook、互动图与 starter

按 `notebooks/core/05_tiny_gpt.ipynb` 的顺序完成：单 batch 过拟合、短语料 train/val、checkpoint 往返、greedy/temperature/top-k 生成。用 `docs/interactive/architecture-evolution.html` 确认 Tiny GPT 与现代 decoder 的组件差异，避免把教学基线误叫当前完整配方。填写 `exercises/starter/13_tiny_gpt.py` 并运行 `uv run llm-course exercises check 13`；训练默认不创建 cache，公共核查之外再写 checkpoint logits 一致性测试。

## 实验任务

实验 A：固定 1 个 batch，关闭 dropout，训练到 loss 显著低于 `log(V)`；保存 loss 曲线、梯度范数和预测准确率，若失败逐项排查。实验 B：按文档切 train/val，在 3 个 seed 下报告两条曲线，禁止用 test 调参。实验 C：保存后新建模型进程加载，对固定输入断言 logits 在 `1e-6` 容差内一致，并验证 tokenizer 哈希。实验 D：同一 prompt 用 greedy、`tau={0.7,1.0,1.3}` 与 `top_k={5,20}` 各生成多样本，报告重复、非法 decode、EOS 与上下文截断，不以单个最好样本排名。

## 验收 rubric

满分 10 分：数据边界与 next-token shift 正确 2 分；单 batch 过拟合且梯度有限 2 分；train/val 无泄漏、多 seed 1 分；checkpoint 配置/tokenizer/权重往返与 logits 一致 2 分；采样边界和随机性控制正确 1 分；starter 核查通过 1 分；三个失败案例有机制解释而非只贴文本 1 分。若 target 未 shift、测试集调参、只保存权重导致无法复现，或训练复用 cache，则不通过。

## 一手来源

- [OpenAI GPT-2 技术报告](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)：decoder-only 自回归语言模型与零样本实验。
- [OpenAI GPT-2 官方代码](https://github.com/openai/gpt-2)：模型、采样、tokenizer 和 checkpoint 的原始实现。
- [PyTorch `CrossEntropyLoss` 官方文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)：class-index target、ignore index 与 reduction 契约。
- [PyTorch `multinomial` 官方文档](https://docs.pytorch.org/docs/stable/generated/torch.multinomial.html)：按非负权重采样与 generator 行为。
- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)：模型、数据与计算对交叉熵的经验尺度关系及适用背景。
