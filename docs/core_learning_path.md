# 15 周核心学习路径

这条路径面向“会写少量 Python、第一次系统学习 LLM”的学习者。完成后，你应当能从文本开始训练并解释一个 Tiny GPT；48 周路线中的现代 Decoder、前沿注意力、MoE、后训练和推理优化是后续进阶模块。

## 开始前的诊断

先完成以下检查。任何一项不会，都先补环境或 Python 基础，不要直接跳到 Transformer。

- 能运行 `uv run python -c "import torch; print(torch.tensor([1, 2]) + 1)"`。
- 能解释函数参数、返回值、列表、字典、循环和类。
- 能看懂 Python traceback 的最后一行。
- 能运行 `uv run pytest -q`，知道失败测试不是“环境坏了”的同义词。
- 能区分标量、向量和矩阵，不要求熟练证明。

环境准备见[环境搭建](00_environment.md)。基础交互实验集中在[核心概念交互实验室](interactive/core-concepts.html)；完成核心路线后，用[架构演化图](interactive/architecture-evolution.html)串起 RoPE、注意力和 MoE。

## 每周使用方法

每周按同一顺序完成：

1. 先写下你对问题的直觉答案。
2. 阅读讲义并标注每个张量形状。
3. 操作交互图，先预测再拖动参数。
4. 关闭参考实现，完成 starter。
5. 运行最小测试；失败时记录最小反例。
6. 最后阅读参考实现和论文。
7. 在 `progress.yaml` 写本周反思和一个仍未解决的问题。

## 周计划

| 周 | 核心问题 | 阅读与实验 | 独立产出 | 完成标准 |
|---|---|---|---|---|
| 1 | 项目环境如何保持可复现？ | [环境](00_environment.md)、`uv run llm-course doctor` | 解释解释器、虚拟环境、依赖锁 | doctor 和 pytest 能运行 |
| 2 | 张量形状如何流动？ | [基础讲义](stages/01_foundations.md)、Notebook 00 | 手写三段矩阵运算形状 | 能指出被求和的维 |
| 3 | 梯度怎样沿计算图传播？ | Notebook 00、`autograd.py` | 手算并验证一个链式法则 | 有限差分与解析梯度一致 |
| 4 | logits 如何变为训练信号？ | [Softmax 交互图](interactive/core-concepts.html#softmax)、starter 01 | 实现稳定 Softmax | 大 logits 不溢出，概率和为 1 |
| 5 | 如何避免“在测试集上学习”？ | [神经语言模型](stages/02_neural_lm.md) | 画训练/验证曲线并诊断 | 能区分欠拟合、过拟合和泄漏 |
| 6 | 非线性为什么必要？ | 两层 MLP 单 batch 实验 | 不用 `nn.Sequential` 训练 MLP | 小数据可过拟合 |
| 7 | next-token prediction 是什么？ | Bigram 与 embedding 实验 | 从文本构造 input/target | target 恰好右移一位 |
| 8 | RNN 为什么推动了注意力？ | RNN 复制任务 | 比较固定窗口与循环状态 | 能说明串行和长梯度问题 |
| 9 | 字符、字节和 token 有何区别？ | `tokenization.py` | 比较中英文字符/字节/token 数 | encode/decode 往返一致 |
| 10 | BPE 怎样合并相邻对？ | Byte BPE 训练与 merge 日志 | 在小语料上手算前三次 merge | 合并结果确定可复现 |
| 11 | 没有位置时会发生什么？ | 位置 embedding 排列实验 | 构造顺序不同但 token 相同的样例 | 能解释 permutation equivariance |
| 12 | Query 如何读取 Key/Value？ | [注意力交互图](interactive/core-concepts.html#attention)、Notebook 01、starter 02 | 实现单头因果注意力 | 与 PyTorch SDPA 数值一致 |
| 13 | 多头与因果掩码如何组合？ | `attention.py`、未来 token 扰动测试 | 实现 causal MHA | 修改未来不影响过去 |
| 14 | Decoder Block 如何形成 GPT？ | [Transformer 讲义](stages/03_transformer.md)、`transformer.py` | 组装 Norm/Attention/MLP/Residual | 前反传、参数量测试通过 |
| 15 | Tiny GPT 是否形成完整闭环？ | Notebook 03、`examples/train_tiny_gpt.py` | 过拟合一个 batch 并分析三个失败案例 | loss 明显下降，能保存、加载和生成 |


## 核心论文节奏

论文条目都在 `papers/catalog.yaml`，不要从 78 篇目录中随机挑选。

| 周 | paper_id | 阅读遍数 | 关注问题 |
|---|---|---|---|
| 7 | `bengio_nplm` | 第一遍 | 神经语言模型相对 n-gram 解决了什么？ |
| 8 | `seq2seq`、`bahdanau_attention` | 第一遍 | 固定状态的瓶颈怎样推动注意力？ |
| 12–14 | `transformer` | 第一遍到第二遍 | QKV、并行训练和复杂度证据是什么？ |
## 核心之后

完成第 15 周后，再按兴趣进入：

- 现代 Decoder：RMSNorm、SwiGLU、RoPE、GQA、[KV Cache 交互图](interactive/core-concepts.html#kv-cache)，再看[RoPE/注意力演化](architecture_evolution.md#位置编码与-rope-演化)。
- MoE：先操作[MoE 路由交互图](interactive/core-concepts.html#moe)，再用[MoE 演化时间轴](interactive/architecture-evolution.html#timeline)理解容量、负载均衡、共享专家与现代组合。
- 后训练：先掌握 SFT 的 token shift 和 answer mask，再学习 LoRA、DPO 和 GRPO。
- 推理：先实现缓存生成，再学习量化、PagedAttention 和块式推测解码。

## 两种检查不要混淆

- `uv run llm-course course check` 检查仓库、课程资产和参考实现是否健康。
- starter 是否由你独立完成，需要运行对应文件并口述实现；仓库健康检查不能替代学习验收。
