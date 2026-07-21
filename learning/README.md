<!-- 由 `uv run llm-course course path --write` 生成，请勿手动维护。 -->
# LLM-Zero2Pro 学习目录（第 1–48 课）

这是课程唯一入口。不要随机挑 Notebook：从第 1 课开始，完成本课验收后再进入下一课。
课程主线严格聚焦文本 LLM；多模态位于文末选修区，不计入 48 课毕业要求。

## 第一次使用

1. 在仓库根目录运行 `uv sync` 和 `uv run llm-course doctor`。
2. VS Code 用户运行 `uv run llm-course vscode`；Explorer 只显示 `learning/`。
   手动方式是 `code LLM-Zero2Pro.code-workspace learning/README.md`。
3. 在 VS Code 选择项目 `.venv` 解释器；打开 `.ipynb` 时选择同一个 Kernel。
4. 每个实验只需选择 `.ipynb` 或同名 `# %%` Python 文件之一，不要重复完成。
5. 互动 HTML 使用系统浏览器或 VS Code Live Preview 打开。

环境细节见 [VS Code 指南](../setup/vscode.md)、[Windows/macOS 环境指南](../setup/environment.md)和[JupyterLab 可选指南](../setup/jupyterlab.md)。

## 固定学习顺序

每课都按：**讲义 → 补充阅读 → 互动图 → 实验二选一 → starter → 自动核查 → 交付物与验收**。
没有独立 starter 的研究课，以清单中的交付物和完成标准验收。

## 九阶段总览

| 阶段 | 课次 | 主题 |
|---:|---:|---|
| 1 | 1-4 | 工具、数学与 PyTorch |
| 2 | 5-8 | 神经网络与早期语言模型 |
| 3 | 9-15 | 从分词到 GPT |
| 4 | 16-21 | 现代 Decoder 组件 |
| 5 | 22-28 | 预训练、数据、训练系统与评测 |
| 6 | 29-34 | 注意力与序列建模前沿 |
| 7 | 35-39 | Mixture of Experts |
| 8 | 40-44 | 后训练与推理能力 |
| 9 | 45-48 | 推理优化与毕业项目 |

## 五个贯穿式大作业

单函数 starter 用于练习局部正确性；贯穿式大作业用于证明你能把数据、模型、训练和评测连接起来。
建设中的项目先阅读规格，不计入当前自动验收；完成后会在同一 48 课主线中启用。

| 编号 | 相关课次 | 状态 | 大作业 | 核查 |
|---:|---:|---|---|---|
| 01 | 9–21 | 可开始 | [从字节 BPE 到可恢复训练的完整语言模型](labs/projects/01_end_to_end_lm/README.md) | `uv run llm-course projects check 01` |
| 02 | 22–23、28 | 建设中 | [从网页文档到可审计预训练数据](labs/projects/02_real_data_pipeline/README.md) | 规格预览 |
| 03 | 25–26、29 | 建设中 | [Profiling、Triton、DDP 与分片训练](labs/projects/03_gpu_systems/README.md) | 规格预览 |
| 04 | 27–28 | 建设中 | [受预算约束的 Scaling Law 实验](labs/projects/04_scaling_laws/README.md) | 规格预览 |
| 05 | 40–44 | 建设中 | [SFT、rollout、可验证奖励与策略更新](labs/projects/05_alignment_rl/README.md) | 规格预览 |

## 阶段 1：工具、数学与 PyTorch（第 1-4 课）

### 第 01 课：命令行、Git、Python 与可复现环境

- **学习目标**：理解仓库、环境和依赖的区别；会运行 Python、测试和 Notebook
- **前置知识**：课程环境准备
- **预计时间**：8-10 小时，可按掌握程度拆分多天完成
- **本课内容**：运行 doctor；修改并执行一个 Python 函数
- **按此顺序学习**：
  1. 阅读 [本课完整讲义](readings/lessons/01_environment_reproducibility.md)。
  2. 按需阅读 [environment.md](../setup/environment.md)、[learning_method.md](readings/references/learning_method.md)、[01_foundations.md](readings/stages/01_foundations.md)，用于补齐阶段背景和方法。
  3. 打开 [互动图](readings/interactive/foundations-lab.html)，先预测控件变化，再观察结果。
  4. 本课无独立代码实验；运行 `uv run llm-course doctor` 并保存诊断结果。
  5. 本课没有独立 starter，完成研究记录或实验报告。
  6. 按讲义中的断言复现结果，并检查输出中没有隐藏状态。
  7. **交付物**：环境诊断快照与复现命令。
- **完成标准**：doctor 通过；能说明工作区、解释器、虚拟环境和锁文件
- **一手来源**：https://docs.python.org/3/library/venv.html

### 第 02 课：向量、矩阵与张量形状

- **学习目标**：用几何和表格理解向量/矩阵；熟练追踪 batch、time、feature 维
- **前置知识**：完成第 01 课
- **预计时间**：8-10 小时，可按掌握程度拆分多天完成
- **本课内容**：NumPy 广播；矩阵乘法可视化
- **按此顺序学习**：
  1. 阅读 [本课完整讲义](readings/lessons/02_tensor_shapes.md)。
  2. 按需阅读 [01_foundations.md](readings/stages/01_foundations.md)，用于补齐阶段背景和方法。
  3. 打开 [互动图](readings/interactive/foundations-lab.html)，先预测控件变化，再观察结果。
  4. 实验二选一：[.ipynb](labs/01_shapes_and_autograd.ipynb) 或 [VS Code # %% .py](labs/01_shapes_and_autograd.py)。
  5. 本课没有独立 starter，完成研究记录或实验报告。
  6. 按讲义中的断言复现结果，并检查输出中没有隐藏状态。
  7. **交付物**：三段张量代码的逐步形状表。
- **完成标准**：解释点积和矩阵乘法；通过形状练习
- **一手来源**：https://docs.pytorch.org/docs/stable/notes/broadcasting.html

### 第 03 课：导数、链式法则与自动微分

- **学习目标**：把导数理解为局部敏感度；手算并实现反向传播
- **前置知识**：完成第 02 课
- **预计时间**：8-10 小时，可按掌握程度拆分多天完成
- **本课内容**：有限差分；标量自动微分
- **按此顺序学习**：
  1. 阅读 [本课完整讲义](readings/lessons/03_chain_rule_autograd.md)。
  2. 按需阅读 [01_foundations.md](readings/stages/01_foundations.md)，用于补齐阶段背景和方法。
  3. 打开 [互动图](readings/interactive/foundations-lab.html)，先预测控件变化，再观察结果。
  4. 实验二选一：[.ipynb](labs/01_shapes_and_autograd.ipynb) 或 [VS Code # %% .py](labs/01_shapes_and_autograd.py)。
  5. 填写 [starter 11](labs/starter/11_autograd.py) 中保留的核心空缺。
  6. 自动核查：`uv run llm-course exercises check 11`。
  7. **交付物**：分支计算图、有限差分和 starter 核查。
- **完成标准**：数值梯度与解析梯度一致；画出一个计算图
- **一手来源**：https://github.com/karpathy/micrograd

### 第 04 课：概率、Softmax、交叉熵与 PyTorch

- **学习目标**：理解概率分布和负对数似然；会写基础 PyTorch 训练循环
- **前置知识**：完成第 03 课
- **预计时间**：8-10 小时，可按掌握程度拆分多天完成
- **本课内容**：稳定 Softmax；交叉熵曲线
- **按此顺序学习**：
  1. 阅读 [本课完整讲义](readings/lessons/04_softmax_cross_entropy.md)。
  2. 按需阅读 [01_foundations.md](readings/stages/01_foundations.md)，用于补齐阶段背景和方法。
  3. 打开 [互动图](readings/interactive/foundations-lab.html)，先预测控件变化，再观察结果。
  4. 实验二选一：[.ipynb](labs/01_shapes_and_autograd.ipynb) 或 [VS Code # %% .py](labs/01_shapes_and_autograd.py)。
  5. 填写 [starter 01](labs/starter/01_stable_softmax.py) 中保留的核心空缺。
  6. 自动核查：`uv run llm-course exercises check 01`。
  7. **交付物**：稳定 Softmax 与训练循环记录。
- **完成标准**：实现稳定 Softmax；完成一个线性分类器训练
- **一手来源**：bengio_nplm


## 阶段 2：神经网络与早期语言模型（第 5-8 课）

### 第 05 课：监督学习、泛化与数据切分

- **学习目标**：区分训练、验证和测试；理解欠拟合与过拟合
- **前置知识**：完成第 04 课
- **预计时间**：8-10 小时，可按掌握程度拆分多天完成
- **本课内容**：拟合一维函数；观察训练/验证曲线
- **按此顺序学习**：
  1. 阅读 [本课完整讲义](readings/lessons/05_supervision_generalization.md)。
  2. 按需阅读 [02_neural_lm.md](readings/stages/02_neural_lm.md)，用于补齐阶段背景和方法。
  3. 打开 [互动图](readings/interactive/foundations-lab.html)，先预测控件变化，再观察结果。
  4. 实验二选一：[.ipynb](labs/02_neural_language_models.ipynb) 或 [VS Code # %% .py](labs/02_neural_language_models.py)。
  5. 本课没有独立 starter，完成研究记录或实验报告。
  6. 按讲义中的断言复现结果，并检查输出中没有隐藏状态。
  7. **交付物**：无泄漏数据切分说明。
- **完成标准**：不会用测试集调参；能解释泛化间隙
- **一手来源**：the_pile

### 第 06 课：MLP、激活函数与优化

- **学习目标**：理解层的组合；观察 ReLU、tanh、SiLU 的差异
- **前置知识**：完成第 05 课
- **预计时间**：8-10 小时，可按掌握程度拆分多天完成
- **本课内容**：从零 MLP；梯度流可视化
- **按此顺序学习**：
  1. 阅读 [本课完整讲义](readings/lessons/06_mlp_activations_optimization.md)。
  2. 按需阅读 [02_neural_lm.md](readings/stages/02_neural_lm.md)，用于补齐阶段背景和方法。
  3. 打开 [互动图](readings/interactive/foundations-lab.html)，先预测控件变化，再观察结果。
  4. 实验二选一：[.ipynb](labs/02_neural_language_models.ipynb) 或 [VS Code # %% .py](labs/02_neural_language_models.py)。
  5. 填写 [starter 12](labs/starter/12_neural_lm.py) 中保留的核心空缺。
  6. 自动核查：`uv run llm-course exercises check 12`。
  7. **交付物**：MLP 小数据过拟合与梯度图。
- **完成标准**：小数据过拟合；解释非线性的作用
- **一手来源**：bengio_nplm

### 第 07 课：词向量与神经语言模型

- **学习目标**：把 token 映射为可学习向量；理解 next-token prediction
- **前置知识**：完成第 06 课
- **预计时间**：8-10 小时，可按掌握程度拆分多天完成
- **本课内容**：Bigram 表；Embedding + MLP LM
- **按此顺序学习**：
  1. 阅读 [本课完整讲义](readings/lessons/07_embeddings_neural_lm.md)。
  2. 按需阅读 [02_neural_lm.md](readings/stages/02_neural_lm.md)，用于补齐阶段背景和方法。
  3. 打开 [互动图](readings/interactive/foundations-lab.html)，先预测控件变化，再观察结果。
  4. 实验二选一：[.ipynb](labs/02_neural_language_models.ipynb) 或 [VS Code # %% .py](labs/02_neural_language_models.py)。
  5. 填写 [starter 12](labs/starter/12_neural_lm.py) 中保留的核心空缺。
  6. 自动核查：`uv run llm-course exercises check 12`。
  7. **交付物**：Bigram 与固定窗口 MLP 对照。
- **完成标准**：实现 bigram LM；能解释 logits 和概率
- **一手来源**：bengio_nplm

### 第 08 课：RNN、状态与序列瓶颈

- **学习目标**：理解循环状态；知道长依赖和串行计算为何困难
- **前置知识**：完成第 07 课
- **预计时间**：8-10 小时，可按掌握程度拆分多天完成
- **本课内容**：手写 Elman RNN；梯度随时间变化
- **按此顺序学习**：
  1. 阅读 [本课完整讲义](readings/lessons/08_rnn_state_and_sequence_bottleneck.md)。
  2. 按需阅读 [02_neural_lm.md](readings/stages/02_neural_lm.md)，用于补齐阶段背景和方法。
  3. 打开 [互动图](readings/interactive/foundations-lab.html)，先预测控件变化，再观察结果。
  4. 实验二选一：[.ipynb](labs/02_neural_language_models.ipynb) 或 [VS Code # %% .py](labs/02_neural_language_models.py)。
  5. 填写 [starter 12](labs/starter/12_neural_lm.py) 中保留的核心空缺。
  6. 自动核查：`uv run llm-course exercises check 12`。
  7. **交付物**：RNN 状态、BPTT 和长依赖反例。
- **完成标准**：实现 RNN 单元；说出注意力替代 RNN 的动机
- **一手来源**：seq2seq


## 阶段 3：从分词到 GPT（第 9-15 课）

### 第 09 课：文本、Unicode、字节与 token

- **学习目标**：理解字符不等于字节也不等于 token；能构造确定的词表
- **前置知识**：完成第 08 课
- **预计时间**：8-10 小时，可按掌握程度拆分多天完成
- **本课内容**：UTF-8 拆解；字符级 tokenizer
- **按此顺序学习**：
  1. 阅读 [本课完整讲义](readings/lessons/09_text_unicode_bytes_tokens.md)。
  2. 按需阅读 [03_transformer.md](readings/stages/03_transformer.md)，用于补齐阶段背景和方法。
  3. 打开 [互动图](readings/interactive/core-concepts.html)，先预测控件变化，再观察结果。
  4. 实验二选一：[.ipynb](labs/03_tokenization_and_bpe.ipynb) 或 [VS Code # %% .py](labs/03_tokenization_and_bpe.py)。
  5. 本课没有独立 starter，完成研究记录或实验报告。
  6. 按讲义中的断言复现结果，并检查输出中没有隐藏状态。
  7. **交付物**：中英文字符、字节和 token 对照表。
- **完成标准**：encode/decode 往返一致；解释未知 token 问题
- **一手来源**：gpt2

### 第 10 课：从零实现 BPE

- **学习目标**：理解频繁相邻对合并；处理确定性和特殊 token
- **前置知识**：完成第 09 课
- **预计时间**：8-10 小时，可按掌握程度拆分多天完成
- **本课内容**：训练 Byte-level BPE；查看合并过程
- **按此顺序学习**：
  1. 阅读 [本课完整讲义](readings/lessons/10_byte_bpe_from_scratch.md)。
  2. 按需阅读 [03_transformer.md](readings/stages/03_transformer.md)，用于补齐阶段背景和方法。
  3. 打开 [互动图](readings/interactive/core-concepts.html)，先预测控件变化，再观察结果。
  4. 实验二选一：[.ipynb](labs/03_tokenization_and_bpe.ipynb) 或 [VS Code # %% .py](labs/03_tokenization_and_bpe.py)。
  5. 填写 [starter 06](labs/starter/06_byte_bpe.py) 中保留的核心空缺。
  6. 自动核查：`uv run llm-course exercises check 06`。
  7. **交付物**：确定性 Byte BPE 往返核查。
- **完成标准**：BPE 往返一致；合并顺序可复现
- **一手来源**：gpt2

### 第 11 课：Embedding 与位置信息

- **学习目标**：理解 token/position embedding；知道没有位置时注意力看不见顺序
- **前置知识**：完成第 10 课
- **预计时间**：8-10 小时，可按掌握程度拆分多天完成
- **本课内容**：打乱 token 测试；绝对位置 embedding
- **按此顺序学习**：
  1. 阅读 [本课完整讲义](readings/lessons/11_embeddings_and_position.md)。
  2. 按需阅读 [03_transformer.md](readings/stages/03_transformer.md)，用于补齐阶段背景和方法。
  3. 打开 [互动图](readings/interactive/core-concepts.html)，先预测控件变化，再观察结果。
  4. 实验二选一：[.ipynb](labs/04_attention_mechanics.ipynb) 或 [VS Code # %% .py](labs/04_attention_mechanics.py)。
  5. 本课没有独立 starter，完成研究记录或实验报告。
  6. 按讲义中的断言复现结果，并检查输出中没有隐藏状态。
  7. **交付物**：无位置编码时的顺序反例。
- **完成标准**：解释 permutation equivariance；实现位置向量相加
- **一手来源**：transformer

### 第 12 课：缩放点积注意力

- **学习目标**：从查询、钥匙、内容理解 QKV；推导缩放与 Softmax
- **前置知识**：完成第 11 课
- **预计时间**：8-10 小时，可按掌握程度拆分多天完成
- **本课内容**：单头注意力热力图；有无缩放的熵
- **按此顺序学习**：
  1. 阅读 [本课完整讲义](readings/lessons/12_scaled_dot_product_attention.md)。
  2. 按需阅读 [03_transformer.md](readings/stages/03_transformer.md)，用于补齐阶段背景和方法。
  3. 打开 [互动图](readings/interactive/core-concepts.html)，先预测控件变化，再观察结果。
  4. 实验二选一：[.ipynb](labs/04_attention_mechanics.ipynb) 或 [VS Code # %% .py](labs/04_attention_mechanics.py)。
  5. 填写 [starter 02](labs/starter/02_causal_attention.py) 中保留的核心空缺。
  6. 自动核查：`uv run llm-course exercises check 02`。
  7. **交付物**：可编辑 Q/K/V 与缩放注意力。
- **完成标准**：形状正确；与 PyTorch SDPA 数值一致
- **一手来源**：transformer

### 第 13 课：因果掩码与多头注意力

- **学习目标**：阻止未来信息泄漏；理解多头的独立投影与拼接
- **前置知识**：完成第 12 课
- **预计时间**：8-10 小时，可按掌握程度拆分多天完成
- **本课内容**：未来 token 扰动测试；多头注意力
- **按此顺序学习**：
  1. 阅读 [本课完整讲义](readings/lessons/13_causal_mask_and_multihead_attention.md)。
  2. 按需阅读 [03_transformer.md](readings/stages/03_transformer.md)，用于补齐阶段背景和方法。
  3. 打开 [互动图](readings/interactive/core-concepts.html)，先预测控件变化，再观察结果。
  4. 实验二选一：[.ipynb](labs/04_attention_mechanics.ipynb) 或 [VS Code # %% .py](labs/04_attention_mechanics.py)。
  5. 填写 [starter 02](labs/starter/02_causal_attention.py) 中保留的核心空缺。
  6. 自动核查：`uv run llm-course exercises check 02`。
  7. **交付物**：未来扰动与全遮蔽行测试。
- **完成标准**：未来 token 不影响过去输出；梯度通过所有投影
- **一手来源**：transformer

### 第 14 课：Transformer Block 与残差

- **学习目标**：组装 Attention、MLP、Norm、Residual；理解残差的信息高速路
- **前置知识**：完成第 13 课
- **预计时间**：8-10 小时，可按掌握程度拆分多天完成
- **本课内容**：无残差消融；梯度范数记录
- **按此顺序学习**：
  1. 阅读 [本课完整讲义](readings/lessons/14_transformer_block_and_residual.md)。
  2. 按需阅读 [03_transformer.md](readings/stages/03_transformer.md)，用于补齐阶段背景和方法。
  3. 打开 [互动图](readings/interactive/core-concepts.html)，先预测控件变化，再观察结果。
  4. 实验二选一：[.ipynb](labs/05_tiny_gpt.ipynb) 或 [VS Code # %% .py](labs/05_tiny_gpt.py)。
  5. 填写 [starter 13](labs/starter/13_tiny_gpt.py) 中保留的核心空缺。
  6. 自动核查：`uv run llm-course exercises check 13`。
  7. **交付物**：Pre-Norm Block 与梯度范数。
- **完成标准**：端到端前反传；参数量计算正确
- **一手来源**：transformer

### 第 15 课：训练并采样 Tiny GPT

- **学习目标**：完成数据到生成的闭环；理解温度和 top-k 采样
- **前置知识**：完成第 14 课
- **预计时间**：8-10 小时，可按掌握程度拆分多天完成
- **本课内容**：单 batch 过拟合；短语料训练
- **按此顺序学习**：
  1. 阅读 [本课完整讲义](readings/lessons/15_train_and_sample_tiny_gpt.md)。
  2. 按需阅读 [03_transformer.md](readings/stages/03_transformer.md)，用于补齐阶段背景和方法。
  3. 打开 [互动图](readings/interactive/core-concepts.html)，先预测控件变化，再观察结果。
  4. 实验二选一：[.ipynb](labs/05_tiny_gpt.ipynb) 或 [VS Code # %% .py](labs/05_tiny_gpt.py)。
  5. 填写 [starter 13](labs/starter/13_tiny_gpt.py) 中保留的核心空缺。
  6. 自动核查：`uv run llm-course exercises check 13`。
  7. **交付物**：TinyGPT 保存、加载和生成样例。
- **完成标准**：损失显著下降；可保存/加载并生成文本
- **一手来源**：gpt2


## 阶段 4：现代 Decoder 组件（第 16-21 课）

### 第 16 课：Pre-Norm 与 RMSNorm

- **学习目标**：理解归一化位置对梯度的影响；从零实现 RMSNorm
- **前置知识**：完成第 15 课
- **预计时间**：8-10 小时，可按掌握程度拆分多天完成
- **本课内容**：LayerNorm/RMSNorm 对照；深层梯度
- **按此顺序学习**：
  1. 阅读 [本课完整讲义](readings/lessons/16_rmsnorm_prenorm.md)。
  2. 按需阅读 [04_modern_decoder.md](readings/stages/04_modern_decoder.md)，用于补齐阶段背景和方法。
  3. 打开 [互动图](readings/interactive/architecture-lab.html)，先预测控件变化，再观察结果。
  4. 实验二选一：[.ipynb](labs/06_modern_decoder.ipynb) 或 [VS Code # %% .py](labs/06_modern_decoder.py)。
  5. 填写 [starter 09](labs/starter/09_modern_decoder.py) 中保留的核心空缺。
  6. 自动核查：`uv run llm-course exercises check 09`。
  7. **交付物**：经典与 RMSNorm 消融。
- **完成标准**：与参考公式一致；解释 Pre-Norm 的训练优势
- **一手来源**：rmsnorm

### 第 17 课：SwiGLU 与门控 MLP

- **学习目标**：把门控理解为内容过滤；匹配参数预算
- **前置知识**：完成第 16 课
- **预计时间**：8-10 小时，可按掌握程度拆分多天完成
- **本课内容**：GELU/SwiGLU 对照；激活分布
- **按此顺序学习**：
  1. 阅读 [本课完整讲义](readings/lessons/17_swiglu.md)。
  2. 按需阅读 [04_modern_decoder.md](readings/stages/04_modern_decoder.md)，用于补齐阶段背景和方法。
  3. 打开 [互动图](readings/interactive/architecture-lab.html)，先预测控件变化，再观察结果。
  4. 实验二选一：[.ipynb](labs/06_modern_decoder.ipynb) 或 [VS Code # %% .py](labs/06_modern_decoder.py)。
  5. 填写 [starter 09](labs/starter/09_modern_decoder.py) 中保留的核心空缺。
  6. 自动核查：`uv run llm-course exercises check 09`。
  7. **交付物**：等参数预算 MLP/SwiGLU 对照。
- **完成标准**：实现 SwiGLU；说明隐藏维为何常改变
- **一手来源**：swiglu

### 第 18 课：RoPE：用旋转编码相对位置

- **学习目标**：从二维旋转理解 RoPE；正确处理频率和维度
- **前置知识**：完成第 17 课
- **预计时间**：8-10 小时，可按掌握程度拆分多天完成
- **本课内容**：旋转向量动画；相对位移点积
- **按此顺序学习**：
  1. 阅读 [本课完整讲义](readings/lessons/18_rope_and_extensions.md)。
  2. 按需阅读 [04_modern_decoder.md](readings/stages/04_modern_decoder.md)，用于补齐阶段背景和方法。
  3. 打开 [互动图](readings/interactive/architecture-lab.html)，先预测控件变化，再观察结果。
  4. 实验二选一：[.ipynb](labs/06_modern_decoder.ipynb) 或 [VS Code # %% .py](labs/06_modern_decoder.py)。
  5. 填写 [starter 07](labs/starter/07_rope.py) 中保留的核心空缺。
  6. 自动核查：`uv run llm-course exercises check 07`。
  7. **交付物**：RoPE 范数与相对位移核查。
- **完成标准**：保持向量范数；相对位置测试通过
- **一手来源**：rope

### 第 19 课：MQA 与 GQA

- **学习目标**：理解共享 KV 如何节省缓存；追踪 query head 到 KV head 映射
- **前置知识**：完成第 18 课
- **预计时间**：8-10 小时，可按掌握程度拆分多天完成
- **本课内容**：MHA/GQA 缓存估算；重复 KV 数值对照
- **按此顺序学习**：
  1. 阅读 [本课完整讲义](readings/lessons/19_mqa_gqa.md)。
  2. 按需阅读 [04_modern_decoder.md](readings/stages/04_modern_decoder.md)，用于补齐阶段背景和方法。
  3. 打开 [互动图](readings/interactive/architecture-lab.html)，先预测控件变化，再观察结果。
  4. 实验二选一：[.ipynb](labs/06_modern_decoder.ipynb) 或 [VS Code # %% .py](labs/06_modern_decoder.py)。
  5. 填写 [starter 08](labs/starter/08_grouped_query_attention.py) 中保留的核心空缺。
  6. 自动核查：`uv run llm-course exercises check 08`。
  7. **交付物**：MHA/MQA/GQA 数值与缓存对照。
- **完成标准**：数值一致；能计算缓存节省比例
- **一手来源**：gqa

### 第 20 课：KV Cache 与逐 token 解码

- **学习目标**：区分 prefill 和 decode；避免重复计算历史 K/V
- **前置知识**：完成第 19 课
- **预计时间**：8-10 小时，可按掌握程度拆分多天完成
- **本课内容**：完整前缀/缓存解码对照；缓存增长曲线
- **按此顺序学习**：
  1. 阅读 [本课完整讲义](readings/lessons/20_kv_cache_decode.md)。
  2. 按需阅读 [04_modern_decoder.md](readings/stages/04_modern_decoder.md)，用于补齐阶段背景和方法。
  3. 打开 [互动图](readings/interactive/architecture-lab.html)，先预测控件变化，再观察结果。
  4. 实验二选一：[.ipynb](labs/06_modern_decoder.ipynb) 或 [VS Code # %% .py](labs/06_modern_decoder.py)。
  5. 填写 [starter 03](labs/starter/03_kv_cache_budget.py) 中保留的核心空缺。
  6. 自动核查：`uv run llm-course exercises check 03`。
  7. **交付物**：prefill/decode cache 等价报告。
- **完成标准**：缓存与非缓存 logits 一致；说明速度与内存权衡
- **一手来源**：gqa

### 第 21 课：初始化、AdamW 与训练稳定性

- **学习目标**：理解方差传播；区分权重衰减和 L2
- **前置知识**：完成第 20 课
- **预计时间**：8-10 小时，可按掌握程度拆分多天完成
- **本课内容**：不同初始化的激活；梯度裁剪
- **按此顺序学习**：
  1. 阅读 [本课完整讲义](readings/lessons/21_initialization_stability.md)。
  2. 按需阅读 [04_modern_decoder.md](readings/stages/04_modern_decoder.md)，用于补齐阶段背景和方法。
  3. 打开 [互动图](readings/interactive/architecture-lab.html)，先预测控件变化，再观察结果。
  4. 实验二选一：[.ipynb](labs/06_modern_decoder.ipynb) 或 [VS Code # %% .py](labs/06_modern_decoder.py)。
  5. 本课没有独立 starter，完成研究记录或实验报告。
  6. 按讲义中的断言复现结果，并检查输出中没有隐藏状态。
  7. **交付物**：经典到现代组件逐项消融表。
- **完成标准**：训练无 NaN；能解释 warmup 和 clipping
- **一手来源**：llama


## 阶段 5：预训练、数据、训练系统与评测（第 22-28 课）

### 第 22 课：预训练数据从哪里来

- **学习目标**：理解数据来源、许可和泄漏；设计数据卡
- **前置知识**：完成第 21 课
- **预计时间**：8-10 小时，可按掌握程度拆分多天完成
- **本课内容**：检查小语料编码；数据统计
- **按此顺序学习**：
  1. 阅读 [本课完整讲义](readings/lessons/22_data_governance.md)。
  2. 按需阅读 [05_pretraining.md](readings/stages/05_pretraining.md)，用于补齐阶段背景和方法。
  3. 打开 [互动图](readings/interactive/training-and-alignment.html)，先预测控件变化，再观察结果。
  4. 实验二选一：[.ipynb](labs/07_pretraining_systems.ipynb) 或 [VS Code # %% .py](labs/07_pretraining_systems.py)。
  5. 填写 [starter 14](labs/starter/14_data_pipeline.py) 中保留的核心空缺。
  6. 自动核查：`uv run llm-course exercises check 14`。
  7. **交付物**：数据卡、许可与来源追踪。
- **完成标准**：记录来源与许可；区分训练和评测污染
- **一手来源**：the_pile

### 第 23 课：过滤、去重与数据混合

- **学习目标**：理解质量过滤和近重复；观察数据配比影响
- **前置知识**：完成第 22 课
- **预计时间**：8-10 小时，可按掌握程度拆分多天完成
- **本课内容**：MinHash 概念实验；重复数据过拟合
- **按此顺序学习**：
  1. 阅读 [本课完整讲义](readings/lessons/23_dedup_mixing_packing.md)。
  2. 按需阅读 [05_pretraining.md](readings/stages/05_pretraining.md)，用于补齐阶段背景和方法。
  3. 打开 [互动图](readings/interactive/training-and-alignment.html)，先预测控件变化，再观察结果。
  4. 实验二选一：[.ipynb](labs/07_pretraining_systems.ipynb) 或 [VS Code # %% .py](labs/07_pretraining_systems.py)。
  5. 填写 [starter 14](labs/starter/14_data_pipeline.py) 中保留的核心空缺。
  6. 自动核查：`uv run llm-course exercises check 14`。
  7. **交付物**：过滤、去重、混合和 packing 统计。
- **完成标准**：统计每步保留率；避免静默丢数据
- **一手来源**：dedup_training_data

### 第 24 课：优化器、Warmup 与学习率计划

- **学习目标**：理解动量和自适应缩放；设计可解释的 scheduler
- **前置知识**：完成第 23 课
- **预计时间**：8-10 小时，可按掌握程度拆分多天完成
- **本课内容**：SGD/AdamW 对照；cosine schedule
- **按此顺序学习**：
  1. 阅读 [本课完整讲义](readings/lessons/24_adamw_schedules.md)。
  2. 按需阅读 [05_pretraining.md](readings/stages/05_pretraining.md)，用于补齐阶段背景和方法。
  3. 打开 [互动图](readings/interactive/training-and-alignment.html)，先预测控件变化，再观察结果。
  4. 实验二选一：[.ipynb](labs/07_pretraining_systems.ipynb) 或 [VS Code # %% .py](labs/07_pretraining_systems.py)。
  5. 填写 [starter 15](labs/starter/15_adamw_schedule.py) 中保留的核心空缺。
  6. 自动核查：`uv run llm-course exercises check 15`。
  7. **交付物**：AdamW 单步 oracle 与学习率曲线。
- **完成标准**：与 PyTorch 单步一致；绘制学习率曲线
- **一手来源**：adamw

### 第 25 课：FLOPs、显存与混合精度

- **学习目标**：估算参数/激活/优化器内存；理解算力与带宽瓶颈
- **前置知识**：完成第 24 课
- **预计时间**：8-10 小时，可按掌握程度拆分多天完成
- **本课内容**：resource accounting；dtype 误差
- **按此顺序学习**：
  1. 阅读 [本课完整讲义](readings/lessons/25_mixed_precision_resources.md)。
  2. 按需阅读 [05_pretraining.md](readings/stages/05_pretraining.md)，用于补齐阶段背景和方法。
  3. 打开 [互动图](readings/interactive/training-and-alignment.html)，先预测控件变化，再观察结果。
  4. 实验二选一：[.ipynb](labs/07_pretraining_systems.ipynb) 或 [VS Code # %% .py](labs/07_pretraining_systems.py)。
  5. 本课没有独立 starter，完成研究记录或实验报告。
  6. 按讲义中的断言复现结果，并检查输出中没有隐藏状态。
  7. **交付物**：参数、激活和优化器内存预算。
- **完成标准**：写清估算假设；区分训练与推理内存
- **一手来源**：llama

### 第 26 课：并行训练、检查点与故障恢复

- **学习目标**：区分数据/张量/流水线并行；理解 ZeRO/FSDP 的状态分片
- **前置知识**：完成第 25 课
- **预计时间**：8-10 小时，可按掌握程度拆分多天完成
- **本课内容**：纸面内存分片；可恢复检查点清单
- **按此顺序学习**：
  1. 阅读 [本课完整讲义](readings/lessons/26_parallel_checkpointing.md)。
  2. 按需阅读 [05_pretraining.md](readings/stages/05_pretraining.md)，用于补齐阶段背景和方法。
  3. 打开 [互动图](readings/interactive/training-and-alignment.html)，先预测控件变化，再观察结果。
  4. 实验二选一：[.ipynb](labs/07_pretraining_systems.ipynb) 或 [VS Code # %% .py](labs/07_pretraining_systems.py)。
  5. 本课没有独立 starter，完成研究记录或实验报告。
  6. 按讲义中的断言复现结果，并检查输出中没有隐藏状态。
  7. **交付物**：并行拓扑与可恢复检查点方案。
- **完成标准**：标明通信边界；恢复后数据顺序和优化器状态可追溯
- **一手来源**：gshard

### 第 27 课：Scaling Laws 与计算最优

- **学习目标**：理解幂律拟合；不外推到证据范围之外
- **前置知识**：完成第 26 课
- **预计时间**：8-10 小时，可按掌握程度拆分多天完成
- **本课内容**：拟合微型 scaling curve；置信区间
- **按此顺序学习**：
  1. 阅读 [本课完整讲义](readings/lessons/27_scaling_and_evaluation.md)。
  2. 按需阅读 [05_pretraining.md](readings/stages/05_pretraining.md)，用于补齐阶段背景和方法。
  3. 打开 [互动图](readings/interactive/training-and-alignment.html)，先预测控件变化，再观察结果。
  4. 实验二选一：[.ipynb](labs/07_pretraining_systems.ipynb) 或 [VS Code # %% .py](labs/07_pretraining_systems.py)。
  5. 本课没有独立 starter，完成研究记录或实验报告。
  6. 按讲义中的断言复现结果，并检查输出中没有隐藏状态。
  7. **交付物**：Scaling Law 拟合与残差。
- **完成标准**：报告拟合残差；说明缩尺结论局限
- **一手来源**：chinchilla

### 第 28 课：可靠评测、去污染与故障诊断

- **学习目标**：建立版本固定且可重复的评测；区分能力、污染、实现故障和统计波动
- **前置知识**：完成第 27 课
- **预计时间**：8-10 小时，可按掌握程度拆分多天完成
- **本课内容**：随机与旧 checkpoint 基线；答案 parser、近重复污染与分桶评测
- **按此顺序学习**：
  1. 阅读 [本课完整讲义](readings/lessons/28_reliable_evaluation.md)。
  2. 按需阅读 [05_pretraining.md](readings/stages/05_pretraining.md)，用于补齐阶段背景和方法。
  3. 打开 [互动图](readings/interactive/training-and-alignment.html)，先预测控件变化，再观察结果。
  4. 实验二选一：[.ipynb](labs/07_pretraining_systems.ipynb) 或 [VS Code # %% .py](labs/07_pretraining_systems.py)。
  5. 本课没有独立 starter，完成研究记录或实验报告。
  6. 按讲义中的断言复现结果，并检查输出中没有隐藏状态。
  7. **交付物**：含去污染声明、区间与故障核查的评测卡。
- **完成标准**：原始输出、样本分母、种子与环境完整；污染命中保存证据且报告处理前后结果
- **一手来源**：helm


## 阶段 6：注意力与序列建模前沿（第 29-34 课）

### 第 29 课：FlashAttention 与 IO-aware 思维

- **学习目标**：区分数学等价和执行顺序；理解分块减少 HBM 往返
- **前置知识**：完成第 28 课
- **预计时间**：8-10 小时，可按掌握程度拆分多天完成
- **本课内容**：分块在线 Softmax；IO 计数模型
- **按此顺序学习**：
  1. 阅读 [本课完整讲义](readings/lessons/29_flashattention.md)。
  2. 按需阅读 [06_attention_frontiers.md](readings/stages/06_attention_frontiers.md)，用于补齐阶段背景和方法。
  3. 打开 [互动图](readings/interactive/architecture-lab.html)，先预测控件变化，再观察结果。
  4. 实验二选一：[.ipynb](labs/08_attention_frontiers.ipynb) 或 [VS Code # %% .py](labs/08_attention_frontiers.py)。
  5. 填写 [starter 16](labs/starter/16_attention_frontiers.py) 中保留的核心空缺。
  6. 自动核查：`uv run llm-course exercises check 16`。
  7. **交付物**：在线 Softmax 与标准注意力等价。
- **完成标准**：与标准注意力一致；解释为何不是近似注意力
- **一手来源**：flashattention

### 第 30 课：滑窗、块稀疏与全局 token

- **学习目标**：理解稀疏模式；分析感受野随层数增长
- **前置知识**：完成第 29 课
- **预计时间**：8-10 小时，可按掌握程度拆分多天完成
- **本课内容**：稀疏 mask 可视化；长距离复制任务
- **按此顺序学习**：
  1. 阅读 [本课完整讲义](readings/lessons/30_sparse_attention.md)。
  2. 按需阅读 [06_attention_frontiers.md](readings/stages/06_attention_frontiers.md)，用于补齐阶段背景和方法。
  3. 打开 [互动图](readings/interactive/architecture-lab.html)，先预测控件变化，再观察结果。
  4. 实验二选一：[.ipynb](labs/08_attention_frontiers.ipynb) 或 [VS Code # %% .py](labs/08_attention_frontiers.py)。
  5. 填写 [starter 16](labs/starter/16_attention_frontiers.py) 中保留的核心空缺。
  6. 自动核查：`uv run llm-course exercises check 16`。
  7. **交付物**：滑窗/稀疏感受野反例。
- **完成标准**：mask 正确；给出无法连接的反例
- **一手来源**：longformer

### 第 31 课：MLA 与低维 KV 表示

- **学习目标**：理解 KV 压缩、历史重建与投影吸收；认识 decoupled RoPE
- **前置知识**：完成第 30 课
- **预计时间**：8-10 小时，可按掌握程度拆分多天完成
- **本课内容**：MHA/GQA/MLA 缓存估算；latent-cache reconstruction baseline
- **按此顺序学习**：
  1. 阅读 [本课完整讲义](readings/lessons/31_mla.md)。
  2. 按需阅读 [06_attention_frontiers.md](readings/stages/06_attention_frontiers.md)，用于补齐阶段背景和方法。
  3. 打开 [互动图](readings/interactive/architecture-lab.html)，先预测控件变化，再观察结果。
  4. 实验二选一：[.ipynb](labs/08_attention_frontiers.ipynb) 或 [VS Code # %% .py](labs/08_attention_frontiers.py)。
  5. 填写 [starter 17](labs/starter/17_mla_delta.py) 中保留的核心空缺。
  6. 自动核查：`uv run llm-course exercises check 17`。
  7. **交付物**：MLA baseline 成本与边界说明。
- **完成标准**：输出/缓存形状正确；不把历史重建误称为 absorbed decode
- **一手来源**：deepseek_v2

### 第 32 课：线性注意力的结合律与归一化状态

- **学习目标**：理解正特征映射和归一化分母；连接并行训练与递归解码
- **前置知识**：完成第 31 课
- **预计时间**：8-10 小时，可按掌握程度拆分多天完成
- **本课内容**：二次/线性实现对照；parallel/recurrent 数值等价
- **按此顺序学习**：
  1. 阅读 [本课完整讲义](readings/lessons/32_linear_attention.md)。
  2. 按需阅读 [06_attention_frontiers.md](readings/stages/06_attention_frontiers.md)，用于补齐阶段背景和方法。
  3. 打开 [互动图](readings/interactive/architecture-lab.html)，先预测控件变化，再观察结果。
  4. 实验二选一：[.ipynb](labs/08_attention_frontiers.ipynb) 或 [VS Code # %% .py](labs/08_attention_frontiers.py)。
  5. 填写 [starter 16](labs/starter/16_attention_frontiers.py) 中保留的核心空缺。
  6. 自动核查：`uv run llm-course exercises check 16`。
  7. **交付物**：线性注意力并行/递归等价。
- **完成标准**：并行与递归形式一致；明确它不等于 Softmax 注意力
- **一手来源**：performer

### 第 33 课：Mamba-2、DeltaNet 与选择性状态更新

- **学习目标**：理解 SSD 对偶和 delta rule；观察选择性遗忘与定向擦除
- **前置知识**：完成第 32 课
- **预计时间**：8-10 小时，可按掌握程度拆分多天完成
- **本课内容**：选择性衰减；冲突键值写入；并行/递归路径
- **按此顺序学习**：
  1. 阅读 [本课完整讲义](readings/lessons/33_gated_deltanet.md)。
  2. 按需阅读 [06_attention_frontiers.md](readings/stages/06_attention_frontiers.md)，用于补齐阶段背景和方法。
  3. 打开 [互动图](readings/interactive/architecture-lab.html)，先预测控件变化，再观察结果。
  4. 实验二选一：[.ipynb](labs/08_attention_frontiers.ipynb) 或 [VS Code # %% .py](labs/08_attention_frontiers.py)。
  5. 填写 [starter 17](labs/starter/17_mla_delta.py) 中保留的核心空缺。
  6. 自动核查：`uv run llm-course exercises check 17`。
  7. **交付物**：Delta 状态冲突写入实验。
- **完成标准**：状态更新数值测试通过；说明 decay、beta 与注意力的差异
- **一手来源**：gated_deltanet

### 第 34 课：混合架构、DSA 与长上下文评测

- **学习目标**：理解 full/linear 混合；区分稀疏检索与固定窗口
- **前置知识**：完成第 33 课
- **预计时间**：8-10 小时，可按掌握程度拆分多天完成
- **本课内容**：混合层消融；needle/复制任务
- **按此顺序学习**：
  1. 阅读 [本课完整讲义](readings/lessons/34_hybrid_dsa_evaluation.md)。
  2. 按需阅读 [06_attention_frontiers.md](readings/stages/06_attention_frontiers.md)，用于补齐阶段背景和方法。
  3. 打开 [互动图](readings/interactive/architecture-lab.html)，先预测控件变化，再观察结果。
  4. 实验二选一：[.ipynb](labs/08_attention_frontiers.ipynb) 或 [VS Code # %% .py](labs/08_attention_frontiers.py)。
  5. 本课没有独立 starter，完成研究记录或实验报告。
  6. 按讲义中的断言复现结果，并检查输出中没有隐藏状态。
  7. **交付物**：混合架构长上下文公平消融。
- **完成标准**：报告质量/速度双指标；不把上下文长度等同有效利用
- **一手来源**：kimi_linear


## 阶段 7：Mixture of Experts（第 35-39 课）

### 第 35 课：条件计算与 Top-k 路由

- **学习目标**：区分总参数和活跃参数；实现 token-choice routing
- **前置知识**：完成第 34 课
- **预计时间**：8-10 小时，可按掌握程度拆分多天完成
- **本课内容**：路由热力图；Dense/MoE 活跃 FLOPs
- **按此顺序学习**：
  1. 阅读 [本课完整讲义](readings/lessons/35_topk_routing.md)。
  2. 按需阅读 [07_moe.md](readings/stages/07_moe.md)，用于补齐阶段背景和方法。
  3. 打开 [互动图](readings/interactive/architecture-lab.html)，先预测控件变化，再观察结果。
  4. 实验二选一：[.ipynb](labs/09_moe.ipynb) 或 [VS Code # %% .py](labs/09_moe.py)。
  5. 填写 [starter 10](labs/starter/10_moe_router.py) 中保留的核心空缺。
  6. 自动核查：`uv run llm-course exercises check 10`。
  7. **交付物**：Top-1/Top-k 路由语义和梯度核查。
- **完成标准**：门控权重归一化；所有被选择且未丢弃的专家可反传；构造批次覆盖全部专家
- **一手来源**：sparsely_gated_moe

### 第 36 课：容量、token dropping 与负载均衡

- **学习目标**：理解系统容量约束；测量 expert utilization
- **前置知识**：完成第 35 课
- **预计时间**：8-10 小时，可按掌握程度拆分多天完成
- **本课内容**：容量溢出；均衡损失前后对照
- **按此顺序学习**：
  1. 阅读 [本课完整讲义](readings/lessons/36_capacity_and_balance.md)。
  2. 按需阅读 [07_moe.md](readings/stages/07_moe.md)，用于补齐阶段背景和方法。
  3. 打开 [互动图](readings/interactive/architecture-lab.html)，先预测控件变化，再观察结果。
  4. 实验二选一：[.ipynb](labs/09_moe.ipynb) 或 [VS Code # %% .py](labs/09_moe.py)。
  5. 填写 [starter 05](labs/starter/05_moe_capacity.py) 中保留的核心空缺。
  6. 自动核查：`uv run llm-course exercises check 05`。
  7. **交付物**：容量、dropping 与 dropless 对照。
- **完成标准**：溢出行为显式；负载统计正确
- **一手来源**：switch_transformer

### 第 37 课：Router z-loss、数值精度与稳定性

- **学习目标**：控制 router logits；理解路由专用 FP32
- **前置知识**：完成第 36 课
- **预计时间**：8-10 小时，可按掌握程度拆分多天完成
- **本课内容**：极端 logits；z-loss 消融
- **按此顺序学习**：
  1. 阅读 [本课完整讲义](readings/lessons/37_router_stability.md)。
  2. 按需阅读 [07_moe.md](readings/stages/07_moe.md)，用于补齐阶段背景和方法。
  3. 打开 [互动图](readings/interactive/architecture-lab.html)，先预测控件变化，再观察结果。
  4. 实验二选一：[.ipynb](labs/09_moe.ipynb) 或 [VS Code # %% .py](labs/09_moe.py)。
  5. 填写 [starter 18](labs/starter/18_moe_systems.py) 中保留的核心空缺。
  6. 自动核查：`uv run llm-course exercises check 18`。
  7. **交付物**：FP32 router、z-loss 和 BF16 实验。
- **完成标准**：无 NaN；能解释 z-loss 不等于负载均衡
- **一手来源**：st_moe

### 第 38 课：共享专家、细粒度专家与 upcycling

- **学习目标**：理解公共知识和专业知识拆分；从 Dense 初始化 MoE
- **前置知识**：完成第 37 课
- **预计时间**：8-10 小时，可按掌握程度拆分多天完成
- **本课内容**：复制 FFN 权重；共享专家消融
- **按此顺序学习**：
  1. 阅读 [本课完整讲义](readings/lessons/38_shared_experts_upcycling.md)。
  2. 按需阅读 [07_moe.md](readings/stages/07_moe.md)，用于补齐阶段背景和方法。
  3. 打开 [互动图](readings/interactive/architecture-lab.html)，先预测控件变化，再观察结果。
  4. 实验二选一：[.ipynb](labs/09_moe.ipynb) 或 [VS Code # %% .py](labs/09_moe.py)。
  5. 填写 [starter 18](labs/starter/18_moe_systems.py) 中保留的核心空缺。
  6. 自动核查：`uv run llm-course exercises check 18`。
  7. **交付物**：共享专家与 upcycling 初始化。
- **完成标准**：初始化输出可比较；说明对称性如何被打破
- **一手来源**：sparse_upcycling

### 第 39 课：专家并行、DeepSeekMoE 与现代变体

- **学习目标**：理解 all-to-all 通信；串起 DeepSeek V1-V3 路由演化
- **前置知识**：完成第 38 课
- **预计时间**：8-10 小时，可按掌握程度拆分多天完成
- **本课内容**：通信量纸面估算；aux-loss-free bias 模拟
- **按此顺序学习**：
  1. 阅读 [本课完整讲义](readings/lessons/39_expert_parallel.md)。
  2. 按需阅读 [07_moe.md](readings/stages/07_moe.md)，用于补齐阶段背景和方法。
  3. 打开 [互动图](readings/interactive/architecture-lab.html)，先预测控件变化，再观察结果。
  4. 实验二选一：[.ipynb](labs/09_moe.ipynb) 或 [VS Code # %% .py](labs/09_moe.py)。
  5. 填写 [starter 18](labs/starter/18_moe_systems.py) 中保留的核心空缺。
  6. 自动核查：`uv run llm-course exercises check 18`。
  7. **交付物**：dispatch/all-to-all 通信估算。
- **完成标准**：区分算法收益与系统收益；能批判技术报告证据
- **一手来源**：deepseekmoe


## 阶段 8：后训练与推理能力（第 40-44 课）

### 第 40 课：指令数据与 SFT

- **学习目标**：理解 prompt/response masking；建立 chat template 概念
- **前置知识**：完成第 39 课
- **预计时间**：8-10 小时，可按掌握程度拆分多天完成
- **本课内容**：玩具指令微调；只对回答计算损失
- **按此顺序学习**：
  1. 阅读 [本课完整讲义](readings/lessons/40_sft_data_contract.md)。
  2. 按需阅读 [08_posttraining.md](readings/stages/08_posttraining.md)，用于补齐阶段背景和方法。
  3. 打开 [互动图](readings/interactive/training-and-alignment.html)，先预测控件变化，再观察结果。
  4. 实验二选一：[.ipynb](labs/10_posttraining.ipynb) 或 [VS Code # %% .py](labs/10_posttraining.py)。
  5. 填写 [starter 04](labs/starter/04_sft_shift.py) 中保留的核心空缺。
  6. 自动核查：`uv run llm-course exercises check 04`。
  7. **交付物**：SFT response mask 和数据契约。
- **完成标准**：prompt token 不计损失；保留基础能力评测
- **一手来源**：instructgpt

### 第 41 课：LoRA 与 QLoRA

- **学习目标**：理解低秩更新；区分训练参数和冻结参数
- **前置知识**：完成第 40 课
- **预计时间**：8-10 小时，可按掌握程度拆分多天完成
- **本课内容**：低秩矩阵拟合；rank 消融
- **按此顺序学习**：
  1. 阅读 [本课完整讲义](readings/lessons/41_lora_qlora.md)。
  2. 按需阅读 [08_posttraining.md](readings/stages/08_posttraining.md)，用于补齐阶段背景和方法。
  3. 打开 [互动图](readings/interactive/training-and-alignment.html)，先预测控件变化，再观察结果。
  4. 实验二选一：[.ipynb](labs/10_posttraining.ipynb) 或 [VS Code # %% .py](labs/10_posttraining.py)。
  5. 填写 [starter 19](labs/starter/19_posttraining.py) 中保留的核心空缺。
  6. 自动核查：`uv run llm-course exercises check 19`。
  7. **交付物**：LoRA merge/unmerge 等价。
- **完成标准**：合并前后输出一致；仅 LoRA 参数获得梯度
- **一手来源**：lora

### 第 42 课：偏好、奖励模型与 PPO

- **学习目标**：把偏好转为相对分数；理解策略、价值和 KL 约束
- **前置知识**：完成第 41 课
- **预计时间**：8-10 小时，可按掌握程度拆分多天完成
- **本课内容**：成对奖励损失；KL 惩罚曲线
- **按此顺序学习**：
  1. 阅读 [本课完整讲义](readings/lessons/42_reward_model_ppo.md)。
  2. 按需阅读 [08_posttraining.md](readings/stages/08_posttraining.md)，用于补齐阶段背景和方法。
  3. 打开 [互动图](readings/interactive/training-and-alignment.html)，先预测控件变化，再观察结果。
  4. 实验二选一：[.ipynb](labs/10_posttraining.ipynb) 或 [VS Code # %% .py](labs/10_posttraining.py)。
  5. 填写 [starter 19](labs/starter/19_posttraining.py) 中保留的核心空缺。
  6. 自动核查：`uv run llm-course exercises check 19`。
  7. **交付物**：奖励模型与 PPO toy objective 边界。
- **完成标准**：优选样本分数上升；解释 reward hacking
- **一手来源**：ppo

### 第 43 课：DPO：绕开奖励模型的偏好优化

- **学习目标**：理解策略/参考模型 log-ratio；实现稳定 DPO loss
- **前置知识**：完成第 42 课
- **预计时间**：8-10 小时，可按掌握程度拆分多天完成
- **本课内容**：玩具偏好数据；beta 消融
- **按此顺序学习**：
  1. 阅读 [本课完整讲义](readings/lessons/43_dpo_preference_optimization.md)。
  2. 按需阅读 [08_posttraining.md](readings/stages/08_posttraining.md)，用于补齐阶段背景和方法。
  3. 打开 [互动图](readings/interactive/training-and-alignment.html)，先预测控件变化，再观察结果。
  4. 实验二选一：[.ipynb](labs/10_posttraining.ipynb) 或 [VS Code # %% .py](labs/10_posttraining.py)。
  5. 填写 [starter 19](labs/starter/19_posttraining.py) 中保留的核心空缺。
  6. 自动核查：`uv run llm-course exercises check 19`。
  7. **交付物**：sequence log-prob 与 DPO 数值核查。
- **完成标准**：数值测试通过；说明 DPO 的数据假设
- **一手来源**：dpo

### 第 44 课：GRPO、RLVR 与推理训练

- **学习目标**：理解组内相对优势；区分可验证奖励和主观奖励
- **前置知识**：完成第 43 课
- **预计时间**：8-10 小时，可按掌握程度拆分多天完成
- **本课内容**：group advantage；玩具算术 verifier
- **按此顺序学习**：
  1. 阅读 [本课完整讲义](readings/lessons/44_grpo_rlvr_reasoning.md)。
  2. 按需阅读 [08_posttraining.md](readings/stages/08_posttraining.md)，用于补齐阶段背景和方法。
  3. 打开 [互动图](readings/interactive/training-and-alignment.html)，先预测控件变化，再观察结果。
  4. 实验二选一：[.ipynb](labs/10_posttraining.ipynb) 或 [VS Code # %% .py](labs/10_posttraining.py)。
  5. 填写 [starter 19](labs/starter/19_posttraining.py) 中保留的核心空缺。
  6. 自动核查：`uv run llm-course exercises check 19`。
  7. **交付物**：GRPO/RLVR 组内优势与安全评测。
- **完成标准**：常数奖励时优势为零；解释奖励稀疏与作弊
- **一手来源**：deepseekmath


## 阶段 9：推理优化与毕业项目（第 45-48 课）

### 第 45 课：权重量化与误差

- **学习目标**：理解 scale/zero-point/group；测量大小与误差权衡
- **前置知识**：完成第 44 课
- **预计时间**：8-10 小时，可按掌握程度拆分多天完成
- **本课内容**：对称 int8/int4 伪量化；层级误差
- **按此顺序学习**：
  1. 阅读 [本课完整讲义](readings/lessons/45_quantization.md)。
  2. 按需阅读 [09_inference.md](readings/stages/09_inference.md)，用于补齐阶段背景和方法。
  3. 打开 [互动图](readings/interactive/serving-lab.html)，先预测控件变化，再观察结果。
  4. 实验二选一：[.ipynb](labs/11_inference_serving.ipynb) 或 [VS Code # %% .py](labs/11_inference_serving.py)。
  5. 填写 [starter 20](labs/starter/20_inference_systems.py) 中保留的核心空缺。
  6. 自动核查：`uv run llm-course exercises check 20`。
  7. **交付物**：量化误差、体积和饱和报告。
- **完成标准**：饱和处理正确；报告误差和理论体积
- **一手来源**：gptq

### 第 46 课：PagedAttention 与推测解码

- **学习目标**：理解非连续 KV 页面；理解 draft/verify 保持分布正确
- **前置知识**：完成第 45 课
- **预计时间**：8-10 小时，可按掌握程度拆分多天完成
- **本课内容**：页面分配模拟；块式贪心推测解码：一次 target 调用验证多个候选
- **按此顺序学习**：
  1. 阅读 [本课完整讲义](readings/lessons/46_paged_attention_continuous_batching.md)。
  2. 按需阅读 [09_inference.md](readings/stages/09_inference.md)，用于补齐阶段背景和方法。
  3. 打开 [互动图](readings/interactive/serving-lab.html)，先预测控件变化，再观察结果。
  4. 实验二选一：[.ipynb](labs/11_inference_serving.ipynb) 或 [VS Code # %% .py](labs/11_inference_serving.py)。
  5. 填写 [starter 20](labs/starter/20_inference_systems.py) 中保留的核心空缺。
  6. 自动核查：`uv run llm-course exercises check 20`。
  7. **交付物**：分页缓存、连续批处理和随机推测解码。
- **完成标准**：draft 等于 target 时完整接受；解释拒绝后的修正
- **一手来源**：pagedattention

### 第 47 课：基准设计与公平比较

- **学习目标**：区分 latency、throughput、TTFT、TPOT；记录完整实验条件
- **前置知识**：完成第 46 课
- **预计时间**：8-10 小时，可按掌握程度拆分多天完成
- **本课内容**：预热与重复测量；参数/FLOPs 匹配
- **按此顺序学习**：
  1. 阅读 [本课完整讲义](readings/lessons/47_speculative_decoding_benchmarks.md)。
  2. 按需阅读 [capstone.md](readings/references/capstone.md)、[09_inference.md](readings/stages/09_inference.md)，用于补齐阶段背景和方法。
  3. 打开 [互动图](readings/interactive/serving-lab.html)，先预测控件变化，再观察结果。
  4. 实验二选一：[.ipynb](labs/11_inference_serving.ipynb) 或 [VS Code # %% .py](labs/11_inference_serving.py)。
  5. 填写 [starter 20](labs/starter/20_inference_systems.py) 中保留的核心空缺。
  6. 自动核查：`uv run llm-course exercises check 20`。
  7. **交付物**：TTFT/TPOT/吞吐公平基准。
- **完成标准**：一次只改变一个因素；原始结果可追溯
- **一手来源**：helm

### 第 48 课：毕业项目与知识答辩

- **学习目标**：完成 Dense/Attention/MoE 对照；区分证据、推断和限制
- **前置知识**：完成第 47 课
- **预计时间**：8-10 小时，可按掌握程度拆分多天完成
- **本课内容**：三个模型公平训练；生成和失败案例分析
- **按此顺序学习**：
  1. 阅读 [本课完整讲义](readings/lessons/48_capstone_defense.md)。
  2. 按需阅读 [capstone.md](readings/references/capstone.md)、[09_inference.md](readings/stages/09_inference.md)，用于补齐阶段背景和方法。
  3. 打开 [互动图](readings/interactive/serving-lab.html)，先预测控件变化，再观察结果。
  4. 实验二选一：[.ipynb](labs/11_inference_serving.ipynb) 或 [VS Code # %% .py](labs/11_inference_serving.py)。
  5. 本课没有独立 starter，完成研究记录或实验报告。
  6. 按讲义中的断言复现结果，并检查输出中没有隐藏状态。
  7. **交付物**：可复现毕业报告、rubric 与口述答辩。
- **完成标准**：测试全部通过；报告损失/速度/内存/路由/限制；能解释关键实现而非只运行代码
- **一手来源**：transformer

## 48 课之后：多模态选修

先阅读[多模态桥接讲义](readings/extensions/multimodal.md)和[数据流互动图](readings/interactive/multimodal-flow.html)，再从以下格式任选一种：

- [多模态 Notebook](labs/optional/80_multimodal_bridge.ipynb)
- [多模态 VS Code 实验](labs/optional/80_multimodal_bridge.py)
- [多模态 starter](labs/starter/21_multimodal_bridge.py)

## 学习目录边界

`learning/readings/` 只放阅读与互动材料；`learning/labs/` 只放可运行或填写的实验。
环境在 `setup/`，课程配置在 `course/`，公开核查在 `checks/`，参考实现位于 `src/`。
