# LLM-Zero2Pro：从零到前沿的文本 LLM 课程

![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white) ![PyTorch 2.7+](https://img.shields.io/badge/PyTorch-2.7%2B-EE4C2C?logo=pytorch&logoColor=white) ![Course](https://img.shields.io/badge/Course-48%20weeks-6C5CE7)

一套中文、从零实现优先、CPU 必修可跑的文本大语言模型教程。目标不是背模型名称，而是建立可以反复验证的学习闭环：

> 直觉 → 张量形状 → 必要公式 → 最小代码 → 对照实验 → 论文证据

课程从张量、自动微分和 BPE 开始，经过 Transformer、RoPE、GQA、KV Cache、MoE、后训练与推理优化，最终完成 Tiny GPT 和缩尺对照实验。48 周主线严格聚焦文本 LLM；多模态是课外选修包；不包含 RAG 和 Agent。

## 第一次来这里，从这四步开始

先阅读[跨平台环境搭建](docs/00_environment.md)，在仓库根目录完成：

```text
uv python install 3.12
uv sync
uv run llm-course doctor
uv run llm-course lab
```

最后一条命令会启动 JupyterLab，并直接打开 `notebooks/00_START_HERE.ipynb`。不要第一次进入 Lab 后随意浏览全部目录；欢迎页会先检查 Kernel，再帮助你判断路线并打开第一本 Notebook。

如果你已经执行过 `uv sync`，以后回来通常只需要：

```text
uv run llm-course lab
```

## 系统支持

| 系统 | 本课程路径 | 安装入口 |
|---|---|---|
| Windows 10/11 x64 | CPU 必修完整支持 | [Windows 安装步骤](docs/00_environment.md#windows-安装) |
| macOS 14+ Apple Silicon | CPU 必修完整支持；可检测 MPS | [macOS 安装步骤](docs/00_environment.md#macos-apple-silicon-安装) |
| Intel Mac | 当前锁文件没有对应的最新 PyTorch wheel | [Intel Mac 说明](docs/00_environment.md#intel-mac-怎么办) |
| Colab / Kaggle Linux | 可选 GPU、Triton 与性能实验 | [托管 GPU 路径](docs/00_environment.md#免费托管-gpu-选修) |

所有必修 Notebook 默认以 CPU 为数值基线，不要求 CUDA，也不会因为 Mac 有 MPS 就改变验收结果。

## 进入 JupyterLab 后怎么做

1. 保持启动 JupyterLab 的终端窗口运行。
2. 在欢迎页运行第一个环境检查单元，确认 Python 3.12、PyTorch 和项目路径。
3. 按下方的路线判断表选择 15 周、48 周或专题学习；不确定就从 [`notebooks/core/01_shapes_and_autograd.ipynb`](notebooks/core/01_shapes_and_autograd.ipynb) 开始。
4. 每个 Notebook 完成后使用 **Kernel → Restart Kernel and Run All Cells**，排除残留变量。
5. 回到终端运行对应 starter 核查，而不是直接复制 `src/` 中的参考实现。

如果你使用了普通的 `uv run jupyter lab`，请手动打开 [`notebooks/00_START_HERE.ipynb`](notebooks/00_START_HERE.ipynb)。推荐始终使用课程命令，它会固定仓库根目录并打开欢迎页。

## 先选择适合你的课程路线

这里的“选择路线”不是安装另一个版本，也不会隐藏或删除文件。三条路线共用同一套环境、讲义、Notebook 和代码模板；路线只决定**学习哪些主题、采用什么顺序以及按什么标准结课**。当前 CLI 不会替你注册账号或锁定路线，最终以你正在使用的路线表和 `progress.yaml` 中的记录为准。

### 先用一分钟判断

| 如果你符合下面的情况 | 选择 | 原因 |
|---|---|---|
| 第一次系统学习 LLM；对张量、梯度、交叉熵或注意力仍不熟 | **15 周核心路线** | 先建立从数学、分词、注意力到 Tiny GPT、现代 Decoder 和 MoE 的主干 |
| 已经能解释并实现 causal self-attention 和 Transformer Block，希望系统学习训练、前沿架构、后训练与推理服务 | **48 周完整路线** | 不跳过数据治理、训练系统、评测、对齐和服务工程 |
| 已完成经典 Transformer，只想补 RoPE/GQA、注意力前沿、MoE、后训练或推理中的一个主题 | **专题路线** | 先做前置自测，再按专题顺序学习；它不是独立的完整毕业路线 |
| 主要想了解图像怎样接入文本 Decoder | **多模态选修** | 先完成 Tiny GPT 和现代 Decoder，再做课外扩展；它不占 48 周主线 |

如果有任何一项拿不准，选 15 周路线。这里的“一周”表示一个学习单元，不要求与自然周严格对齐；每周只有 3–4 小时时，可以把一个单元拆成两周。

### 15 周核心路线：第一次系统学习时选它

它从 48 周课程中抽取 15 个关键学习单元，因此路线表中的“原课程周”会从 4 跳到 7、从 20 跳到 35，这是有意的，不是文件缺失。15 个学习单元仍然按表格从上到下连续完成。

在仓库根目录执行：

```text
# 先在终端查看完整的 15 个学习单元
uv run llm-course course path --weeks 15

# 再进入统一的 Jupyter 入口
uv run llm-course lab
```

然后打开[15 周核心学习路径](docs/core_learning_path.md)，从表格第一行开始。不要按照原课程周编号自行补齐被跳过的周次；被跳过的内容留给以后升级到 48 周路线。

### 48 周完整路线：想完整掌握训练到服务时选它

48 周路线按原课程周 1 → 48 顺序学习，覆盖全部九个阶段。它不要求你一开始就会所有现代主题，但默认你至少能读懂 Python/PyTorch，并愿意完整完成训练系统、评测和工程实验。

```text
# 输出 48 周逐周路线；从第 1 周顺序执行，不要按 Notebook 数量估算周数
uv run llm-course course path --weeks 48

uv run llm-course lab
```

同一本 Notebook 可能服务连续数周：例如 `06_modern_decoder.ipynb` 同时承载 RMSNorm、SwiGLU、RoPE、GQA 和 KV Cache。每一周只完成路线表指定的知识点、starter 或产出，不是第一次打开就把整本 Notebook 当作一周做完。

### 专题路线：只补一个主题时这样选

专题路线适合复习和补缺，不替代 15/48 周结课要求。按下面的顺序进入，不要直接从参考实现 `src/` 开始：

| 目标专题 | 最低前置知识 | 建议顺序 |
|---|---|---|
| 注意力、RoPE、GQA 与 KV Cache | 能解释 Softmax、矩阵乘法和张量形状 | `04_attention_mechanics` → `06_modern_decoder` → 架构互动图 |
| FlashAttention、MLA、线性注意力与 DeltaNet | 已完成经典注意力和现代 Decoder | `06_modern_decoder` → `08_attention_frontiers` |
| MoE | 会使用 PyTorch 自动微分，并理解 Transformer MLP | `06_modern_decoder` → `09_moe` → starter 10/05/18 |
| SFT、LoRA、DPO 与 GRPO | 能训练 Tiny GPT，并理解 token-level log-prob | `05_tiny_gpt` → `10_posttraining` |
| 量化、PagedAttention 与推测解码 | 理解 KV Cache 和自回归生成 | `06_modern_decoder` → `11_inference_serving` |
| 多模态接口（选修） | 已理解文本 Decoder、Embedding 和位置编码 | `05_tiny_gpt` → `06_modern_decoder` → `optional/80_multimodal_bridge` |

完整的论文关系和方法演化见[架构演化指南](docs/architecture_evolution.md)，可交互实验从[互动实验总入口](docs/interactive/index.html)进入。

### 选完路线以后，第一周具体做什么

无论选择哪条路线，都对路线表的**当前一行**执行同一个闭环：

1. 打开该行对应的 `docs/` 讲义，只读当前知识点。
2. 运行该行对应的 Notebook 实验，并写下预测、形状和一个失败反例。
3. 在 `exercises/starter/` 填写该行列出的 starter；如果这一周是研究产出，则按 deliverable 写笔记或报告。
4. 运行 `uv run llm-course exercises check <编号或别名>`；没有 starter 的周次按讲义 rubric 自查。
5. 使用 **Restart Kernel and Run All Cells** 重跑 Notebook。
6. 在 `progress.yaml` 中把对应的**原课程周编号**更新为 `completed`，记录投入时间和复盘，再进入路线表下一行。

例如，15 周路线的第一个学习单元对应原课程第 1 周：先运行环境检查并保存诊断信息；第二个学习单元对应原课程第 2 周，再进入形状与自动微分实验。15 周路线中即使下一行跳到原课程第 7 周，`progress.yaml` 也应更新第 7 周，而不是把它改写成“第 5 周”。

### 中途能否切换路线

- **15 周 → 48 周**：保留已经完成的原课程周状态，运行 `course path --weeks 48`，从最早一个未完成周次继续，并补齐此前跳过的内容。
- **48 周 → 15 周**：无需删除进度，只按 15 周路线列出的原课程周继续；未选中的周次保留原状态。
- **专题 → 主线**：专题完成记录可以保留，但仍要从 15/48 周路线中最早未完成的前置单元继续。

课程选择不会改变 Git 分支或依赖环境，所以切换路线不需要重新安装。不要修改 `course/stages/*.yaml` 来记录个人选择；这些文件是全体学习者共享的课程契约。

## 每周固定学习循环

每周只处理当前主题，按相同顺序完成：

1. **讲义**：写下直觉答案，标注所有张量形状。
2. **交互图**：先预测参数变化结果，再拖动滑块。
3. **Notebook**：运行最小实验，并保留失败输入。
4. **starter**：关闭参考实现，填写核心 `TODO`。
5. **核查**：运行单题公开测试，从第一条失败信息开始修正。
6. **参考实现与论文**：最后才比较实现差异和证据。
7. **复盘**：在 `progress.yaml` 记录完成状态、反例和未解决问题。

推荐每周 8–10 小时：讲义与直觉 2 小时、数学 1.5 小时、代码 3 小时、论文 2 小时、复盘 1 小时。

## 每类文件负责什么

| 目录 | 用途 | 什么时候打开 | 是否建议修改 |
|---|---|---|---|
| `docs/` | 中文讲义、环境和架构演化 | Notebook 前后都可查 | 通常只读 |
| `notebooks/core/` | 11 本按依赖排序的必修实验 | 当前周实验阶段 | 可以添加自己的观察单元 |
| `notebooks/optional/` | 多模态桥接与 GPU 环境选修 | 完成相应前置知识后 | 按需使用 |
| `exercises/starter/` | 故意留空的核心代码 | 学完知识点后 | **主要填写区域** |
| `exercises/checks/` | starter 的公开行为测试 | 填写过程中 | 通常只读 |
| `src/llm_from_scratch/` | 完整参考实现 | starter 通过之后 | 不要用它替代练习 |
| [`course/`](course/README.md) | 课程控制中心：元数据、9 个阶段、48 周资产映射 | CLI 自动读取；维护课程时查看 | 学习者通常只读 |
| `papers/` | 论文目录、候选池和笔记模板 | 建立代码直觉之后 | 在 `notes/` 写阅读记录 |
| `progress.yaml` | 个人学习进度 | 每周结束 | **持续更新** |

仓库顶层只保留入口和跨模块配置，学习资产按职责分开：

```text
LLM-Zero2Pro/
├─ course/                 # 课程控制层，不存讲义正文
│  ├─ course.yaml          # 课程级元数据与路径
│  ├─ roadmap.yaml         # 小型清单，声明阶段文件
│  └─ stages/              # 9 个阶段、48 周资产契约
├─ docs/                   # 讲义与互动页面
├─ notebooks/
│  ├─ 00_START_HERE.ipynb  # 唯一 Jupyter 入口
│  ├─ core/                # 01–11 必修实验
│  └─ optional/            # 80/90 选修实验
├─ exercises/              # starter、checker 与 manifest
├─ src/                    # 完整参考实现与 CLI
└─ tests/                  # 数值、课程资产与 Notebook 执行测试
```

`course/` 不是另一套教程。它相当于课程数据库，供 `llm-course course path/check` 和 Notebook 元数据同步读取；真正学习内容在 `docs/`、`notebooks/`、`exercises/`。

## Notebook 顺序

| 顺序 | Notebook | 核心问题 |
|---:|---|---|
| 0 | `notebooks/00_START_HERE.ipynb` | 当前环境和学习入口是否正确？ |
| 1 | `notebooks/core/01_shapes_and_autograd.ipynb` | 张量形状和梯度怎样流动？ |
| 2 | `notebooks/core/02_neural_language_models.ipynb` | Bigram、MLP 与 RNN 怎样逐步扩大上下文？ |
| 3 | `notebooks/core/03_tokenization_and_bpe.ipynb` | Unicode、字节和 BPE merge 怎样改变 token？ |
| 4 | `notebooks/core/04_attention_mechanics.ipynb` | Q/K/V 与组合 mask 怎样阻止未来和 padding 泄漏？ |
| 5 | `notebooks/core/05_tiny_gpt.ipynb` | 如何训练、保存、加载并生成 Tiny GPT？ |
| 6 | `notebooks/core/06_modern_decoder.ipynb` | RMSNorm、SwiGLU、RoPE、GQA 怎样组成现代 Decoder？ |
| 7 | `notebooks/core/07_pretraining_systems.ipynb` | 数据去重、packing、优化器和评测怎样形成训练闭环？ |
| 8 | `notebooks/core/08_attention_frontiers.ipynb` | Flash、稀疏、MLA、线性/Delta 状态有什么边界？ |
| 9 | `notebooks/core/09_moe.ipynb` | Top-k 路由、容量和 dropping 如何影响负载？ |
| 10 | `notebooks/core/10_posttraining.ipynb` | SFT、LoRA、DPO 与 GRPO 的输入契约是什么？ |
| 11 | `notebooks/core/11_inference_serving.ipynb` | 分页缓存、连续批处理和推测解码如何影响服务指标？ |
| 选修 | `notebooks/optional/80_multimodal_bridge.ipynb` | 视觉 patch 怎样经 projector 接到文本 Decoder？ |
| 选修 | `notebooks/optional/90_gpu_environment_check.ipynb` | 托管 GPU 是否真的可用？ |

更详细的运行约定见 [Notebook 使用指南](notebooks/README.md)。

## 代码模板与核查

练习清单包含 20 个文本主线 starter/checker 与 1 个多模态选修。既保留稳定编号 01–10，也补齐自动微分、神经 LM、TinyGPT、数据/优化器、前沿注意力、MoE 系统、后训练和推理服务；实际周次与路径以 `exercises/manifest.yaml` 为唯一来源。

```text
# 查看编号、周次、主题和填写状态
uv run llm-course exercises list

# 核查单题：编号和别名都可以
uv run llm-course exercises check 07
uv run llm-course exercises check rope

# 核查全部已填写模板
uv run llm-course exercises check all
```

模板初始包含 `NotImplementedError`，所以第一次练习核查失败是正常起点。`course check` 验证仓库健康，`exercises check` 验证你的填空实现，两者不能互相替代。完整说明见[代码模板与核查](docs/code_templates.md)。

## 交互式学习资源

- [互动实验总入口](docs/interactive/index.html)：按阶段进入全部离线实验。
- [基础实验室](docs/interactive/foundations-lab.html)：计算图分支梯度、BPE merge 与上下文模型。
- [架构实验室](docs/interactive/architecture-lab.html)：RoPE 在 Q/K 内的旋转、MHA/MQA/GQA/MLA/线性状态与缓存。
- [训练与对齐](docs/interactive/training-and-alignment.html)：数据过滤、并行内存、SFT/DPO/GRPO。
- [推理服务](docs/interactive/serving-lab.html)：PagedAttention、连续批处理、TTFT/TPOT 与推测解码。
- [架构演化图](docs/interactive/architecture-evolution.html)：按基础共识、公开模型采用、前沿预印本分层。
- [多模态数据流](docs/interactive/multimodal-flow.html)：选修的 patch/projector/token 接口。

页面均为无网络依赖的独立 HTML，支持键盘和窄屏。克隆后用浏览器打开；GitHub 文件预览不会执行交互脚本。

## 常用命令

| 目的 | 命令 |
|---|---|
| 环境诊断 | `uv run llm-course doctor` |
| 打开欢迎页和 JupyterLab | `uv run llm-course lab` |
| 远程机器启动 Lab | `uv run llm-course lab --no-browser` |
| 指定端口 | `uv run llm-course lab --port 8890` |
| 列出代码模板 | `uv run llm-course exercises list` |
| 核查代码模板 | `uv run llm-course exercises check <编号或别名>` |
| 校验论文目录 | `uv run llm-course papers validate` |
| 生成论文关系图 | `uv run llm-course papers graph` |
| 完整课程健康检查 | `uv run llm-course course check` |
| 生成 15/48 周路线 | `uv run llm-course course path --weeks 15` |
| 验证 48 周资产闭环但不跑测试 | `uv run llm-course course check --no-tests` |

论文候选更新和三遍阅读法分别见 [论文目录](papers/README.md)与[论文阅读工作流](docs/01_paper_workflow.md)。

## 常见问题

- **Jupyter 里找不到 `llm_course`**：关闭当前服务，从仓库根目录执行 `uv sync`，再运行 `uv run llm-course lab`。
- **打开 Lab 后不知道点哪里**：打开 `notebooks/00_START_HERE.ipynb`，不要直接从 `src/` 开始读。
- **PyTorch 下载很慢**：CPU wheel 仍然较大；不要中断后改用多个全局 Python。平台细节见环境文档。
- **starter 测试全部失败**：先只核查当前编号。未填写模板失败是预期行为。
- **交互 HTML 只显示源码**：在系统文件管理器中用浏览器打开 `docs/interactive/*.html`。
- **想使用 vLLM/Triton/CUDA**：它们属于 Linux GPU 选修，不是 Windows/macOS CPU 主线的前置条件。

## 学完后的毕业标准

- 能从零解释并实现 tokenizer、因果注意力、Transformer、KV Cache 和小型 GPT。
- 能比较 MHA、GQA、MLA、线性注意力和稀疏注意力的计算/内存权衡。
- 能实现 Top-k MoE，诊断负载不均、容量溢出和路由不稳定。
- 能解释 SFT、LoRA、DPO、GRPO、量化、PagedAttention 和推测解码的核心机制。
- 能用相同数据、训练 token 数和近似活跃 FLOPs 完成缩尺对照实验。
- 能区分论文中的作者主张、实验证据、自己的推断和未复现部分。

## 资料时效与边界

论文库核验至 **2026-07-19**，当前收录 96 篇。资料优先使用原论文、官方技术报告、模型卡、代码和文档；候选更新器只进入待审池，不会自动把热门结果升级为必读。涉及“当前主流”的结论必须保留版本日期和证据位置。
