# 从零到前沿：文本 LLM 学习仓库

![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white) ![PyTorch 2.7+](https://img.shields.io/badge/PyTorch-2.7%2B-EE4C2C?logo=pytorch&logoColor=white) ![Course](https://img.shields.io/badge/Course-48%20weeks-6C5CE7)

[15 周核心路径](docs/core_learning_path.md) · [代码模板与核查](docs/code_templates.md) · [交互实验室](docs/interactive/core-concepts.html) · [架构演化图](docs/interactive/architecture-evolution.html) · [论文目录](papers/README.md)


这是一个 **48 周、中文、从零实现优先、CPU 必修可跑** 的文本大语言模型课程。目标不是背诵模型名，而是建立一条能够反复验证的知识链：

> 直觉 → 张量形状 → 必要公式 → 最小代码 → 对照实验 → 论文证据

课程主线借鉴 Stanford CS336 的“从零构建语言模型”方式，并扩展到现代 Decoder、MLA、线性注意力、Gated DeltaNet、稀疏注意力、MoE、后训练和推理优化。首版只学习文本 LLM，不包含多模态、RAG 和 Agent。

## 五分钟开始

Windows 尚未安装工具链时，可先使用 Scoop：

```powershell
scoop install git uv
```

然后在仓库根目录执行：

```powershell
uv python install 3.12
uv sync
uv run llm-course doctor
uv run llm-course papers validate
uv run llm-course course check
```

启动 Notebook：

```powershell
uv run jupyter lab
```

若下载 PyTorch 较慢，请阅读[环境搭建教程](docs/00_environment.md)和 [Scoop 可选工具教程](docs/00_scoop.md)。所有必修测试默认使用 CPU，不需要 CUDA；Python 项目依赖仍由 `uv` 和 `uv.lock` 管理。

## 怎样使用这个仓库

1. 初学者先走[15 周核心路径](docs/core_learning_path.md)，完成 Tiny GPT 闭环后再进入前沿模块。
2. 用[核心概念交互实验室](docs/interactive/core-concepts.html)建立基础直觉，再用[架构演化图](docs/interactive/architecture-evolution.html)串起 RoPE、注意力和 MoE 的代表工作。
3. 从[48 周路线](course/roadmap.yaml)查看本周目标，阅读 `docs/` 中对应阶段讲义。
4. 运行 `notebooks/` 的实验，再阅读 `src/llm_from_scratch/` 的参考实现。
5. 独立完成 `exercises/starter/` 和 `exercises/`，最后再看提示或答案。
6. 按[三遍论文阅读法](docs/01_paper_workflow.md)填写论文笔记。
7. 每周运行 `uv run llm-course course check`，确认仓库和教学资产没有损坏。

推荐每周 8–10 小时：直觉讲解 2 小时、数学 1.5 小时、代码 3 小时、论文 2 小时、复盘 1 小时。完全零基础可把前 8 周各拆为两周。

## 仓库地图

```text
course/                  48 周课程清单和阶段验收
docs/                    中文讲义、环境教程、概念解释
notebooks/               可视化与交互实验
src/llm_from_scratch/    BPE、GPT、注意力、MoE、后训练、推理实现
src/llm_course/          doctor、论文库和课程校验 CLI
papers/                  论文目录、候选池、阅读模板
knowledge/               概念图和结构化知识
exercises/               练习、提示与答案
tests/                   数值一致性和验收测试
```

## 统一命令

```powershell
# 检查 Python、PyTorch、设备和最小张量运算
uv run llm-course doctor

# 校验目录；从三个公开来源拉取候选项；生成 Mermaid 论文图
uv run llm-course papers validate
uv run llm-course papers update --source all --max-results 20
uv run llm-course papers graph

# 校验 48 周课程并执行单元测试与必修 Notebook
uv run llm-course course check

# 代码模板说明见 docs/code_templates.md
uv run llm-course exercises list
uv run llm-course exercises check 07
uv run llm-course exercises check all
```

论文更新器支持 `arxiv`、`semantic-scholar`、`huggingface` 或默认的 `all`。Semantic Scholar 公共请求可能在高峰期被限流；设置可选的 `SEMANTIC_SCHOLAR_API_KEY` 会更稳定。某个来源临时失败时，成功来源仍会写入候选池并记录错误。

`papers update` 只把结果放入候选池，不会自动把热门论文标为必读。候选项会按 arXiv ID、DOI 和规范化标题去重；技术报告、预印本和正式发表版本需人工核验后才能升级。

## 学完后的毕业标准

- 能从零解释并实现 tokenizer、因果注意力、Transformer、KV Cache 和小型 GPT。
- 能比较 MHA、GQA、MLA、线性注意力和稀疏注意力的计算/内存权衡。
- 能实现 Top-k MoE，诊断负载不均、容量溢出和路由不稳定。
- 能解释 SFT、LoRA、DPO、GRPO、量化、PagedAttention 和推测解码的核心机制。
- 能用相同数据、训练 token 数和近似活跃 FLOPs，完成 Dense、注意力变体、MoE 的缩尺对照实验。
- 能清楚区分论文中的作者主张、实验证据、自己的推断和未复现部分。

## 资料时效

论文库核验至 **2026-07-18**，当前收录 78 篇，包含 25 篇 Core、40+ 篇 Deep Dive 和滚动 Frontier。运行论文更新命令可生成新的候选项；所有被提升为课程材料的论文仍需人工核验。
