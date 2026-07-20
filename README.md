# LLM-Zero2Pro：从零到前沿的文本 LLM 课程

![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white) ![PyTorch 2.7+](https://img.shields.io/badge/PyTorch-2.7%2B-EE4C2C?logo=pytorch&logoColor=white) ![Course](https://img.shields.io/badge/Course-48%20lessons-6C5CE7)

一套中文、从零实现优先、CPU 可完成的文本大语言模型课程。主线从张量、自动微分、分词和 Transformer 开始，覆盖 RoPE、GQA、KV Cache、FlashAttention、MLA、线性注意力、MoE、SFT、LoRA、DPO、GRPO、量化、PagedAttention 与推测解码。多模态位于 48 课之外的选修区；不包含 RAG 和 Agent。

## 唯一学习入口

打开 **[learning/README.md](learning/README.md)**。这个目录按第 1–48 课写明每课内容、学习顺序、讲义、互动图、实验、starter、核查命令、交付物和完成标准。

课程没有短版/完整版、初级/高级等分叉：

- 初学者从第 1 课逐项完成。
- 有基础的学习者仍按顺序做每课验收；已经掌握的课可以快速通过。
- 中途回来时，从第一个尚未满足“完成标准”的课继续。
- “一课”不是固定七天；可以按自己的时间拆分或合并进度。

## 推荐：使用 VS Code

### Windows 与 macOS 通用

克隆仓库并进入根目录后运行：

```text
uv python install 3.12
uv sync
uv run llm-course doctor
uv run llm-course vscode
```

最后一条命令只打开 VS Code workspace 和课程目录，不会启动 JupyterLab。也可以手动运行：

```text
code .
```

然后打开 `learning/README.md`。完整操作见 [VS Code 学习指南](setup/vscode.md)。

在 VS Code 中选择项目解释器：

- Windows：`.venv\Scripts\python.exe`
- macOS/Linux：`.venv/bin/python`

每个实验同时提供 `.ipynb` 和带 `# %%` 单元的 `.py`，同一实验任选一种即可。

## 可选：使用 JupyterLab

如果更喜欢浏览器 Notebook 界面：

```text
uv run llm-course lab
```

它只打开 `learning/` 文件浏览器，不再跳转欢迎 Notebook。先看 `learning/README.md`，再进入 `learning/labs/`。详见 [JupyterLab 可选指南](setup/jupyterlab.md)。

## 安装支持

| 系统 | 支持情况 | 指南 |
|---|---|---|
| Windows 10/11 x64 | 完整 CPU 主线 | [Windows 安装](setup/environment.md#windows-安装) |
| macOS 14+ Apple Silicon | 完整 CPU 主线，可检测 MPS | [macOS 安装](setup/environment.md#macos-apple-silicon-安装) |
| Intel Mac | 当前 PyTorch 锁文件不保证支持 | [Intel Mac 说明](setup/environment.md#intel-mac) |
| Colab/Kaggle Linux | 可选 GPU 实验 | [托管 GPU](setup/environment.md#托管-gpu-选修) |

## 仓库结构

```text
learning/
├─ README.md                 # 唯一课程目录，第 1–48 课
├─ readings/                 # 讲义、阶段总结、互动图、论文与多模态扩展
└─ labs/                     # ipynb/py 实验、starter、示例

setup/                       # Windows、macOS、VS Code、JupyterLab 环境
course/                      # 机器读取的 48 课和练习清单
checks/                      # starter 的公开行为核查
src/                         # CLI 与完整参考实现
solutions/                   # 选定答案说明，核查通过后再读
tests/                       # 仓库级回归测试
scripts/                     # 课程资产同步工具
.vscode/                     # 推荐扩展、设置和课程任务
```

互动资源总入口是 [learning/readings/interactive/index.html](learning/readings/interactive/index.html)。学习时通常只需要打开 `learning/`。`course/` 是课程数据库，不是另一套教程；`checks/`、`src/` 和 `tests/` 也不是阅读入口。

## 代码模板与核查

starter 位于 `learning/labs/starter/`，核心实现故意留空。不要先复制 `src/llm_from_scratch/`：

```text
uv run llm-course exercises list
uv run llm-course exercises check 07
uv run llm-course exercises check rope
```

`exercises check` 检查学员填写内容；`course check` 检查仓库和课程资产，两者职责不同。

## 常用命令

| 目的 | 命令 |
|---|---|
| 打开 VS Code 和课程目录 | `uv run llm-course vscode` |
| 环境诊断 | `uv run llm-course doctor` |
| 查看 48 课目录文本 | `uv run llm-course course path` |
| 同步 `learning/README.md` | `uv run llm-course course path --write` |
| 列出 starter | `uv run llm-course exercises list` |
| 核查 starter | `uv run llm-course exercises check <编号或别名>` |
| 可选 JupyterLab | `uv run llm-course lab` |
| 检查双格式实验 | `uv run python scripts/sync_labs.py --check` |
| 完整课程健康检查 | `uv run llm-course course check` |

## 学习闭环与毕业标准

每课固定按“讲义 → 互动图 → 实验 → starter → 核查 → 交付物”推进。完成 48 课后，应能：

- 从零解释并实现 tokenizer、因果注意力、Transformer、KV Cache 和 Tiny GPT。
- 比较 MHA、GQA、MLA、线性/稀疏注意力的计算和缓存权衡。
- 实现并诊断 Top-k MoE 的路由、容量、负载与数值稳定性。
- 解释 SFT、LoRA、DPO、GRPO、量化、分页缓存和推测解码的输入契约与边界。
- 用可复现实验区分论文主张、公开证据、自己的推断和未复现部分。

课程资料核验日期记录在 `course/course.yaml`；事实优先使用原论文、官方技术报告、模型卡、代码和文档。