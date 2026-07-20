# 第 1 周：命令行、Git、Python 与可复现环境

## 课程定位

这一周不是“安装软件”的旁支，而是整门课的实验地基。LLM 实验常同时依赖解释器、原生算子、数据、随机种子和硬件；只记录一句“在我电脑上能跑”无法复现结果。本周建立一个可审计的闭环：从仓库根目录出发，用锁文件恢复隔离环境，用同一入口运行 Python、测试与 Jupyter，并把失败时真正有用的环境快照留下来。后续每个数值结论都应能回答：代码是哪一版、依赖是哪一组、输入来自哪里、命令是什么、输出如何验收。

## 学习目标

学完后应能区分工作区、Git 仓库、Python 解释器、虚拟环境、项目声明 `pyproject.toml` 与锁文件 `uv.lock`；能在 Windows PowerShell 和 macOS/Linux 终端从仓库根目录执行同一组项目命令；能解释为什么 `.venv` 可删除重建而源代码与锁文件必须版本化；能用 `doctor`、测试和 Notebook 三层信号定位问题，而不是反复重装。最终交付物不是截图，而是一段另一台机器可以照抄的复现记录。

## 前置

只要求会打开终端、复制命令并知道文件路径。Windows 命令提示符、PowerShell 与 macOS 的 zsh 语法略有差异，但本课程的项目命令均通过 `uv run ...` 调用，尽量不依赖“激活环境”这一有平台差异的状态。先确认自己位于含 `pyproject.toml`、`uv.lock`、`course/` 的仓库根目录；目录错了时，依赖、模块导入和 Notebook 相对路径都会给出彼此矛盾的错误。

## 自洽直觉

把仓库想成实验配方，把 Python 解释器想成炉子，把虚拟环境想成这次实验专用的器皿，锁文件则是带精确批次的原料清单。`pyproject.toml` 表达“允许使用什么范围的依赖”，`uv.lock` 记录解析后的确定版本；`uv sync --locked` 要求严格按现成锁文件恢复，若声明和锁不一致就失败，从而阻止静默改配方。虚拟环境本身不是成果：它包含大量平台相关二进制，复制给别人既笨重又不可靠，所以应从声明和锁重新生成。

## 数据契约与复现契约

本周的“数据”是一次运行的环境元数据。最小契约包括：仓库提交标识、操作系统、CPU/GPU、Python 与 PyTorch 版本、锁文件状态、随机种子、工作目录、完整入口命令和验收输出。路径契约是“从仓库根目录执行”；解释器契约是“由 `uv run` 选择项目环境”；依赖契约是“锁文件不被临时更新”；随机性契约是“代码显式设置种子，但不承诺不同平台和不同 PyTorch 版本逐位一致”。PyTorch 官方也明确说明，跨版本、跨平台以及 CPU/GPU 之间不能保证完全相同的结果，因此复现记录必须写清边界。

## 机制与复现推导

一次实验可抽象为输出 `y = F(c, e, d, s, h)`，其中 `c` 是代码提交，`e` 是依赖环境，`d` 是数据版本，`s` 是随机状态，`h` 是硬件与算子后端。只固定 `s` 而改变 `e`，浮点归约顺序或算子实现仍可能改变 `y`；只提交 `c` 而不提交锁文件，依赖解析也可能漂移。因而可复现不是单个 seed，而是对这五项建立可追溯约束。Git 负责 `c`，`uv.lock` 约束 `e`，数据校验和约束 `d`，seed 约束 `s`，诊断快照记录 `h`。

## 命令例与数值例

假设 A 机器使用 PyTorch 2.x、CPU、seed 2026，B 机器仅知道 seed，却解析到不同依赖；两次损失分别为 2.3011 与 2.3007，不能据此断言代码回归。若两台机器同时固定 commit、lock、输入字节和执行入口，且验收采用合理容差，例如 `abs(loss_a-loss_b) < 1e-5`，差异才具有诊断意义。首次运行建议依次执行：`uv sync --locked`、`uv run llm-course doctor`、`uv run pytest -q`、`uv run jupyter lab`。在 Windows 与 macOS 上命令相同，区别主要是安装 `uv` 的方式以及浏览器如何打开。

## 最小可运行代码

下面代码只读打印环境，不修改系统；把输出连同 `git rev-parse HEAD` 保存到实验记录中。

```python
from pathlib import Path
import platform
import random
import sys
import torch

seed = 2026
random.seed(seed)
torch.manual_seed(seed)
root = Path.cwd()
assert (root / "pyproject.toml").exists(), "请先 cd 到仓库根目录"
print({
    "python": sys.version.split()[0],
    "torch": torch.__version__,
    "platform": platform.platform(),
    "cwd": str(root),
    "seed": seed,
})
print(torch.rand(3))
```

## 反例与调试

最常见反例是 Notebook 能导入而终端不能：Jupyter kernel 指向旧环境。先在单元格打印 `sys.executable`，再与 `uv run python -c "import sys; print(sys.executable)"` 比较。第二类是从 `notebooks/` 子目录启动导致相对路径失效，解决方法是回到仓库根目录启动。第三类是直接 `pip install` 临时补包；这会让本机状态超出锁文件，短期“修好”却无法交付，应修改项目声明并重新锁定，或恢复 `uv sync --locked`。第四类是把 `.venv` 提交 Git；它不可移植且体积巨大，应删除并从锁重建。若 `doctor` 报错，先读第一条根因，不要被后续级联导入错误淹没。

## 主流工作与边界

当前 Python 项目通常用声明式依赖加锁文件，容器则进一步冻结系统库和运行时；但容器也不能自动冻结驱动、GPU 微码、远程数据或非确定算子。课程采用 `uv` 是为了统一 Windows/macOS/Linux 的入口与加速恢复，不意味着 `venv`、Conda 或容器无效。研究级复现还需记录数据许可、预处理代码、checkpoint、评测脚本和硬件拓扑。本周边界是单机 CPU 教程：目标是可重建和可诊断，不追求不同硬件逐 bit 相等。

## 对应 Notebook、互动图与 starter

先从 `learning/README.md` 确认学习顺序，再进入 `learning/labs/01_shapes_and_autograd.ipynb`。互动入口是 `learning/readings/interactive/index.html`；本周无填空 starter，验证入口是 `uv run llm-course doctor` 与 `uv run pytest -q`。阶段概览 `learning/readings/stages/01_foundations.md` 用作导航，本讲义才是第 1 周的唯一 lecture。

## 实验

实验一是在当前机器执行完整四步命令并记录环境快照。实验二是关闭终端后重新打开，不激活任何环境，直接用 `uv run` 重现同一随机张量。实验三是故意从错误目录运行一次，记录报错，再回到根目录修复。若有第二台 Windows 或 macOS 机器，只迁移 Git 仓库，不复制 `.venv`，用锁文件重建并比较诊断字段。

## 验收 rubric

合格：能从干净终端运行 doctor、测试和 Jupyter，并指出当前解释器路径。良好：复现记录包含 commit、lock、平台、版本、seed、命令和输出，且能定位一次 kernel 错配。优秀：能解释为何固定 seed 不等于跨平台逐位复现，能把“环境漂移、代码回归、数据漂移”设计成三个可区分的检查。任何把系统 Python 当项目环境、依赖靠口头描述或只提交截图的交付均不通过。

## 一手来源

- Python 官方关于隔离、可丢弃和不可移动虚拟环境的说明：https://docs.python.org/3/library/venv.html
- uv 官方项目锁文件与环境同步机制：https://docs.astral.sh/uv/concepts/projects/sync/
- uv 官方项目布局与 `uv.lock` 语义：https://docs.astral.sh/uv/concepts/projects/layout/
- PyTorch 官方随机性与可复现边界：https://docs.pytorch.org/docs/stable/notes/randomness.html
- Jupyter 官方 kernels 文档：https://jupyter-client.readthedocs.io/en/stable/kernels.html
