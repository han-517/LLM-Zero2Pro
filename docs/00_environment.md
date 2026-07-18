# 环境搭建：Windows、macOS 与托管 GPU

本课程使用 Python 3.12、`uv` 和锁文件管理依赖。`uv` 负责下载合适的 Python、创建 `.venv` 并按 `uv.lock` 安装库，因此不需要先安装 Anaconda，也不建议把课程依赖装进系统 Python。

## 支持矩阵

| 平台 | 支持级别 | 说明 |
|---|---|---|
| Windows 10/11 x64 | 必修主线 | PyTorch CPU，完整执行全部必修 Notebook 和测试 |
| macOS 14+ Apple Silicon | 必修主线 | 当前锁文件包含 arm64 PyTorch wheel；必修仍以 CPU 为基线，可检测 MPS |
| Intel Mac | 不在当前锁定主线 | 最新锁定 PyTorch 没有 x86_64 macOS wheel，建议使用 Colab/Kaggle |
| Linux x86_64 / arm64 | 可运行 | 本地 CPU 可用，课程主要把 Linux 用于托管 GPU 选修 |

`uv` 官方对 Apple Silicon、Intel macOS、Windows x64 和 Linux x64 提供 Tier 1 支持，但课程还依赖 PyTorch wheel；因此操作系统支持不等于当前课程锁文件覆盖所有硬件。参考 [uv 平台支持](https://docs.astral.sh/uv/reference/policies/platforms/)和 [PyTorch 本地安装说明](https://docs.pytorch.org/get-started/locally/)。

## 安装完成后的共同命令

无论 Windows 还是受支持的 macOS，进入仓库根目录后都运行：

```text
uv python install 3.12
uv sync
uv run llm-course doctor
uv run llm-course lab
```

成功标准：

- `doctor` 显示 Python `3.12.x`、PyTorch 版本和 `matmul_ok: True`。
- `project_root` 指向当前克隆的 `LLM-Zero2Pro`。
- JupyterLab 自动打开 `notebooks/00_START_HERE.ipynb`。
- 欢迎页第一个代码单元输出 `✅ 环境入口正常`。

## Windows 安装

### 1. 安装 Git 和 uv

Windows 10/11 可以在 PowerShell 使用 WinGet：

```powershell
winget install --id Git.Git -e
winget install --id astral-sh.uv -e
```

也可以使用 Scoop，详见 [Scoop 可选工具教程](00_scoop.md)。若只缺少 `uv`，官方安装器是：

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

关闭并重新打开 PowerShell，然后验证：

```powershell
git --version
uv --version
```

### 2. 克隆并同步课程

```powershell
git clone https://github.com/han-517/LLM-Zero2Pro.git
Set-Location LLM-Zero2Pro
uv python install 3.12
uv sync
uv run llm-course doctor
uv run llm-course lab
```

不需要执行 `.venv\Scripts\activate`。在仓库根目录使用 `uv run ...`，uv 会自动选择项目环境。

## macOS Apple Silicon 安装

当前本机主线面向 macOS 14+ 的 Apple Silicon（M1/M2/M3/M4 及后续 arm64 机型）。先在 Terminal 检查架构：

```bash
uname -m
sw_vers -productVersion
```

第一条应输出 `arm64`。如果输出 `x86_64`，请先阅读后面的 Intel Mac 说明。

### 1. 安装 Git

macOS 可以通过 Xcode Command Line Tools 安装 Apple 提供的 Git：

```bash
xcode-select --install
```

安装窗口完成后验证：

```bash
git --version
```

如果你已经使用 Homebrew，也可以执行 `brew install git`。两种方式都列在 [Git 官方 macOS 安装页](https://git-scm.com/install/mac)。

### 2. 安装 uv

使用 [uv 官方安装器](https://docs.astral.sh/uv/getting-started/installation/)：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
exec "$SHELL" -l
uv --version
```

也可以使用 Homebrew：`brew install uv`。不要同时用多个包管理器重复安装。

### 3. 克隆并启动课程

```bash
git clone https://github.com/han-517/LLM-Zero2Pro.git
cd LLM-Zero2Pro
uv python install 3.12
uv sync
uv run llm-course doctor
uv run llm-course lab
```

`doctor` 会显示 `mps_available`。课程必修仍默认使用 CPU，以保证 Windows/macOS 结果可比较；MPS 是本地加速信息，不是完成课程的条件。

## Intel Mac 怎么办

运行 `uname -m` 输出 `x86_64` 表示 Intel Mac。当前 `uv.lock` 选择的最新 PyTorch 版本只包含 Apple Silicon macOS wheel，因此直接 `uv sync` 可能报告找不到兼容分发包。

推荐选择之一：

1. 使用 Colab/Kaggle 跑 Notebook，把本地仓库用于阅读和填写纯 Python 练习。
2. 在支持的 Linux/Windows 机器上执行完整 CPU 主线。
3. 如果必须使用 Intel Mac，自行建立兼容旧版 PyTorch 的独立环境；它不再受本课程锁文件和测试基线保证。

不要在 `uv sync` 失败后随意删除 `uv.lock` 或混装不同架构的 Python/Torch，这会让问题更难复现。

## 正确启动 JupyterLab

推荐命令：

```text
uv run llm-course lab
```

它做两件事：从仓库根目录启动 JupyterLab，并直接打开 `notebooks/00_START_HERE.ipynb`。

其他场景：

```text
# 端口被占用
uv run llm-course lab --port 8890

# SSH/远程服务器，不自动打开浏览器
uv run llm-course lab --no-browser
```

普通的 `uv run jupyter lab` 仍然可用，但不会保证打开课程欢迎页。

## Jupyter 常见问题

### `ModuleNotFoundError: llm_course`

通常是从错误目录启动或使用了系统 Jupyter。关闭服务，回到包含 `pyproject.toml` 的仓库根目录：

```text
uv sync
uv run llm-course lab
```

### Kernel 显示的 Python 不是 3.12

在 Notebook 运行：

```python
import sys
print(sys.executable)
print(sys.version)
```

路径应位于当前仓库的 `.venv`，版本应为 3.12。若不一致，关闭所有 Jupyter 服务后使用课程命令重启，不要在 Lab 中临时安装另一份 Python。

### 浏览器没有自动打开

终端会打印形如 `http://localhost:8888/lab?...token=...` 的地址。复制完整地址到浏览器即可。不要把带 token 的地址发送给他人。

### PyTorch 下载很大或很慢

CPU wheel 仍然较大。保持 `pyproject.toml` 中的 `pytorch-cpu` 源和 `uv.lock`，让 uv 继续下载；不要同时运行 pip/conda 安装另一份 Torch。

### 企业代理或证书错误

先运行 `uv sync -v` 查看失败域名。企业代理需要使用组织提供的证书/代理配置；不要关闭 TLS 校验。uv 在需要系统证书时提供 `--system-certs` 选项，具体用法见其官方 CLI 文档。

## 如何验证整个仓库

```text
uv run llm-course course check
```

它会校验课程清单、论文目录、学习入口、代码模板、参考实现和必修 Notebook。第一次运行比单个 Notebook 慢是正常现象。

## 免费托管 GPU 选修

在 Colab 或 Kaggle 新建 Linux GPU Notebook：

```bash
!git clone https://github.com/han-517/LLM-Zero2Pro.git
%cd LLM-Zero2Pro
!python -m pip install -r requirements-hosted-gpu.txt
!python -m pip install -e . --no-deps
!python -m llm_course doctor
```

这里不运行 `uv sync`：项目锁文件固定 CPU 路线，直接同步可能覆盖平台预装的 CUDA PyTorch。`doctor` 必须显示 `cuda_available: True` 和实际 GPU 名称，否则不要记录 GPU 性能结论。

免费实例可能回收：先用极小 batch 跑正确性，及时下载 checkpoint，并记录 GPU 型号、dtype、batch、序列长度和预热次数。

## vLLM、Triton 与本机系统

vLLM 和 Triton 的课程实验以 Linux GPU 为目标。Windows 原生和 macOS 不是这些工具的主运行路径；Windows 可在后期使用 WSL2/Linux 容器，macOS 学习其算法与接口概念即可。它们都不是开课前置条件。

## 可复现约定

- 必修 Notebook 默认设备是 `cpu`。
- 所有实验显式设置随机种子。
- 小模型先做单个 batch 过拟合，再扩大数据。
- 每个性能数字同时记录硬件、dtype、batch、序列长度和预热次数。
- 不手工修改 `.venv` 内文件；依赖变更统一通过 `pyproject.toml`、`uv.lock` 和 uv 完成。
