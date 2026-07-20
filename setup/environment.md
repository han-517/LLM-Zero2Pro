# 环境搭建：Windows、macOS 与托管 GPU

课程使用 Python 3.12、`uv` 和 `uv.lock`。不需要 Anaconda，也不需要手动激活 `.venv`；从仓库根目录运行 `uv run ...` 即可。

## 支持矩阵

| 平台 | 支持级别 | 说明 |
|---|---|---|
| Windows 10/11 x64 | 完整主线 | PyTorch CPU、VS Code/JupyterLab、全部必修测试 |
| macOS 14+ Apple Silicon | 完整主线 | PyTorch CPU；MPS 仅作可选加速信息 |
| Intel Mac | 非锁定主线 | 当前 PyTorch 锁文件不保证提供 x86_64 macOS wheel |
| Linux x86_64/arm64 | 可运行 | 本地 CPU 或托管 GPU；GPU 不是必修条件 |

## Windows 安装

### 1. 安装 Git、uv 与 VS Code

PowerShell 中可使用 WinGet：

```powershell
winget install --id Git.Git -e
winget install --id astral-sh.uv -e
winget install --id Microsoft.VisualStudioCode -e
```

也可以使用 [Scoop 可选指南](windows_scoop.md)。安装后重开 PowerShell：

```powershell
git --version
uv --version
code --version
```

### 2. 克隆、同步并打开课程

```powershell
git clone https://github.com/han-517/LLM-Zero2Pro.git
Set-Location LLM-Zero2Pro
uv python install 3.12
uv sync
uv run llm-course doctor
uv run llm-course vscode
```

如果 `code` 不在 PATH，请重新运行 VS Code 安装器并勾选 **Add to PATH**。环境仍可通过 `uv run llm-course doctor` 验证。

## macOS Apple Silicon 安装

先检查架构：

```bash
uname -m
sw_vers -productVersion
```

`uname -m` 应输出 `arm64`。

### 1. 安装 Git、uv 与 VS Code

```bash
xcode-select --install
curl -LsSf https://astral.sh/uv/install.sh | sh
exec "$SHELL" -l
```

从 [VS Code 官方网站](https://code.visualstudio.com/) 安装应用。打开 VS Code，按 `Cmd+Shift+P`，运行 **Shell Command: Install 'code' command in PATH**。

### 2. 克隆、同步并打开课程

```bash
git clone https://github.com/han-517/LLM-Zero2Pro.git
cd LLM-Zero2Pro
uv python install 3.12
uv sync
uv run llm-course doctor
uv run llm-course vscode
```

`doctor` 可能显示 `mps_available: true`。必修验收仍以 CPU 为基线，MPS 不是继续学习的条件。

## Intel Mac

如果 `uname -m` 输出 `x86_64`，当前锁定的 PyTorch 版本可能没有兼容 wheel。推荐使用 Colab/Kaggle，或在受支持的 Windows/Linux/Apple Silicon 机器执行完整主线。不要通过删除 `uv.lock`、混装系统 Python 和旧版 Torch 来绕过错误。

## 验证安装

```text
uv run llm-course doctor
```

成功时应看到 Python 3.12、PyTorch 版本、正确的项目根目录和 `matmul_ok: true`。之后打开 [课程目录](../learning/README.md)。

在只显示 `learning/` 的学习 workspace 中，VS Code 的解释器和 Notebook Kernel 必须来自父级仓库：

- Windows：`..\.venv\Scripts\python.exe`
- macOS/Linux：`../.venv/bin/python`

## 托管 GPU 选修

Colab 或 Kaggle 中：

```bash
!git clone https://github.com/han-517/LLM-Zero2Pro.git
%cd LLM-Zero2Pro
!python -m pip install -r requirements-hosted-gpu.txt
!python -m pip install -e . --no-deps
!python -m llm_course doctor
```

这里不运行 `uv sync`，避免覆盖平台预装的 CUDA PyTorch。只有 `doctor` 显示实际 GPU 名称和 `cuda_available: true` 时才能记录 GPU 性能。

## 常见问题

- `ModuleNotFoundError`：回到含 `pyproject.toml` 的仓库根目录执行 `uv sync`。
- PyTorch 下载较慢：CPU wheel 仍然较大；保留锁文件和配置的 PyTorch CPU 源。
- 企业代理或证书错误：使用组织提供的代理和证书，不要关闭 TLS 校验。
- Kernel 不一致：关闭当前 Kernel，在 VS Code/JupyterLab 重新选择项目 `.venv`。
- 想使用 vLLM/Triton：它们主要面向 Linux GPU，是后期选修，不是 Windows/macOS 主线前置。

完整 VS Code 操作见 [VS Code 指南](vscode.md)，浏览器工作流见 [JupyterLab 可选指南](jupyterlab.md)。