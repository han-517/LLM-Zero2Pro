# 环境搭建：Windows CPU 主线与免费 GPU 选修

## 先理解我们在搭什么

Python 环境像一个独立工具箱：课程把 Python 版本、库版本和命令都记录下来，避免“昨天能运行，今天突然坏了”。`uv.lock` 相当于工具箱的封条。

本课程选择：

- Windows 原生 + Python 3.12 + PyTorch CPU：完成全部必修内容。
- Colab/Kaggle Linux GPU：只用于可选的 Triton、FlashAttention 和稍大模型实验。
- WSL2：理解 Linux 或将来使用 vLLM 时再安装，不是开课前置条件。

## Windows 首次安装

在仓库根目录打开 PowerShell：

```powershell
uv --version
uv python install 3.12
uv sync
uv run llm-course doctor
```

成功时，`doctor` 会显示 Python、PyTorch、CPU 设备、随机种子检查和一个矩阵乘法结果。

若系统没有 `uv`：

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

关闭并重新打开 PowerShell 后再执行前面的四条命令。

## 常见问题

### `python` 命令不存在

这不影响课程。`uv run python` 会使用项目自己的 Python。不要把多个全局 Python 混在一起。

### PyTorch 下载很大

课程锁定 CPU wheel，不会下载 CUDA 工具链。检查 `pyproject.toml` 中的 `pytorch-cpu` 源是否保留。

### Notebook 找不到项目代码

必须从仓库根目录启动：

```powershell
uv run jupyter lab
```

### 如何彻底验证环境

```powershell
uv run llm-course course check
```

它会检查课程清单、论文目录并运行单元测试。

## 免费 GPU 选修路径

在 Colab 或 Kaggle 新建 Linux GPU Notebook，然后：

```bash
!git clone <你的仓库地址> llm-learning
%cd llm-learning
!python -m pip install -r requirements-hosted-gpu.txt
!python -m pip install -e . --no-deps
!python -m llm_course doctor
```

这里特意不运行 `uv sync`：主线 `pyproject.toml`/`uv.lock` 固定的是 CPU-only
PyTorch，直接同步会覆盖 Colab/Kaggle 预装的 CUDA PyTorch。上面的命令保留平台
自带 Torch，只安装课程其余依赖和本仓库代码。`doctor` 输出中必须同时看到
`cuda available: True` 和实际 GPU 名称；否则不要记录 GPU 性能结论。

免费实例随时可能回收，因此：

- 每次实验先用极小 batch 跑通正确性。
- checkpoint 和结果表及时下载，训练数据不要只放临时磁盘。
- 不把获得某种 GPU 型号作为课程验收条件。

## vLLM、Triton 与 Windows

vLLM 的正式运行环境是 Linux，Windows 原生不作为本课程路径。若未来本机增加 NVIDIA GPU，可使用 WSL2 或 Linux 容器；当前阶段只在免费 Linux GPU 上观察其 API 和性能概念。

## 可复现约定

- 所有实验显式设置随机种子。
- 必修 Notebook 默认设备是 `cpu`。
- 小模型先做“单个 batch 过拟合”，再扩大数据。
- 每个性能数字同时记录硬件、dtype、batch、序列长度和预热次数。

