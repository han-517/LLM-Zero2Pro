# Scoop 可选安装路径

你可以用 Scoop 管理系统级工具；课程 Python 包仍由 `uv` 和 `uv.lock` 管理。

```powershell
scoop install git uv
# 只有需要本机渲染论文 PDF 时才需要
scoop install poppler
```

随后在仓库根目录执行：

```powershell
uv python install 3.12
uv sync
uv run llm-course doctor
```

不建议同时用 Scoop、Microsoft Store 和 Conda 安装三套 Python。课程命令统一写成 `uv run ...`，可绕开全局 PATH 混乱。

