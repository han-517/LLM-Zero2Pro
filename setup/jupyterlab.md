# 可选：使用 JupyterLab

VS Code 是默认入口；只有偏好浏览器 Notebook 界面时才需要 JupyterLab。

在仓库根目录运行：

```text
uv sync
uv run llm-course lab
```

命令会把 JupyterLab 打开在 `learning/` 文件浏览器，不会自动跳转到欢迎 Notebook。先打开 `README.md` 查看当前课，再进入 `labs/` 选择实验。

常用参数：

```text
uv run llm-course lab --port 8890
uv run llm-course lab --no-browser
```

终端必须保持运行，退出时回到终端按 `Ctrl+C`。Kernel 应指向当前仓库 `.venv` 中的 Python 3.12；不要在 Notebook 中临时安装另一套依赖。