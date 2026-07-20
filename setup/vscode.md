# 在 VS Code 中学习

VS Code 是本课程默认入口，不需要启动 JupyterLab。课程目录是 [`learning/README.md`](../learning/README.md)。

## 第一次打开

在仓库根目录运行：

```text
uv python install 3.12
uv sync
uv run llm-course doctor
uv run llm-course vscode
```

最后一条命令会打开 `LLM-Zero2Pro.code-workspace` 和课程目录。这个 workspace 的 Explorer **只显示 `learning/`**；根目录中的环境、检查、答案和维护代码不会混入学习空间。

手动打开同一学习 workspace：

```text
code LLM-Zero2Pro.code-workspace learning/README.md
```

只需要最简文件夹视图时也可以运行 `code learning`，但这种方式不会载入课程核查任务。只有维护课程仓库时才使用 `code .`。

如果终端提示找不到 `code`：

- Windows：在 VS Code 安装器中启用 **Add to PATH**，或重新安装后勾选该选项。
- macOS：在 VS Code 命令面板运行 **Shell Command: Install 'code' command in PATH**，再重开 Terminal。

## 选择项目解释器

按 `Ctrl+Shift+P`（macOS 为 `Cmd+Shift+P`），运行 **Python: Select Interpreter**：

- Windows：选择 `..\.venv\Scripts\python.exe`。
- macOS/Linux：选择 `../.venv/bin/python`。

不要选择系统 Python、Conda base 或其他项目的环境。打开 `.ipynb` 后，右上角 Kernel 也要选择同一个 `.venv`。

## 阅读 Markdown

1. 打开 `learning/README.md`。
2. 按 `Ctrl+Shift+V`；macOS 使用 `Cmd+Shift+V` 打开 Markdown 预览。
3. 保持目录预览在一侧，在另一侧打开本课讲义或实验。
4. 互动 HTML 右键选择 **Open with Live Server**，或在文件管理器中用浏览器打开。

## 选择实验格式

每个实验同时提供两种格式，只需选择一种：

- `.ipynb`：使用 VS Code Notebook 编辑器，适合图表、Markdown 与逐格执行。
- `.py`：使用 `# %%` 单元和 **Run Cell**，适合代码审查、Git diff 和普通 Python 编辑。

不要在同一课来回修改两个版本。课程维护时以 `.ipynb` 为源，通过 `uv run python scripts/sync_labs.py` 同步 `.py`。

## 运行 starter 与核查

starter 位于 `learning/labs/starter/`。填写当前课指定的空缺后，在集成终端运行：

```text
uv run llm-course exercises list
uv run llm-course exercises check 07
```

也可以运行 **Terminal → Run Task**，选择“课程：核查当前 starter”，输入练习编号或别名。
workspace 会自动让这些任务在父级仓库根目录执行，因此 Explorer 无需显示 `checks/`、`course/` 或 `src/`。

## 常见问题

- `.ipynb` 没有 Run 按钮：安装工作区推荐的 Python 和 Jupyter 扩展。
- Kernel 一直启动失败：确认解释器来自当前 `.venv`，执行 `uv sync` 后重载窗口。
- `ModuleNotFoundError`：优先使用 `uv run llm-course vscode` 打开学习 workspace；不要把 `learning/labs/` 单独当成工程。
- Markdown 链接打开源码：HTML 互动图必须使用浏览器或 Live Preview，GitHub/Markdown 预览不会执行脚本。