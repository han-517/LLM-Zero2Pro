from __future__ import annotations

import importlib.util
import subprocess
import sys

from llm_course.paths import PROJECT_ROOT

WELCOME_NOTEBOOK = PROJECT_ROOT / "notebooks" / "00_START_HERE.ipynb"


def build_lab_command(*, no_browser: bool = False, port: int | None = None) -> list[str]:
    if port is not None and not 1 <= port <= 65_535:
        raise ValueError("port 必须在 1..65535")
    command = [
        sys.executable,
        "-m",
        "jupyterlab",
        str(WELCOME_NOTEBOOK.relative_to(PROJECT_ROOT)),
    ]
    if no_browser:
        command.append("--no-browser")
    if port is not None:
        command.append(f"--port={port}")
    return command


def launch_lab(*, no_browser: bool = False, port: int | None = None) -> int:
    """从项目根目录启动 JupyterLab，并直接打开课程欢迎 Notebook。"""

    if importlib.util.find_spec("jupyterlab") is None:
        print("错误: 当前环境没有 JupyterLab；请先运行 `uv sync`。")
        return 1
    if not WELCOME_NOTEBOOK.is_file():
        print(f"错误: 缺少 Jupyter 欢迎页: {WELCOME_NOTEBOOK}")
        return 1
    try:
        command = build_lab_command(no_browser=no_browser, port=port)
    except ValueError as exc:
        print(f"错误: {exc}")
        return 2
    print("正在从仓库根目录启动 JupyterLab……")
    print("默认打开 notebooks/00_START_HERE.ipynb；终端保持运行是正常现象。")
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    return completed.returncode
