from __future__ import annotations

import importlib.util
import subprocess
import sys

from llm_course.paths import PROJECT_ROOT

LEARNING_ROOT = PROJECT_ROOT / "learning"


def build_lab_command(*, no_browser: bool = False, port: int | None = None) -> list[str]:
    if port is not None and not 1 <= port <= 65_535:
        raise ValueError("port 必须在 1..65535")
    command = [
        sys.executable,
        "-m",
        "jupyterlab",
        str(LEARNING_ROOT.relative_to(PROJECT_ROOT)),
    ]
    if no_browser:
        command.append("--no-browser")
    if port is not None:
        command.append(f"--port={port}")
    return command


def launch_lab(*, no_browser: bool = False, port: int | None = None) -> int:
    """从项目根目录启动可选 JupyterLab，并显示 learning/ 文件浏览器。"""

    if importlib.util.find_spec("jupyterlab") is None:
        print("错误: 当前环境没有 JupyterLab；请先运行 `uv sync`。")
        return 1
    if not LEARNING_ROOT.is_dir():
        print(f"错误: 缺少学习目录: {LEARNING_ROOT}")
        return 1
    try:
        command = build_lab_command(no_browser=no_browser, port=port)
    except ValueError as exc:
        print(f"错误: {exc}")
        return 2
    print("正在打开 learning/ 实验目录；终端保持运行是正常现象。")
    print("课程目录是 learning/README.md，实验位于 learning/labs/。")
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    return completed.returncode
