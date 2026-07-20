from __future__ import annotations

import shutil
import subprocess

from llm_course.paths import PROJECT_ROOT

CATALOG_PATH = PROJECT_ROOT / "learning" / "README.md"
WORKSPACE_PATH = PROJECT_ROOT / "LLM-Zero2Pro.code-workspace"


def build_vscode_command(code_executable: str) -> list[str]:
    """构造 VS Code 命令，不启动 Jupyter 服务。"""

    return [
        code_executable,
        "--reuse-window",
        str(WORKSPACE_PATH),
        str(CATALOG_PATH),
    ]


def launch_vscode() -> int:
    """打开项目 workspace 和唯一课程目录。"""

    code_executable = shutil.which("code")
    if code_executable is None:
        print("错误: PATH 中找不到 `code` 命令。")
        print("可先在 VS Code 中安装 code shell command，或手动执行 `code .`。")
        print(f"然后打开课程目录: {CATALOG_PATH}")
        return 1
    if not CATALOG_PATH.is_file() or not WORKSPACE_PATH.is_file():
        print("错误: 缺少 learning/README.md 或 VS Code workspace 文件。")
        return 1
    completed = subprocess.run(
        build_vscode_command(code_executable), cwd=PROJECT_ROOT, check=False
    )
    return completed.returncode