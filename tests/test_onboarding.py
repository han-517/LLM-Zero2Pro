from __future__ import annotations

import json
from pathlib import Path

import pytest

import llm_course.lab as lab_module
import llm_course.vscode as vscode_module
from llm_course.cli import _build_parser
from llm_course.lab import LEARNING_ROOT, build_lab_command, launch_lab
from llm_course.vscode import CATALOG_PATH, WORKSPACE_PATH, build_vscode_command, launch_vscode

ROOT = Path(__file__).resolve().parents[1]


def test_markdown_catalog_is_the_only_48_lesson_entry() -> None:
    text = CATALOG_PATH.read_text(encoding="utf-8")
    assert CATALOG_PATH == ROOT / "learning" / "README.md"
    assert text.count("### 第 ") == 48
    assert "第 01 课" in text and "第 48 课" in text
    assert "讲义 → 补充阅读 → 互动图 → 实验二选一 → starter" in text
    assert "15 周" not in text and "15/48" not in text
    assert not (ROOT / "notebooks" / "00_START_HERE.ipynb").exists()


def test_lab_command_opens_learning_directory() -> None:
    command = build_lab_command(port=8899)
    assert command[1:3] == ["-m", "jupyterlab"]
    assert command[3] == "learning"
    assert LEARNING_ROOT == ROOT / "learning"
    assert "--port=8899" in command


@pytest.mark.parametrize("port", [0, 65_536])
def test_lab_command_rejects_bad_port(port: int) -> None:
    with pytest.raises(ValueError):
        build_lab_command(port=port)


def test_cli_supports_vscode_and_optional_lab() -> None:
    vscode_args = _build_parser().parse_args(["vscode"])
    lab_args = _build_parser().parse_args(["lab", "--no-browser", "--port", "8899"])
    assert vscode_args.command == "vscode"
    assert lab_args.command == "lab"
    assert lab_args.no_browser is True and lab_args.port == 8899


def test_vscode_command_opens_workspace_and_catalog() -> None:
    command = build_vscode_command("code")
    assert command[:2] == ["code", "--reuse-window"]
    assert Path(command[2]) == WORKSPACE_PATH
    assert Path(command[3]) == CATALOG_PATH
    workspace = json.loads(WORKSPACE_PATH.read_text(encoding="utf-8"))
    assert workspace["folders"][0]["path"] == "learning"


def test_launch_vscode_reports_missing_code(monkeypatch, capsys) -> None:
    monkeypatch.setattr(vscode_module.shutil, "which", lambda _name: None)
    assert launch_vscode() == 1
    assert "code" in capsys.readouterr().out


def test_launch_vscode(monkeypatch) -> None:
    class Completed:
        returncode = 0

    monkeypatch.setattr(vscode_module.shutil, "which", lambda _name: "code")
    monkeypatch.setattr(vscode_module.subprocess, "run", lambda *_a, **_k: Completed())
    assert launch_vscode() == 0


def test_launch_lab(monkeypatch) -> None:
    class Completed:
        returncode = 0

    monkeypatch.setattr(lab_module.importlib.util, "find_spec", lambda _name: object())
    monkeypatch.setattr(lab_module.subprocess, "run", lambda *_a, **_k: Completed())
    assert launch_lab() == 0


def test_cli_path_has_no_route_choice() -> None:
    args = _build_parser().parse_args(["course", "path"])
    assert args.command == "course" and args.course_command == "path"
    assert args.write is False and args.output is None
    assert _build_parser().parse_args(["course", "path", "--write"]).write is True
    with pytest.raises(SystemExit):
        _build_parser().parse_args(["course", "path", "--weeks", "15"])


def test_readme_defaults_to_vscode_and_supports_windows_and_macos() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "[learning/README.md](learning/README.md)" in readme
    assert "llm-course vscode" in readme
    assert "只显示 `learning/`" in readme
    assert "JupyterLab" in readme and "可选" in readme
    assert "Windows" in readme and "macOS" in readme
    assert "15 周" not in readme and "15/48" not in readme
