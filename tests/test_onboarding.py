from __future__ import annotations

from pathlib import Path

import nbformat
import pytest

import llm_course.lab as lab_module
from llm_course.cli import _build_parser
from llm_course.lab import WELCOME_NOTEBOOK, build_lab_command, launch_lab

ROOT = Path(__file__).resolve().parents[1]


def test_start_page_uses_one_learning_path() -> None:
    notebook = nbformat.read(WELCOME_NOTEBOOK, as_version=4)
    text = "\n".join(str(cell.source) for cell in notebook.cells)
    assert notebook.nbformat == 4
    assert "1–48" in text
    assert "../docs/learning_path.md" in text
    assert "15 周" not in text
    assert "15/48" not in text
    assert any(cell.cell_type == "code" for cell in notebook.cells)


def test_lab_command_opens_start_page() -> None:
    command = build_lab_command(port=8899)
    assert command[1:3] == ["-m", "jupyterlab"]
    assert command[3].endswith("00_START_HERE.ipynb")
    assert "--port=8899" in command


@pytest.mark.parametrize("port", [0, 65_536])
def test_lab_command_rejects_bad_port(port: int) -> None:
    with pytest.raises(ValueError):
        build_lab_command(port=port)


def test_cli_lab_arguments() -> None:
    args = _build_parser().parse_args(["lab", "--no-browser", "--port", "8899"])
    assert args.command == "lab"
    assert args.no_browser is True
    assert args.port == 8899


def test_cli_path_has_no_week_choice() -> None:
    args = _build_parser().parse_args(["course", "path"])
    assert args.command == "course"
    assert args.course_command == "path"
    assert args.write is False
    assert args.output is None
    write_args = _build_parser().parse_args(["course", "path", "--write"])
    assert write_args.write is True
    with pytest.raises(SystemExit):
        _build_parser().parse_args(["course", "path", "--weeks", "15"])


def test_launch_lab(monkeypatch) -> None:
    class Completed:
        returncode = 0

    monkeypatch.setattr(lab_module.importlib.util, "find_spec", lambda _name: object())
    monkeypatch.setattr(lab_module.subprocess, "run", lambda *_a, **_k: Completed())
    assert launch_lab() == 0


def test_readme_uses_one_path_and_supports_windows_and_macos() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "(docs/learning_path.md)" in readme
    assert "15 周" not in readme
    assert "15/48" not in readme
    assert "Windows" in readme
    assert "macOS" in readme
