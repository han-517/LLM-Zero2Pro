from __future__ import annotations

from pathlib import Path

import nbformat
import pytest

import llm_course.lab as lab_module
from llm_course.cli import _build_parser
from llm_course.lab import WELCOME_NOTEBOOK, build_lab_command, launch_lab

ROOT = Path(__file__).resolve().parents[1]


def test_welcome_notebook_is_valid_and_has_clear_routes() -> None:
    notebook = nbformat.read(WELCOME_NOTEBOOK, as_version=4)
    text = "\n".join("".join(cell.source) for cell in notebook.cells)
    assert notebook.nbformat == 4
    for marker in (
        "从这里开始",
        "15 周核心路径",
        "00_shapes_and_autograd.ipynb",
        "exercises check 11",
        "NotImplementedError",
    ):
        assert marker in text
    assert any(cell.cell_type == "code" for cell in notebook.cells)


def test_lab_command_opens_welcome_from_project_root() -> None:
    command = build_lab_command(no_browser=True, port=8890)
    assert command[1:3] == ["-m", "jupyterlab"]
    assert command[3] == "notebooks\\00_START_HERE.ipynb" or command[3] == (
        "notebooks/00_START_HERE.ipynb"
    )
    assert "--no-browser" in command
    assert "--port=8890" in command


@pytest.mark.parametrize("port", [0, 65_536])
def test_lab_command_rejects_invalid_port(port: int) -> None:
    with pytest.raises(ValueError):
        build_lab_command(port=port)


def test_cli_parses_lab_options() -> None:
    args = _build_parser().parse_args(["lab", "--no-browser", "--port", "8890"])
    assert args.command == "lab"
    assert args.no_browser is True
    assert args.port == 8890


def test_launch_lab_runs_from_root(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class Completed:
        returncode = 0

    def fake_run(command, *, cwd, check):
        captured.update(command=command, cwd=cwd, check=check)
        return Completed()

    monkeypatch.setattr(lab_module.importlib.util, "find_spec", lambda _: object())
    monkeypatch.setattr(lab_module.subprocess, "run", fake_run)
    assert launch_lab() == 0
    assert captured["cwd"] == ROOT
    assert captured["check"] is False


def test_readme_and_environment_offer_windows_and_macos_paths() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    environment = (ROOT / "docs" / "00_environment.md").read_text(encoding="utf-8")
    notebook_guide = (ROOT / "notebooks" / "README.md").read_text(encoding="utf-8")
    for document in (readme, environment):
        assert "Windows" in document
        assert "macOS" in document
        assert "uv run llm-course lab" in document
    assert "00_START_HERE.ipynb" in readme
    assert "00_START_HERE.ipynb" in notebook_guide
    assert "Apple Silicon" in environment
    assert "Intel Mac" in environment
