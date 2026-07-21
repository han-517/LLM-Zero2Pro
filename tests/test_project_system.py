from __future__ import annotations

from pathlib import Path

import llm_course.projects as project_module
from llm_course.cli import _build_parser
from llm_course.course import render_learning_path
from llm_course.projects import PROJECTS, run_project_checks, validate_project_assets

ROOT = Path(__file__).resolve().parents[1]


def test_manifest_defines_five_integrated_projects() -> None:
    assert [project.project_id for project in PROJECTS] == [
        "01",
        "02",
        "03",
        "04",
        "05",
    ]
    assert PROJECTS[0].slug == "end-to-end-lm"
    assert PROJECTS[0].available
    assert all(not project.available for project in PROJECTS[1:])
    assert validate_project_assets().ok


def test_available_project_is_learner_owned_and_keeps_core_work_blank() -> None:
    student_dir = ROOT / PROJECTS[0].directory / "student_lm"
    sources = [path.read_text(encoding="utf-8") for path in student_dir.glob("*.py")]
    assert sources
    assert any("raise NotImplementedError" in source for source in sources)
    assert all("llm_from_scratch" not in source for source in sources)


def test_cli_parses_project_list_and_check() -> None:
    parser = _build_parser()
    listing = parser.parse_args(["projects", "list"])
    checking = parser.parse_args(["projects", "check", "end-to-end-lm"])
    assert listing.projects_command == "list"
    assert checking.projects_command == "check"
    assert checking.project == "end-to-end-lm"


def test_runner_invokes_all_public_checks_for_available_project(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class Completed:
        returncode = 0

    def fake_run(command, *, cwd, check):
        captured.update(command=command, cwd=cwd, check=check)
        return Completed()

    monkeypatch.setattr(project_module.subprocess, "run", fake_run)
    assert run_project_checks("01") == 0
    command = captured["command"]
    assert isinstance(command, list)
    assert command[:3] == [project_module.sys.executable, "-m", "pytest"]
    assert command[-3:] == [str(ROOT / path) for path in PROJECTS[0].checks]
    assert captured["cwd"] == ROOT
    assert captured["check"] is False


def test_planned_project_does_not_start_pytest(monkeypatch) -> None:
    def unexpected_run(*_args, **_kwargs):
        raise AssertionError("planned projects must not invoke pytest")

    monkeypatch.setattr(project_module.subprocess, "run", unexpected_run)
    assert run_project_checks("02") == 2


def test_generated_catalog_explains_integrated_projects() -> None:
    catalog = render_learning_path()
    assert "## 五个贯穿式大作业" in catalog
    assert "从字节 BPE 到可恢复训练的完整语言模型" in catalog
    assert "uv run llm-course projects check 01" in catalog
