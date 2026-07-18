from __future__ import annotations

from pathlib import Path

import llm_course.exercises as exercise_module
from llm_course.cli import _build_parser
from llm_course.exercises import EXERCISES, select_exercises, validate_exercise_assets

ROOT = Path(__file__).resolve().parents[1]


def test_exercise_catalog_covers_ten_core_templates() -> None:
    assert [exercise.exercise_id for exercise in EXERCISES] == [
        f"{index:02d}" for index in range(1, 11)
    ]
    assert {exercise.slug for exercise in EXERCISES} >= {
        "softmax",
        "bpe",
        "attention",
        "rope",
        "gqa",
        "decoder",
        "kv-cache",
        "moe-capacity",
        "moe-router",
        "sft",
    }
    assert validate_exercise_assets().ok


def test_templates_keep_core_implementation_blank() -> None:
    for exercise in EXERCISES:
        source = (ROOT / exercise.template).read_text(encoding="utf-8")
        assert "TODO" in source
        assert "raise NotImplementedError" in source


def test_select_exercises_accepts_ids_aliases_and_all() -> None:
    assert select_exercises("07")[0].slug == "rope"
    assert select_exercises("rope")[0].exercise_id == "07"
    assert select_exercises("08_grouped_query_attention")[0].slug == "gqa"
    assert select_exercises("all") == EXERCISES


def test_cli_parses_single_and_all_exercise_checks() -> None:
    parser = _build_parser()
    single = parser.parse_args(["exercises", "check", "07"])
    all_exercises = parser.parse_args(["exercises", "check"])
    assert single.exercise == "07"
    assert all_exercises.exercise == "all"


def test_runner_invokes_only_the_selected_public_check(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class Completed:
        returncode = 0

    def fake_run(command, *, cwd, check):
        captured.update(command=command, cwd=cwd, check=check)
        return Completed()

    monkeypatch.setattr(exercise_module.subprocess, "run", fake_run)
    assert exercise_module.run_exercise_checks("rope") == 0
    command = captured["command"]
    assert isinstance(command, list)
    assert command[-1].endswith("test_07_rope.py")
    assert captured["cwd"] == ROOT
    assert captured["check"] is False
