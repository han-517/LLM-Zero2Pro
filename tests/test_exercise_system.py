from __future__ import annotations

import importlib.util
from pathlib import Path

import llm_course.exercises as exercise_module
from llm_course.cli import _build_parser
from llm_course.exercises import EXERCISES, select_exercises, validate_exercise_assets

ROOT = Path(__file__).resolve().parents[1]

STABLE_CORE = {
    "01": ("softmax", (4,)),
    "02": ("attention", (12, 13)),
    "03": ("kv-cache", (20,)),
    "04": ("sft", (40,)),
    "05": ("moe-capacity", (36,)),
    "06": ("bpe", (10,)),
    "07": ("rope", (18,)),
    "08": ("gqa", (19,)),
    "09": ("decoder", (16, 17)),
    "10": ("moe-router", (35,)),
}


def test_exercise_manifest_has_twenty_required_and_one_optional_template() -> None:
    assert len(EXERCISES) == 21
    assert {exercise.exercise_id for exercise in EXERCISES} == {
        f"{index:02d}" for index in range(1, 22)
    }
    required = [exercise for exercise in EXERCISES if not exercise.optional]
    optional = [exercise for exercise in EXERCISES if exercise.optional]
    assert len(required) == 20
    assert [(exercise.exercise_id, exercise.slug, exercise.weeks) for exercise in optional] == [
        ("21", "multimodal-bridge", ())
    ]
    assert validate_exercise_assets().ok


def test_stable_ids_keep_their_topics_and_real_roadmap_weeks() -> None:
    by_id = {exercise.exercise_id: exercise for exercise in EXERCISES}
    assert set(STABLE_CORE) <= by_id.keys()
    for exercise_id, (slug, weeks) in STABLE_CORE.items():
        exercise = by_id[exercise_id]
        assert exercise.slug == slug
        assert exercise.weeks == weeks


def test_exercises_are_sorted_by_first_real_week_with_optional_last() -> None:
    expected = sorted(
        EXERCISES,
        key=lambda exercise: (
            exercise.optional,
            exercise.weeks[0] if exercise.weeks else 49,
        ),
    )
    assert list(EXERCISES) == expected


def test_templates_keep_core_implementation_blank() -> None:
    for exercise in EXERCISES:
        source = (ROOT / exercise.template).read_text(encoding="utf-8")
        assert "TODO" in source
        assert "raise NotImplementedError" in source


def test_public_loader_uses_the_unique_learning_starter_directory() -> None:
    loader_path = ROOT / "checks" / "exercises" / "_loader.py"
    spec = importlib.util.spec_from_file_location("exercise_loader_test", loader_path)
    assert spec is not None and spec.loader is not None
    loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader)
    module = loader.load_starter("20_inference_systems.py")
    assert module.__file__ is not None
    assert Path(module.__file__).parent == ROOT / "learning" / "labs" / "starter"


def test_select_exercises_accepts_ids_aliases_optional_and_all() -> None:
    assert select_exercises("07")[0].slug == "rope"
    assert select_exercises("rope")[0].exercise_id == "07"
    assert select_exercises("08_grouped_query_attention")[0].slug == "gqa"
    assert select_exercises("multimodal_bridge")[0].optional
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
