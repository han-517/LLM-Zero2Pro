from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

from llm_course.paths import PROJECT_ROOT
from llm_course.schemas import ValidationReport


@dataclass(frozen=True)
class Exercise:
    exercise_id: str
    slug: str
    weeks: tuple[int, ...]
    title: str
    template: str
    check: str
    optional: bool = False

    @property
    def week_label(self) -> str:
        return ",".join(str(week) for week in self.weeks) or "选修"


MANIFEST_PATH = PROJECT_ROOT / "course" / "exercises.yaml"


def load_exercises(path: Path = MANIFEST_PATH) -> tuple[Exercise, ...]:
    """从唯一清单加载练习；README、CLI 与课程核查不得再维护另一份注册表。"""

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict) or not isinstance(data.get("exercises"), list):
        raise ValueError("exercise manifest 必须包含 exercises 列表")

    exercises: list[Exercise] = []
    for item in data["exercises"]:
        if not isinstance(item, dict):
            raise ValueError("exercise manifest 中的每条记录必须是 mapping")
        exercises.append(
            Exercise(
                exercise_id=str(item["id"]),
                slug=str(item["slug"]),
                weeks=tuple(int(week) for week in item.get("weeks", ())),
                title=str(item["title"]),
                template=str(item["template"]),
                check=str(item["check"]),
                optional=bool(item.get("optional", False)),
            )
        )
    return tuple(exercises)


EXERCISES = load_exercises()


def select_exercises(identifier: str) -> tuple[Exercise, ...]:
    normalized = identifier.strip().lower().replace("_", "-")
    if normalized == "all":
        return EXERCISES
    for exercise in EXERCISES:
        aliases = {
            exercise.exercise_id,
            exercise.slug,
            Path(exercise.template).stem.lower().replace("_", "-"),
        }
        if normalized in aliases:
            return (exercise,)
    choices = ", ".join(exercise.exercise_id for exercise in EXERCISES)
    raise ValueError(f"未知练习 {identifier!r}；可用编号为 {choices}，或使用 all")


def _template_state(exercise: Exercise) -> str:
    path = PROJECT_ROOT / exercise.template
    try:
        source = path.read_text(encoding="utf-8")
    except OSError:
        return "缺失"
    return "待填写" if "raise NotImplementedError" in source else "已填写，待核查"


def print_exercises() -> int:
    print("编号  周次       状态             核心代码模板")
    for exercise in EXERCISES:
        print(
            f"{exercise.exercise_id:<4}  {exercise.week_label:<9}  "
            f"{_template_state(exercise):<15}  {exercise.title} ({exercise.slug})"
        )
    print("\n核查示例: uv run llm-course exercises check 07")
    return 0


def run_exercise_checks(identifier: str = "all") -> int:
    try:
        selected = select_exercises(identifier)
    except ValueError as exc:
        print(f"错误: {exc}")
        return 2
    paths = [str(PROJECT_ROOT / exercise.check) for exercise in selected]
    names = ", ".join(exercise.exercise_id for exercise in selected)
    print(f"开始核查练习: {names}")
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *paths],
        cwd=PROJECT_ROOT,
        check=False,
    )
    if completed.returncode == 0:
        print("练习核查通过。请再口述张量形状、边界条件和复杂度。")
    else:
        print("练习尚未通过；从第一条失败信息和最小反例开始修正。")
    return completed.returncode


def validate_exercise_assets() -> ValidationReport:
    report = ValidationReport()
    seen_ids: set[str] = set()
    seen_slugs: set[str] = set()
    previous_first_week = 0
    for exercise in EXERCISES:
        if not exercise.exercise_id.isdigit() or len(exercise.exercise_id) != 2:
            report.errors.append(f"练习编号必须是两位数字: {exercise.exercise_id}")
        if exercise.exercise_id in seen_ids:
            report.errors.append(f"练习编号重复: {exercise.exercise_id}")
        if exercise.slug in seen_slugs:
            report.errors.append(f"练习别名重复: {exercise.slug}")
        seen_ids.add(exercise.exercise_id)
        seen_slugs.add(exercise.slug)

        if not exercise.optional and (
            not exercise.weeks or any(not 1 <= week <= 48 for week in exercise.weeks)
        ):
            report.errors.append(f"必修练习周次无效: {exercise.exercise_id}: {exercise.weeks}")
        if tuple(sorted(set(exercise.weeks))) != exercise.weeks:
            report.errors.append(f"练习周次必须升序且不重复: {exercise.exercise_id}")
        if exercise.weeks:
            if exercise.weeks[0] < previous_first_week:
                report.errors.append("exercise manifest 必须按首次出现周次排序")
            previous_first_week = exercise.weeks[0]

        for relative_path in (exercise.template, exercise.check):
            path = PROJECT_ROOT / relative_path
            if not path.is_file():
                report.errors.append(f"缺少练习资产: {relative_path}")
                continue
            try:
                compile(path.read_text(encoding="utf-8"), str(path), "exec")
            except (OSError, SyntaxError) as exc:
                report.errors.append(f"练习资产无法编译: {relative_path}: {exc}")
    return report
