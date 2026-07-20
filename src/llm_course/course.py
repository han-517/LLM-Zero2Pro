from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from llm_course.exercises import EXERCISES, validate_exercise_assets
from llm_course.lectures import validate_weekly_lectures
from llm_course.paths import PROJECT_ROOT, ROADMAP_PATH
from llm_course.schemas import LessonManifest, ValidationReport

LESSON_FIELDS = {
    "week",
    "stage",
    "title",
    "objectives",
    "readings",
    "experiments",
    "exercise",
    "acceptance",
}
ASSET_FIELDS = {"lecture", "notebooks", "exercises", "sources", "mode", "deliverable"}
NOTEBOOK_CONTRACT_FIELDS = {
    "weeks",
    "estimated_minutes",
    "prerequisites",
    "starter_ids",
    "completion_assertion",
}
REQUIRED_COURSE_ASSETS = (
    "docs/00_environment.md",
    "notebooks/00_START_HERE.ipynb",
    "notebooks/README.md",
    "docs/learning_path.md",
    "docs/README.md",
    "docs/weeks/README.md",
    "docs/code_templates.md",
    "docs/interactive/foundations-lab.html",
    "docs/interactive/index.html",
    "docs/interactive/core-concepts.html",
    "docs/interactive/architecture-evolution.html",
    "docs/interactive/architecture-lab.html",
    "docs/interactive/training-and-alignment.html",
    "docs/interactive/serving-lab.html",
    "docs/interactive/multimodal-flow.html",
    "docs/extensions/multimodal.md",
    "exercises/manifest.yaml",
    "exercises/starter/README.md",
    "requirements-hosted-gpu.txt",
)
LEARNING_PATH_DOCUMENT = PROJECT_ROOT / "docs" / "learning_path.md"


def load_roadmap(path: Path = ROADMAP_PATH) -> dict[str, Any]:
    """加载课程清单，并兼容旧版单文件结构。"""

    def read_mapping(candidate: Path) -> dict[str, Any]:
        with candidate.open("r", encoding="utf-8") as handle:
            value = yaml.safe_load(handle)
        if not isinstance(value, dict):
            raise ValueError(f"{candidate} 根节点必须是 mapping")
        return value

    data = read_mapping(path)
    if "stage_files" not in data:
        return data

    base_dir = path.parent.resolve()

    def resolve_local(relative_path: object) -> Path:
        if not isinstance(relative_path, str) or not relative_path:
            raise ValueError("课程入口中的文件路径必须是非空字符串")
        candidate = (base_dir / relative_path).resolve()
        if not candidate.is_relative_to(base_dir):
            raise ValueError(f"课程入口不得引用 course/ 之外的文件: {relative_path}")
        return candidate

    metadata = read_mapping(resolve_local(data.get("course_file")))
    stage_files = data.get("stage_files")
    if not isinstance(stage_files, list) or not stage_files:
        raise ValueError("roadmap.stage_files 必须是非空列表")

    stages: list[dict[str, Any]] = []
    weeks: list[dict[str, Any]] = []
    assets: dict[int, dict[str, Any]] = {}
    for relative_path in stage_files:
        stage_data = read_mapping(resolve_local(relative_path))
        stage = stage_data.get("stage")
        lessons = stage_data.get("lessons")
        if not isinstance(stage, dict) or not isinstance(lessons, list):
            raise ValueError(f"阶段文件 {relative_path} 必须包含 stage 和 lessons")
        stages.append(stage)
        for lesson_with_assets in lessons:
            if not isinstance(lesson_with_assets, dict):
                raise ValueError(f"阶段文件 {relative_path} 的 lesson 必须是 mapping")
            lesson = dict(lesson_with_assets)
            asset = lesson.pop("assets", None)
            week = int(lesson["week"])
            if week in assets or not isinstance(asset, dict):
                raise ValueError(f"week {week} 重复或缺少 assets")
            weeks.append(lesson)
            assets[week] = asset

    return {**metadata, "stages": stages, "weeks": weeks, "assets": assets}


def _stage_week_set(stage: dict[str, Any]) -> set[int]:
    raw = str(stage.get("weeks", ""))
    try:
        start_text, end_text = raw.split("-", maxsplit=1)
        start, end = int(start_text), int(end_text)
    except ValueError as exc:
        raise ValueError(f"非法 stage 周次范围: {raw!r}") from exc
    if start > end:
        raise ValueError(f"stage 周次范围倒置: {raw!r}")
    return set(range(start, end + 1))


def _validate_week_assets(
    data: dict[str, Any], parsed: list[LessonManifest], report: ValidationReport
) -> None:
    from llm_course.papers import load_catalog

    assets = data.get("assets")
    if not isinstance(assets, dict):
        report.errors.append("roadmap.assets 必须是 week 1..48 的 mapping")
        return

    normalized_assets: dict[int, dict[str, Any]] = {}
    for raw_week, item in assets.items():
        try:
            week = int(raw_week)
        except (TypeError, ValueError):
            report.errors.append(f"assets 周次必须是整数: {raw_week!r}")
            continue
        if not isinstance(item, dict):
            report.errors.append(f"assets.{week} 必须是 mapping")
            continue
        normalized_assets[week] = item
    if sorted(normalized_assets) != list(range(1, 49)):
        report.errors.append("roadmap.assets 必须恰好覆盖 week 1..48")

    exercise_by_id = {exercise.exercise_id: exercise for exercise in EXERCISES}
    paper_ids = {paper.id for paper in load_catalog()}
    lesson_by_week = {lesson.week: lesson for lesson in parsed}
    for week in range(1, 49):
        item = normalized_assets.get(week)
        if item is None:
            continue
        missing = ASSET_FIELDS - item.keys()
        if missing:
            report.errors.append(f"assets.{week} 缺字段: {sorted(missing)}")
            continue

        lecture = item["lecture"]
        notebooks = item["notebooks"]
        exercise_ids = item["exercises"]
        sources = item["sources"]
        mode = item["mode"]
        deliverable = item["deliverable"]
        if not isinstance(lecture, str) or not (PROJECT_ROOT / lecture).is_file():
            report.errors.append(f"assets.{week} 讲义不存在: {lecture!r}")
        elif week in lesson_by_week and lecture not in lesson_by_week[week].readings:
            report.errors.append(f"assets.{week} 的 lecture 必须出现在该周 readings 中")

        if not isinstance(notebooks, list) or not notebooks:
            report.errors.append(f"assets.{week}.notebooks 必须是非空列表")
        else:
            for notebook in notebooks:
                if not isinstance(notebook, str) or not (PROJECT_ROOT / notebook).is_file():
                    report.errors.append(f"assets.{week} Notebook 不存在: {notebook!r}")

        if not isinstance(exercise_ids, list):
            report.errors.append(f"assets.{week}.exercises 必须是列表")
            exercise_ids = []
        for exercise_id in exercise_ids:
            exercise = exercise_by_id.get(str(exercise_id))
            if exercise is None:
                report.errors.append(f"assets.{week} 引用未知练习: {exercise_id!r}")
            elif week not in exercise.weeks:
                report.errors.append(
                    f"assets.{week} 与练习 {exercise.exercise_id} 的 manifest 周次不一致"
                )

        if mode not in {"implementation", "research", "capstone"}:
            report.errors.append(f"assets.{week}.mode 非法: {mode!r}")
        if mode == "implementation" and not exercise_ids:
            report.errors.append(f"assets.{week} 是实现周但没有 starter/checker")
        if not isinstance(deliverable, str) or not deliverable.strip():
            report.errors.append(f"assets.{week}.deliverable 不能为空")
        if not isinstance(sources, list) or not sources:
            report.errors.append(f"assets.{week}.sources 必须至少有一个一手来源")
        else:
            for source in sources:
                if not isinstance(source, str) or not (
                    source in paper_ids or source.startswith("https://")
                ):
                    report.errors.append(f"assets.{week} 引用未知来源: {source!r}")


def validate_roadmap(path: Path = ROADMAP_PATH) -> ValidationReport:
    report = ValidationReport()
    try:
        data = load_roadmap(path)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        report.errors.append(f"无法读取课程清单: {exc}")
        return report

    weeks = data.get("weeks")
    stages = data.get("stages")
    if not isinstance(weeks, list):
        report.errors.append("weeks 必须是列表")
        return report
    if not isinstance(stages, list):
        report.errors.append("stages 必须是列表")
        return report

    stage_ids = {item.get("id") for item in stages if isinstance(item, dict)}
    stage_weeks: dict[str, set[int]] = {}
    for stage in stages:
        if not isinstance(stage, dict):
            report.errors.append("每个 stage 必须是 mapping")
            continue
        try:
            stage_weeks[str(stage["id"])] = _stage_week_set(stage)
        except (KeyError, ValueError) as exc:
            report.errors.append(str(exc))

    parsed: list[LessonManifest] = []
    for index, item in enumerate(weeks, start=1):
        if not isinstance(item, dict):
            report.errors.append(f"第 {index} 个 week 不是 mapping")
            continue
        missing = LESSON_FIELDS - item.keys()
        if missing:
            report.errors.append(f"week {item.get('week', index)} 缺字段: {sorted(missing)}")
            continue
        try:
            lesson = LessonManifest.from_dict(item)
        except (KeyError, TypeError, ValueError) as exc:
            report.errors.append(f"week {item.get('week', index)} 无法解析: {exc}")
            continue
        parsed.append(lesson)
        if lesson.stage not in stage_ids:
            report.errors.append(f"week {lesson.week} 使用未知 stage: {lesson.stage}")
        elif lesson.week not in stage_weeks.get(lesson.stage, set()):
            report.errors.append(f"week {lesson.week} 超出 stage {lesson.stage} 声明范围")
        if not lesson.objectives or not lesson.experiments or not lesson.acceptance:
            report.errors.append(f"week {lesson.week} 的目标、实验和验收不能为空")
        for reading in lesson.readings:
            if reading.startswith("http"):
                continue
            if not (PROJECT_ROOT / reading).is_file():
                report.errors.append(f"week {lesson.week} 缺少阅读文件: {reading}")

    numbers = [lesson.week for lesson in parsed]
    if numbers != list(range(1, 49)):
        report.errors.append("课程必须按顺序包含 week 1..48，且不可重复")
    if data.get("course", {}).get("weeks") != 48:
        report.errors.append("course.weeks 应为 48")
    union = set().union(*stage_weeks.values()) if stage_weeks else set()
    if union != set(range(1, 49)) or sum(map(len, stage_weeks.values())) != 48:
        report.errors.append("stage 周次必须无重叠地覆盖 1..48")

    _validate_week_assets(data, parsed, report)
    report.merge(validate_weekly_lectures(data))
    return report


def validate_notebook_contracts() -> ValidationReport:
    report = ValidationReport()
    try:
        data = load_roadmap()
    except (OSError, ValueError, yaml.YAMLError) as exc:
        report.errors.append(f"无法读取 Notebook 契约: {exc}")
        return report

    notebook_weeks: dict[str, set[int]] = {}
    notebook_starters: dict[str, set[str]] = {}
    for raw_week, item in data.get("assets", {}).items():
        if not isinstance(item, dict):
            continue
        week = int(raw_week)
        for path in item.get("notebooks", []):
            notebook_weeks.setdefault(path, set()).add(week)
            notebook_starters.setdefault(path, set()).update(map(str, item.get("exercises", [])))

    for relative_path, expected_weeks in notebook_weeks.items():
        path = PROJECT_ROOT / relative_path
        if not path.is_file():
            continue
        try:
            notebook = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            report.errors.append(f"Notebook 无法读取: {relative_path}: {exc}")
            continue
        contract = notebook.get("metadata", {}).get("llm_course")
        if not isinstance(contract, dict):
            report.errors.append(f"Notebook 缺少 metadata.llm_course: {relative_path}")
            continue
        missing = NOTEBOOK_CONTRACT_FIELDS - contract.keys()
        if missing:
            report.errors.append(f"Notebook 契约缺字段 {relative_path}: {sorted(missing)}")
            continue
        if set(map(int, contract["weeks"])) != expected_weeks:
            report.errors.append(f"Notebook 周次与 roadmap 不一致: {relative_path}")
        if set(map(str, contract["starter_ids"])) != notebook_starters[relative_path]:
            report.errors.append(f"Notebook starter IDs 与 roadmap 不一致: {relative_path}")
        if not isinstance(contract["estimated_minutes"], int) or contract["estimated_minutes"] <= 0:
            report.errors.append(f"Notebook estimated_minutes 必须为正整数: {relative_path}")
        if not isinstance(contract["prerequisites"], list):
            report.errors.append(f"Notebook prerequisites 必须是列表: {relative_path}")
        assertion = contract["completion_assertion"]
        if not isinstance(assertion, str) or not assertion.strip():
            report.errors.append(f"Notebook completion_assertion 不能为空: {relative_path}")
    return report


def render_learning_path() -> str:
    """渲染唯一的 1–48 学习路径。"""
    data = load_roadmap()
    week_by_number = {int(item["week"]): item for item in data["weeks"]}
    assets = {int(key): value for key, value in data["assets"].items()}
    lines = [
        "<!-- 由 `uv run llm-course course path --write` 生成，请勿手动维护。 -->",
        "# LLM 学习路径（1–48）",
        "",
        "所有人都按 1 → 48 推进；学习节奏可以不同，完成标准保持一致。",
        "可以根据自己的基础调整每一周所用的时间，但完成标准保持一致。",
        "",
        "[课程使用说明](../README.md#统一的学习路径怎么使用) · "
        "[架构演进图](architecture_evolution.md) · "
        "[练习清单](../exercises/manifest.yaml)",
        "",
        "| 周次 | 主题 | 讲义 | Notebook | 练习 / 交付物 |",
        "|---:|---|---|---|---|",
    ]
    for week in range(1, 49):
        lesson = week_by_number[week]
        asset = assets[week]
        lecture_path = Path(asset["lecture"])
        try:
            lecture_path = lecture_path.relative_to("docs")
        except ValueError:
            pass
        lecture = f"[进入讲义]({lecture_path.as_posix()})"
        notebooks = ", ".join(
            f"[{Path(notebook).name}](../{Path(notebook).as_posix()})"
            for notebook in asset["notebooks"]
        )
        exercise_ids = ", ".join(map(str, asset["exercises"]))
        output = exercise_ids or asset["deliverable"]
        lines.append(f"| {week} | {lesson['title']} | {lecture} | {notebooks} | {output} |")
    lines.extend(
        [
            "",
            "使用方式：从第 1 周开始，依次完成对应讲义、Notebook 与练习；通过本周的完成标准后，",
            "再进入下一周。中途回来时，从第一个尚未完成的周次继续即可。",
            "",
        ]
    )
    return "\n".join(lines)


def write_learning_path(path: Path | None = None) -> Path:
    """同步唯一的学习路径文档。"""
    path = path or LEARNING_PATH_DOCUMENT
    path.write_text(render_learning_path(), encoding="utf-8")
    return path


def validate_course_assets() -> ValidationReport:
    """检查入口、互动实验、Notebook 契约与自动生成路径是否完整。"""

    report = ValidationReport()
    for relative_path in REQUIRED_COURSE_ASSETS:
        if not (PROJECT_ROOT / relative_path).is_file():
            report.errors.append(f"缺少课程资产: {relative_path}")
    if LEARNING_PATH_DOCUMENT.is_file():
        try:
            actual = LEARNING_PATH_DOCUMENT.read_text(encoding="utf-8").replace("\r\n", "\n")
            expected = render_learning_path().replace("\r\n", "\n")
            if actual != expected:
                report.errors.append(
                    "docs/learning_path.md 已经漂移；"
                    "请运行 `uv run llm-course course path --write` 重新生成。"
                )
        except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
            report.errors.append(f"无法核对统一学习路径: {exc}")
    report.merge(validate_notebook_contracts())
    return report


def run_pytest() -> int:
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"], cwd=PROJECT_ROOT, check=False
    )
    return completed.returncode


def check_course(run_tests: bool = True) -> int:
    from llm_course.papers import validate_catalog

    report = validate_roadmap()
    report.merge(validate_course_assets())
    report.merge(validate_exercise_assets())
    report.merge(validate_catalog())
    for warning in report.warnings:
        print(f"警告: {warning}")
    for error in report.errors:
        print(f"错误: {error}")
    if not report.ok:
        return 1
    print("48/48 周课程闭环、Notebook 契约、练习与资料目录校验通过。")
    if run_tests:
        print("开始运行 CPU 测试……")
        return run_pytest()
    return 0
