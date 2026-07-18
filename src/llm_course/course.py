from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

from llm_course.exercises import validate_exercise_assets
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

REQUIRED_COURSE_ASSETS = (
    "docs/core_learning_path.md",
    "docs/code_templates.md",
    "docs/interactive/core-concepts.html",
    "docs/architecture_evolution.md",
    "docs/interactive/architecture-evolution.html",
    "exercises/starter/01_stable_softmax.py",
    "exercises/starter/02_causal_attention.py",
    "exercises/starter/03_kv_cache_budget.py",
    "exercises/starter/04_sft_shift.py",
    "exercises/starter/05_moe_capacity.py",
    "exercises/starter/README.md",
    "requirements-hosted-gpu.txt",
)


def load_roadmap(path: Path = ROADMAP_PATH) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError("roadmap 根节点必须是 mapping")
    return data


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
    declared = data.get("course", {}).get("weeks")
    if declared != 48:
        report.errors.append(f"course.weeks 应为 48，实际为 {declared!r}")
    return report


def validate_course_assets() -> ValidationReport:
    """检查学习入口、交互实验和 starter 是否随课程一起分发。"""

    report = ValidationReport()
    for relative_path in REQUIRED_COURSE_ASSETS:
        if not (PROJECT_ROOT / relative_path).is_file():
            report.errors.append(f"缺少课程资产: {relative_path}")
    return report


def run_pytest() -> int:
    command = [sys.executable, "-m", "pytest", "-q"]
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
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
    print("课程清单与论文目录校验通过。")
    if run_tests:
        print("开始运行 CPU 测试……")
        return run_pytest()
    return 0
