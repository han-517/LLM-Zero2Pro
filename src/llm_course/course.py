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
ASSET_FIELDS = {"lesson", "lab", "interactive", "exercises", "sources", "mode", "deliverable"}
NOTEBOOK_CONTRACT_FIELDS = {
    "weeks",
    "estimated_minutes",
    "prerequisites",
    "starter_ids",
    "completion_assertion",
}
REQUIRED_COURSE_ASSETS = (
    "setup/environment.md",
    "setup/vscode.md",
    "setup/jupyterlab.md",
    "learning/README.md",
    "learning/labs/README.md",
    "learning/readings/lessons/README.md",
    "learning/readings/references/code_templates.md",
    "learning/readings/interactive/foundations-lab.html",
    "learning/readings/interactive/index.html",
    "learning/readings/interactive/core-concepts.html",
    "learning/readings/interactive/architecture-evolution.html",
    "learning/readings/interactive/architecture-lab.html",
    "learning/readings/interactive/training-and-alignment.html",
    "learning/readings/interactive/serving-lab.html",
    "learning/readings/interactive/multimodal-flow.html",
    "learning/readings/extensions/multimodal.md",
    "course/exercises.yaml",
    "learning/labs/starter/README.md",
    "checks/exercises/__init__.py",
    "LLM-Zero2Pro.code-workspace",
    ".vscode/extensions.json",
    ".vscode/settings.json",
    ".vscode/tasks.json",
    "requirements-hosted-gpu.txt",
)
LEARNING_PATH_DOCUMENT = PROJECT_ROOT / "learning" / "README.md"


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
        report.errors.append("roadmap.assets 必须是 lesson 1..48 的 mapping")
        return

    normalized_assets: dict[int, dict[str, Any]] = {}
    for raw_week, item in assets.items():
        try:
            week = int(raw_week)
        except (TypeError, ValueError):
            report.errors.append(f"assets 课次必须是整数: {raw_week!r}")
            continue
        if not isinstance(item, dict):
            report.errors.append(f"assets.{week} 必须是 mapping")
            continue
        normalized_assets[week] = item
    if sorted(normalized_assets) != list(range(1, 49)):
        report.errors.append("roadmap.assets 必须恰好覆盖 lesson 1..48")

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

        lesson_path = item["lesson"]
        lab = item["lab"]
        interactive = item["interactive"]
        exercise_ids = item["exercises"]
        sources = item["sources"]
        mode = item["mode"]
        deliverable = item["deliverable"]
        if not isinstance(lesson_path, str) or not (PROJECT_ROOT / lesson_path).is_file():
            report.errors.append(f"assets.{week} 讲义不存在: {lesson_path!r}")
        elif week in lesson_by_week and lesson_path not in lesson_by_week[week].readings:
            report.errors.append(f"assets.{week} 的 lesson 必须出现在该课 readings 中")

        if lab is not None:
            if not isinstance(lab, dict) or set(lab) != {"ipynb", "python"}:
                report.errors.append(f"assets.{week}.lab 必须为 null 或包含 ipynb/python")
            else:
                for label, relative_path in lab.items():
                    if (
                        not isinstance(relative_path, str)
                        or not (PROJECT_ROOT / relative_path).is_file()
                    ):
                        report.errors.append(f"assets.{week} {label} 实验不存在: {relative_path!r}")
                if isinstance(lab.get("ipynb"), str) and isinstance(lab.get("python"), str):
                    if Path(lab["ipynb"]).with_suffix(".py") != Path(lab["python"]):
                        report.errors.append(f"assets.{week} 的双格式实验名称不配对")
        elif mode == "implementation":
            report.errors.append(f"assets.{week} 是实现课但没有实验文件")

        if not isinstance(interactive, list) or not interactive:
            report.errors.append(f"assets.{week}.interactive 必须是非空列表")
        else:
            for relative_path in interactive:
                if (
                    not isinstance(relative_path, str)
                    or not (PROJECT_ROOT / relative_path).is_file()
                ):
                    report.errors.append(f"assets.{week} 互动图不存在: {relative_path!r}")

        if not isinstance(exercise_ids, list):
            report.errors.append(f"assets.{week}.exercises 必须是列表")
            exercise_ids = []
        for exercise_id in exercise_ids:
            exercise = exercise_by_id.get(str(exercise_id).zfill(2))
            if exercise is None:
                report.errors.append(f"assets.{week} 引用未知练习: {exercise_id!r}")
            elif week not in exercise.weeks:
                report.errors.append(
                    f"assets.{week} 与练习 {exercise.exercise_id} 的 manifest 课次不一致"
                )

        if mode not in {"implementation", "research", "capstone"}:
            report.errors.append(f"assets.{week}.mode 非法: {mode!r}")
        if mode == "implementation" and not exercise_ids:
            report.errors.append(f"assets.{week} 是实现课但没有 starter/checker")
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
        report.errors.append(f"无法读取实验契约: {exc}")
        return report

    notebook_weeks: dict[str, set[int]] = {}
    notebook_starters: dict[str, set[str]] = {}
    paired_python: dict[str, str] = {}
    for raw_week, item in data.get("assets", {}).items():
        if not isinstance(item, dict) or not isinstance(item.get("lab"), dict):
            continue
        week = int(raw_week)
        notebook = item["lab"].get("ipynb")
        python_lab = item["lab"].get("python")
        if not isinstance(notebook, str) or not isinstance(python_lab, str):
            continue
        notebook_weeks.setdefault(notebook, set()).add(week)
        notebook_starters.setdefault(notebook, set()).update(
            str(exercise_id).zfill(2) for exercise_id in item.get("exercises", [])
        )
        paired_python[notebook] = python_lab

    for relative_path, expected_weeks in notebook_weeks.items():
        path = PROJECT_ROOT / relative_path
        python_path = PROJECT_ROOT / paired_python[relative_path]
        if not path.is_file() or not python_path.is_file():
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
            report.errors.append(f"Notebook 课次与 roadmap 不一致: {relative_path}")
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


def _catalog_target(relative_path: str) -> str:
    path = Path(relative_path)
    try:
        return path.relative_to("learning").as_posix()
    except ValueError:
        return (Path("..") / path).as_posix()


def _catalog_link(relative_path: str, label: str | None = None) -> str:
    return f"[{label or Path(relative_path).name}]({_catalog_target(relative_path)})"


def render_learning_path() -> str:
    """从课程清单渲染唯一的 1–48 课 Markdown 目录。"""

    data = load_roadmap()
    assets = {int(key): value for key, value in data["assets"].items()}
    exercise_by_id = {exercise.exercise_id: exercise for exercise in EXERCISES}
    weekly_hours = data.get("course", {}).get("weekly_hours", "8-10")
    lines = [
        "<!-- 由 `uv run llm-course course path --write` 生成，请勿手动维护。 -->",
        "# LLM-Zero2Pro 学习目录（第 1–48 课）",
        "",
        "这是课程唯一入口。不要随机挑 Notebook：从第 1 课开始，完成本课验收后再进入下一课。",
        "课程主线严格聚焦文本 LLM；多模态位于文末选修区，不计入 48 课毕业要求。",
        "",
        "## 第一次使用",
        "",
        "1. 在仓库根目录运行 `uv sync` 和 `uv run llm-course doctor`。",
        "2. VS Code 用户运行 `uv run llm-course vscode`；Explorer 只显示 `learning/`。",
        "   手动方式是 `code LLM-Zero2Pro.code-workspace learning/README.md`。",
        "3. 在 VS Code 选择项目 `.venv` 解释器；打开 `.ipynb` 时选择同一个 Kernel。",
        "4. 每个实验只需选择 `.ipynb` 或同名 `# %%` Python 文件之一，不要重复完成。",
        "5. 互动 HTML 使用系统浏览器或 VS Code Live Preview 打开。",
        "",
        "环境细节见 [VS Code 指南](../setup/vscode.md)、[Windows/macOS 环境指南]"
        "(../setup/environment.md)和[JupyterLab 可选指南](../setup/jupyterlab.md)。",
        "",
        "## 固定学习顺序",
        "",
        "每课都按：**讲义 → 补充阅读 → 互动图 → 实验二选一 → starter → 自动核查 → 交付物与验收**。",
        "没有独立 starter 的研究课，以清单中的交付物和完成标准验收。",
        "",
        "## 九阶段总览",
        "",
        "| 阶段 | 课次 | 主题 |",
        "|---:|---:|---|",
    ]
    for index, stage in enumerate(data["stages"], start=1):
        lines.append(f"| {index} | {stage['weeks']} | {stage['title']} |")

    for stage_index, stage in enumerate(data["stages"], start=1):
        lines.extend(["", f"## 阶段 {stage_index}：{stage['title']}（第 {stage['weeks']} 课）", ""])
        stage_lessons = [lesson for lesson in data["weeks"] if lesson["stage"] == stage["id"]]
        for lesson in stage_lessons:
            week = int(lesson["week"])
            asset = assets[week]
            prerequisite = "课程环境准备" if week == 1 else f"完成第 {week - 1:02d} 课"
            lines.extend(
                [
                    f"### 第 {week:02d} 课：{lesson['title']}",
                    "",
                    f"- **学习目标**：{'；'.join(lesson['objectives'])}",
                    f"- **前置知识**：{prerequisite}",
                    f"- **预计时间**：{weekly_hours} 小时，可按掌握程度拆分多天完成",
                    f"- **本课内容**：{'；'.join(lesson['experiments'])}",
                    "- **按此顺序学习**：",
                    f"  1. 阅读 {_catalog_link(asset['lesson'], '本课完整讲义')}。",
                ]
            )
            extra_readings = [path for path in lesson["readings"] if path != asset["lesson"]]
            if extra_readings:
                links = "、".join(_catalog_link(path) for path in extra_readings)
                lines.append(f"  2. 按需阅读 {links}，用于补齐阶段背景和方法。")
            else:
                lines.append("  2. 本课没有额外阅读，复述讲义中的形状和公式。")
            interactive_links = "、".join(
                _catalog_link(path, "互动图") for path in asset["interactive"]
            )
            lines.append(f"  3. 打开 {interactive_links}，先预测控件变化，再观察结果。")
            lab = asset["lab"]
            if isinstance(lab, dict):
                lines.append(
                    "  4. 实验二选一："
                    f"{_catalog_link(lab['ipynb'], '.ipynb')} 或 "
                    f"{_catalog_link(lab['python'], 'VS Code # %% .py')}。"
                )
            else:
                lines.append(
                    "  4. 本课无独立代码实验；运行 `uv run llm-course doctor` 并保存诊断结果。"
                )
            exercise_ids = [str(item).zfill(2) for item in asset["exercises"]]
            if exercise_ids:
                starter_links = "、".join(
                    _catalog_link(exercise_by_id[item].template, f"starter {item}")
                    for item in exercise_ids
                )
                lines.append(f"  5. 填写 {starter_links} 中保留的核心空缺。")
                commands = "；".join(
                    f"`uv run llm-course exercises check {item}`" for item in exercise_ids
                )
                lines.append(f"  6. 自动核查：{commands}。")
            else:
                lines.append("  5. 本课没有独立 starter，完成研究记录或实验报告。")
                lines.append("  6. 按讲义中的断言复现结果，并检查输出中没有隐藏状态。")
            lines.extend(
                [
                    f"  7. **交付物**：{asset['deliverable']}。",
                    f"- **完成标准**：{'；'.join(lesson['acceptance'])}",
                    f"- **一手来源**：{'；'.join(str(source) for source in asset['sources'])}",
                    "",
                ]
            )

    lines.extend(
        [
            "## 48 课之后：多模态选修",
            "",
            "先阅读[多模态桥接讲义](readings/extensions/multimodal.md)和"
            "[数据流互动图](readings/interactive/multimodal-flow.html)，再从以下格式任选一种：",
            "",
            "- [多模态 Notebook](labs/optional/80_multimodal_bridge.ipynb)",
            "- [多模态 VS Code 实验](labs/optional/80_multimodal_bridge.py)",
            "- [多模态 starter](labs/starter/21_multimodal_bridge.py)",
            "",
            "## 学习目录边界",
            "",
            "`learning/readings/` 只放阅读与互动材料；`learning/labs/` 只放可运行或填写的实验。",
            "环境在 `setup/`，课程配置在 `course/`，公开核查在 `checks/`，参考实现位于 `src/`。",
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
                    "learning/README.md 已经漂移；"
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
