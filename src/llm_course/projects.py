from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

from llm_course.paths import PROJECT_ROOT
from llm_course.schemas import ValidationReport

MANIFEST_PATH = PROJECT_ROOT / "course" / "projects.yaml"
VALID_STATUSES = {"available", "planned"}


@dataclass(frozen=True)
class Project:
    project_id: str
    slug: str
    title: str
    lessons: tuple[int, ...]
    status: str
    directory: str
    package: str
    checks: tuple[str, ...]

    @property
    def lesson_label(self) -> str:
        ranges: list[str] = []
        start = previous = self.lessons[0]
        for lesson in self.lessons[1:]:
            if lesson == previous + 1:
                previous = lesson
                continue
            ranges.append(str(start) if start == previous else f"{start}–{previous}")
            start = previous = lesson
        ranges.append(str(start) if start == previous else f"{start}–{previous}")
        return "、".join(ranges)

    @property
    def available(self) -> bool:
        return self.status == "available"


def load_projects(path: Path = MANIFEST_PATH) -> tuple[Project, ...]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict) or not isinstance(data.get("projects"), list):
        raise ValueError("project manifest 必须包含 projects 列表")

    projects: list[Project] = []
    for item in data["projects"]:
        if not isinstance(item, dict):
            raise ValueError("project manifest 中的每条记录必须是 mapping")
        projects.append(
            Project(
                project_id=str(item["id"]),
                slug=str(item["slug"]),
                title=str(item["title"]),
                lessons=tuple(int(lesson) for lesson in item["lessons"]),
                status=str(item["status"]),
                directory=str(item["directory"]),
                package=str(item["package"]),
                checks=tuple(str(check) for check in item.get("checks", ())),
            )
        )
    return tuple(projects)


PROJECTS = load_projects()


def select_project(identifier: str) -> Project:
    normalized = identifier.strip().lower().replace("_", "-")
    for project in PROJECTS:
        if normalized in {project.project_id, project.slug}:
            return project
    choices = ", ".join(f"{project.project_id}/{project.slug}" for project in PROJECTS)
    raise ValueError(f"未知大作业 {identifier!r}；可用项目为 {choices}")


def print_projects() -> int:
    print("编号  课次           状态      贯穿式大作业")
    for project in PROJECTS:
        status = "可开始" if project.available else "建设中"
        print(
            f"{project.project_id:<4}  {project.lesson_label:<13}  "
            f"{status:<8}  {project.title} ({project.slug})"
        )
    print("\n核查示例: uv run llm-course projects check 01")
    return 0


def run_project_checks(identifier: str) -> int:
    try:
        project = select_project(identifier)
    except ValueError as exc:
        print(f"错误: {exc}")
        return 2
    if not project.available:
        print(f"大作业 {project.project_id} 尚在建设；先阅读其 README 中的目标和前置课次。")
        return 2

    print(f"开始核查大作业 {project.project_id}: {project.title}")
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *(str(PROJECT_ROOT / p) for p in project.checks)],
        cwd=PROJECT_ROOT,
        check=False,
    )
    if completed.returncode == 0:
        print("大作业公开核查通过。继续运行完整入口并完成 README 中的报告验收。")
    else:
        print("大作业尚未通过；按项目 README 的固定顺序处理第一条失败。")
    return completed.returncode


def validate_project_assets() -> ValidationReport:
    report = ValidationReport()
    seen_ids: set[str] = set()
    seen_slugs: set[str] = set()
    for project in PROJECTS:
        if not project.project_id.isdigit() or len(project.project_id) != 2:
            report.errors.append(f"大作业编号必须是两位数字: {project.project_id}")
        if project.project_id in seen_ids:
            report.errors.append(f"大作业编号重复: {project.project_id}")
        if project.slug in seen_slugs:
            report.errors.append(f"大作业别名重复: {project.slug}")
        seen_ids.add(project.project_id)
        seen_slugs.add(project.slug)
        if project.status not in VALID_STATUSES:
            report.errors.append(f"大作业状态无效: {project.project_id}: {project.status}")
        if not project.package.isidentifier() or not project.package.startswith("student_"):
            report.errors.append(f"大作业 learner-owned package 名称无效: {project.project_id}")
        if tuple(sorted(set(project.lessons))) != project.lessons or any(
            not 1 <= lesson <= 48 for lesson in project.lessons
        ):
            report.errors.append(f"大作业课次必须升序、唯一且位于 1..48: {project.project_id}")

        directory = PROJECT_ROOT / project.directory
        if not directory.is_dir() or not (directory / "README.md").is_file():
            report.errors.append(f"缺少大作业目录或 README: {project.directory}")
            continue
        if project.available and not project.checks:
            report.errors.append(f"可开始的大作业必须有公开核查: {project.project_id}")
        for check in project.checks:
            path = PROJECT_ROOT / check
            if not path.is_file():
                report.errors.append(f"缺少大作业核查: {check}")
                continue
            try:
                compile(path.read_text(encoding="utf-8"), str(path), "exec")
            except (OSError, SyntaxError) as exc:
                report.errors.append(f"大作业核查无法编译: {check}: {exc}")

        if project.available:
            package_directory = directory / project.package
            student_files = sorted(package_directory.glob("*.py"))
            if not student_files:
                report.errors.append(f"大作业缺少 learner-owned package: {project.project_id}")
            has_blank_core = False
            for path in student_files:
                source = path.read_text(encoding="utf-8")
                if "llm_from_scratch" in source:
                    report.errors.append(f"大作业 starter 不得导入参考实现: {path}")
                has_blank_core |= "raise NotImplementedError" in source
                try:
                    compile(source, str(path), "exec")
                except SyntaxError as exc:
                    report.errors.append(f"大作业 starter 无法编译: {path}: {exc}")
            if not has_blank_core:
                report.errors.append(f"大作业必须保留 learner-owned 核心空缺: {project.project_id}")
    return report
