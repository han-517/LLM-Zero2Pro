from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TextIO

from llm_course.course import (
    check_course,
    render_learning_path,
    write_learning_path,
)
from llm_course.discovery import PROFILE_QUERIES, SOURCES, update_inbox_multisource
from llm_course.doctor import run_doctor
from llm_course.exercises import print_exercises, run_exercise_checks
from llm_course.lab import launch_lab
from llm_course.papers import check_links, generate_graph, load_catalog, validate_catalog


def _configure_utf8_stream(stream: TextIO) -> None:
    reconfigure = getattr(stream, "reconfigure", None)
    if sys.platform == "win32" and callable(reconfigure):
        reconfigure(encoding="utf-8", errors="replace")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llm-course", description="文本 LLM 课程工具")
    subparsers = parser.add_subparsers(dest="command", required=True)

    lab_parser = subparsers.add_parser("lab", help="打开课程欢迎页并启动 JupyterLab")
    lab_parser.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    lab_parser.add_argument("--port", type=int, help="指定 JupyterLab 端口")

    doctor_parser = subparsers.add_parser("doctor", help="检查 Python、PyTorch 和设备")
    doctor_parser.add_argument("--json", action="store_true", help="输出 JSON")

    papers_parser = subparsers.add_parser("papers", help="论文目录工具")
    papers_sub = papers_parser.add_subparsers(dest="papers_command", required=True)
    validate_parser = papers_sub.add_parser("validate", help="校验论文元数据")
    validate_parser.add_argument("--check-links", action="store_true", help="联网检查主链接")
    update_parser = papers_sub.add_parser(
        "update", help="从 arXiv、Semantic Scholar 和 Hugging Face 更新候选池"
    )
    update_parser.add_argument(
        "--max-results", type=int, default=20, help="每个来源最多获取的结果数（1..100）"
    )
    update_parser.add_argument(
        "--source", choices=("all", *SOURCES), default="all", help="指定来源；默认尝试全部"
    )
    update_parser.add_argument(
        "--profile",
        choices=tuple(PROFILE_QUERIES),
        default="all",
        help="课程主题检索配置",
    )
    update_parser.add_argument(
        "--since",
        help="只保留此日期及之后的候选，格式 YYYY-MM-DD",
    )
    update_parser.add_argument(
        "--query",
        default=None,
        help="自定义检索词；提供时覆盖 --profile 的内置查询",
    )
    papers_sub.add_parser("graph", help="生成 Mermaid 论文关系图")

    course_parser = subparsers.add_parser("course", help="课程清单、路径与测试")
    course_sub = course_parser.add_subparsers(dest="course_command", required=True)
    check_parser = course_sub.add_parser("check", help="校验 48 周闭环并运行测试")
    check_parser.add_argument("--no-tests", action="store_true", help="只校验数据，不运行 pytest")
    path_parser = course_sub.add_parser("path", help="查看统一的 1–48 学习路线")
    path_parser.add_argument("--write", action="store_true", help="同步到 docs/learning_path.md")
    path_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="可选的输出路径；不提供时只打印",
    )

    exercises_parser = subparsers.add_parser("exercises", help="代码模板与独立核查")
    exercises_sub = exercises_parser.add_subparsers(dest="exercises_command", required=True)
    exercises_sub.add_parser("list", help="列出模板、周次和填写状态")
    exercise_check = exercises_sub.add_parser("check", help="运行练习的公开行为测试")
    exercise_check.add_argument(
        "exercise", nargs="?", default="all", help="编号或别名（如 07、rope）；默认 all"
    )
    return parser


def _print_report(report) -> int:
    for warning in report.warnings:
        print(f"警告: {warning}")
    for error in report.errors:
        print(f"错误: {error}")
    if report.ok:
        print("校验通过。")
        return 0
    return 1


def main(argv: list[str] | None = None) -> int:
    _configure_utf8_stream(sys.stdout)
    _configure_utf8_stream(sys.stderr)
    args = _build_parser().parse_args(argv)
    if args.command == "lab":
        return launch_lab(no_browser=args.no_browser, port=args.port)
    if args.command == "doctor":
        return run_doctor(as_json=args.json)
    if args.command == "course":
        if args.course_command == "check":
            return check_course(run_tests=not args.no_tests)
        if args.course_command == "path":
            if args.write or args.output is not None:
                path = write_learning_path(args.output)
                print(f"已同步 {path}")
            else:
                print(render_learning_path())
            return 0
    if args.command == "exercises":
        if args.exercises_command == "list":
            return print_exercises()
        if args.exercises_command == "check":
            return run_exercise_checks(args.exercise)
    if args.command == "papers":
        if args.papers_command == "validate":
            report = validate_catalog()
            if args.check_links and report.ok:
                report.merge(check_links(load_catalog()))
            return _print_report(report)
        if args.papers_command == "update":
            if not 1 <= args.max_results <= 100:
                print("错误: --max-results 必须在 1..100")
                return 2
            try:
                added, total, errors = update_inbox_multisource(
                    max_results=args.max_results,
                    source=args.source,
                    query=args.query,
                    profile=args.profile,
                    since=args.since,
                )
            except (OSError, RuntimeError, ValueError) as exc:
                print(f"错误: {exc}")
                return 1
            for error in errors:
                print(f"警告: 来源暂不可用：{error}")
            print(f"新增 {added} 篇候选论文；候选池共 {total} 篇。")
            return 0
        if args.papers_command == "graph":
            path = generate_graph(load_catalog())
            print(f"已生成 {path}")
            return 0
    return 2
