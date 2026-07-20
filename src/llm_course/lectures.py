from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from llm_course.paths import PROJECT_ROOT
from llm_course.schemas import ValidationReport

MIN_NON_WHITESPACE_CHARS = 2200
REQUIRED_HEADING_GROUPS: dict[str, tuple[str, ...]] = {
    "学习目标": ("学习目标",),
    "前置知识": ("前置",),
    "直觉": ("直觉",),
    "张量或数据契约": ("张量", "数据契约"),
    "推导或机制": ("推导", "机制"),
    "例子": ("示例", "例子", "手算", "数值例"),
    "代码实现": ("代码", "实现"),
    "反例与调试": ("反例", "调试", "误区"),
    "实验": ("实验",),
    "验收": ("验收",),
    "一手来源": ("一手来源",),
}
MARKDOWN_LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
HTTPS_RE = re.compile(r"https://[^\s)>]+")
HEADING_RE = re.compile(r"(?m)^##+\s+(.+?)\s*$")
INLINE_REPO_PATH_RE = re.compile(
    r"`((?:learning|setup|checks|solutions|src|course)/[^`\n#]+"
    r"(?:#[^`\n]+)?)`"
)


def _has_heading(headings: list[str], keywords: tuple[str, ...]) -> bool:
    return any(any(keyword in heading for keyword in keywords) for heading in headings)


def validate_weekly_lectures(data: dict[str, Any], root: Path = PROJECT_ROOT) -> ValidationReport:
    """验证 48 周讲义的唯一映射、教学结构、篇幅、代码、来源与本地链接。"""

    report = ValidationReport()
    assets = data.get("assets")
    if not isinstance(assets, dict):
        report.errors.append("无法验证逐周讲义：roadmap.assets 不是 mapping")
        return report

    lecture_by_week: dict[int, str] = {}
    for week in range(1, 49):
        item = assets.get(week, assets.get(str(week)))
        if not isinstance(item, dict):
            report.errors.append(f"第 {week} 周缺少讲义资产")
            continue
        lecture = item.get("lesson")
        if not isinstance(lecture, str):
            report.errors.append(f"第 {week} 课 lesson 必须是字符串")
            continue
        expected_prefix = f"learning/readings/lessons/{week:02d}_"
        if not lecture.replace("\\", "/").startswith(expected_prefix):
            report.errors.append(
                f"第 {week} 课必须使用独立讲义 {expected_prefix}*.md，当前为 {lecture!r}"
            )
        lecture_by_week[week] = lecture

    if len(set(lecture_by_week.values())) != 48:
        report.errors.append("48 周必须各自映射到唯一的逐周讲义文件")

    for week, relative_path in lecture_by_week.items():
        path = root / relative_path
        if not path.is_file():
            report.errors.append(f"第 {week} 周逐周讲义不存在: {relative_path}")
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            report.errors.append(f"第 {week} 周逐周讲义无法读取: {exc}")
            continue

        compact_length = len(re.sub(r"\s+", "", text))
        if compact_length < MIN_NON_WHITESPACE_CHARS:
            report.errors.append(
                f"第 {week} 周讲义过短: {compact_length} < {MIN_NON_WHITESPACE_CHARS} 非空白字符"
            )

        first_heading = next(
            (line.strip() for line in text.splitlines() if line.startswith("# ")),
            "",
        )
        if str(week) not in first_heading:
            report.errors.append(f"第 {week} 周讲义 H1 必须包含周次")

        headings = HEADING_RE.findall(text)
        missing = [
            label
            for label, keywords in REQUIRED_HEADING_GROUPS.items()
            if not _has_heading(headings, keywords)
        ]
        if missing:
            report.errors.append(f"第 {week} 周讲义缺少结构标题: {missing}")

        backtick_fences = text.count(chr(96) * 3)
        tilde_fences = text.count("~~~")
        has_closed_fence = (backtick_fences >= 2 and backtick_fences % 2 == 0) or (
            tilde_fences >= 2 and tilde_fences % 2 == 0
        )
        if not has_closed_fence:
            report.errors.append(f"第 {week} 周讲义必须包含闭合的 fenced code block")

        source_urls = set(HTTPS_RE.findall(text))
        if len(source_urls) < 3:
            report.errors.append(
                f"第 {week} 周讲义至少需要 3 个 https 一手来源，当前 {len(source_urls)}"
            )

        backtick_fence = re.escape(chr(96) * 3)
        tilde_fence = re.escape("~~~")
        prose = re.sub(rf"(?ms)^{backtick_fence}.*?^{backtick_fence}$", "", text)
        prose = re.sub(rf"(?ms)^{tilde_fence}.*?^{tilde_fence}$", "", prose)
        for target in MARKDOWN_LINK_RE.findall(prose):
            local_target = target.split("#", maxsplit=1)[0]
            if not local_target or "://" in local_target:
                continue
            resolved = (path.parent / local_target).resolve()
            if not resolved.is_relative_to(root.resolve()) or not resolved.exists():
                report.errors.append(f"第 {week} 周讲义包含失效或越界本地链接: {target}")

        for target in INLINE_REPO_PATH_RE.findall(prose):
            local_target = target.split("#", maxsplit=1)[0]
            resolved = (root / local_target).resolve()
            if not resolved.is_relative_to(root.resolve()) or not resolved.exists():
                report.errors.append(f"第 {week} 周讲义包含失效的代码样式仓库路径: {target}")

    return report
