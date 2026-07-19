from pathlib import Path

from llm_course.lectures import (
    MIN_NON_WHITESPACE_CHARS,
    validate_weekly_lectures,
)

HEADINGS = (
    "学习目标",
    "前置知识",
    "核心直觉",
    "张量与数据契约",
    "公式推导与算法机制",
    "手算与数值示例",
    "最小代码实现",
    "反例、误区与调试",
    "实验与 Notebook 对照",
    "验收标准",
    "一手来源",
)


def _lecture_text(week: int, *, body_size: int = MIN_NON_WHITESPACE_CHARS) -> str:
    sections = "\n".join(f"## {heading}\n本节给出可核查解释。" for heading in HEADINGS)
    body = "机制解释与数值验证。" * body_size
    fence = "\n" + chr(96) * 3 + "python\nprint('offline')\n" + chr(96) * 3
    sources = (
        "\n- [source 1](https://example.org/paper)\n"
        "- [source 2](https://example.org/code)\n"
        "- [source 3](https://example.org/docs)\n"
    )
    return f"# 第 {week} 周：测试讲义\n{sections}\n{body}{fence}{sources}"


def _valid_course(tmp_path: Path) -> dict:
    assets = {}
    for week in range(1, 49):
        relative = f"docs/weeks/{week:02d}_topic.md"
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_lecture_text(week), encoding="utf-8")
        assets[week] = {"lecture": relative}
    return {"assets": assets}


def test_weekly_lecture_contract_accepts_48_substantive_unique_lessons(
    tmp_path: Path,
) -> None:
    data = _valid_course(tmp_path)
    assert validate_weekly_lectures(data, tmp_path).ok


def test_weekly_lecture_contract_rejects_reuse_and_short_outline(
    tmp_path: Path,
) -> None:
    data = _valid_course(tmp_path)
    data["assets"][2]["lecture"] = data["assets"][1]["lecture"]
    short_path = tmp_path / data["assets"][3]["lecture"]
    short_path.write_text(_lecture_text(3, body_size=1), encoding="utf-8")

    report = validate_weekly_lectures(data, tmp_path)
    assert not report.ok
    assert any("唯一" in error for error in report.errors)
    assert any("讲义过短" in error for error in report.errors)


def test_weekly_lecture_contract_rejects_broken_local_link(tmp_path: Path) -> None:
    data = _valid_course(tmp_path)
    path = tmp_path / data["assets"][4]["lecture"]
    path.write_text(
        path.read_text(encoding="utf-8") + "\n[missing](../../not-here.ipynb)\n",
        encoding="utf-8",
    )

    report = validate_weekly_lectures(data, tmp_path)
    assert any("失效或越界本地链接" in error for error in report.errors)


def test_weekly_lecture_contract_rejects_broken_inline_repo_path(
    tmp_path: Path,
) -> None:
    data = _valid_course(tmp_path)
    path = tmp_path / data["assets"][5]["lecture"]
    path.write_text(
        path.read_text(encoding="utf-8") + "\n请打开 `docs/interactive/does-not-exist.html`。\n",
        encoding="utf-8",
    )

    report = validate_weekly_lectures(data, tmp_path)
    assert any("失效的代码样式仓库路径" in error for error in report.errors)
