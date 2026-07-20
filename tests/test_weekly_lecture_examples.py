import re
from pathlib import Path

import pytest

from llm_course.course import load_roadmap
from llm_course.paths import PROJECT_ROOT

PYTHON_FENCE = re.compile(r"(?ms)^(?P<fence>`{3}|~~~)python[^\n]*\n(?P<code>.*?)^(?P=fence)\s*$")


def _weekly_lectures() -> list[tuple[int, Path]]:
    assets = load_roadmap()["assets"]
    return [(week, PROJECT_ROOT / assets[week]["lesson"]) for week in range(1, 49)]


@pytest.mark.parametrize(
    ("week", "lecture_path"),
    _weekly_lectures(),
    ids=lambda value: f"week-{value:02d}" if isinstance(value, int) else None,
)
def test_weekly_lecture_python_examples_execute_offline(
    week: int,
    lecture_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every weekly lecture must contain runnable, self-contained CPU examples."""

    monkeypatch.setenv("MPLBACKEND", "Agg")
    text = lecture_path.read_text(encoding="utf-8")
    blocks = [match.group("code") for match in PYTHON_FENCE.finditer(text)]
    assert blocks, f"第 {week} 周讲义缺少可运行的 Python 示例：{lecture_path}"

    namespace: dict[str, object] = {
        "__file__": str(lecture_path),
        "__name__": f"weekly_lecture_{week:02d}",
    }
    for block_index, code in enumerate(blocks, start=1):
        compiled = compile(
            code,
            f"{lecture_path.as_posix()}::python-block-{block_index}",
            "exec",
        )
        exec(compiled, namespace)
