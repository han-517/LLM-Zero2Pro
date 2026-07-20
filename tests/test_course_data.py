from collections import Counter
from pathlib import Path

import yaml

from llm_course.course import (
    load_roadmap,
    render_learning_path,
    validate_course_assets,
    validate_roadmap,
    write_learning_path,
)
from llm_course.papers import load_catalog, validate_catalog


def test_roadmap_and_learning_path_cover_all_48_weeks() -> None:
    course = load_roadmap()
    assert [week["week"] for week in course["weeks"]] == list(range(1, 49))
    assert validate_roadmap().ok
    assert validate_course_assets().ok

    text = render_learning_path()
    assert "# LLM 学习路径（1–48）" in text
    assert "| 1 |" in text
    assert "| 48 |" in text
    assert text.count("](weeks/") == 48
    assert "15 周" not in text


def test_write_one_learning_path(tmp_path: Path) -> None:
    target = tmp_path / "learning_path.md"
    assert write_learning_path(target) == target
    text = target.read_text(encoding="utf-8")
    assert "# LLM 学习路径（1–48）" in text
    assert text.count("](weeks/") == 48


def test_paper_catalog_meets_all_three_tier_targets() -> None:
    papers = load_catalog()
    counts = Counter(paper.tier for paper in papers)
    assert counts["core"] == 25
    assert 40 <= counts["deep_dive"] <= 60
    assert counts["frontier"] >= 1
    assert len(papers) == sum(counts.values())
    assert set(counts) == {"core", "deep_dive", "frontier"}
    assert validate_catalog().ok


def test_course_manifest_is_small_and_stage_files_are_complete() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest_path = root / "course" / "roadmap.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    assert manifest["format_version"] == 2
    assert manifest["course_file"] == "course.yaml"
    assert len(manifest["stage_files"]) == 9
    assert manifest_path.read_text(encoding="utf-8").count("\n") < 20
    for relative_path in manifest["stage_files"]:
        stage_path = manifest_path.parent / relative_path
        stage_data = yaml.safe_load(stage_path.read_text(encoding="utf-8"))
        assert stage_data["stage"]["id"]
        assert stage_data["lessons"]
        assert all("assets" in lesson for lesson in stage_data["lessons"])


def test_stage_overviews_link_every_weekly_lecture() -> None:
    data = load_roadmap()
    root = Path(__file__).resolve().parents[1]

    for stage in data["stages"]:
        lessons = [lesson for lesson in data["weeks"] if lesson["stage"] == stage["id"]]
        overview_paths = {
            reading
            for lesson in lessons
            for reading in lesson["readings"]
            if reading.startswith("docs/stages/")
        }
        assert len(overview_paths) == 1, stage["id"]
        overview = (root / overview_paths.pop()).read_text(encoding="utf-8")
        assert "完整推导、代码、反例、实验与验收" in overview
        for lesson in lessons:
            lecture_name = Path(data["assets"][lesson["week"]]["lecture"]).name
            assert f"(../weeks/{lecture_name})" in overview
