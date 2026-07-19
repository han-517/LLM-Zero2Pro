from collections import Counter
from pathlib import Path

import yaml

from llm_course.course import load_roadmap, validate_course_assets, validate_roadmap
from llm_course.papers import load_catalog, validate_catalog


def test_roadmap_has_48_decision_complete_weeks() -> None:
    data = load_roadmap()
    assert [week["week"] for week in data["weeks"]] == list(range(1, 49))
    assert validate_roadmap().ok


def test_required_learning_assets_exist() -> None:
    assert validate_course_assets().ok


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
