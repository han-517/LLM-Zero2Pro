from collections import Counter

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
