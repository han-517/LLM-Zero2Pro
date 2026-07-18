from collections import Counter

from llm_course.papers import load_catalog


def test_initial_paper_tier_targets() -> None:
    counts = Counter(paper.tier for paper in load_catalog())
    assert 20 <= counts["core"] <= 30
    assert 40 <= counts["deep_dive"] <= 60
    assert counts["frontier"] >= 1
