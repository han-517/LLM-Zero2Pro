import urllib.error
from pathlib import Path

import pytest
import yaml

import llm_course.papers as papers_module
from llm_course.papers import check_links, load_catalog, validate_catalog
from llm_course.schemas import PaperRecord

SOURCE_TYPES = (
    "peer_reviewed",
    "preprint",
    "technical_report",
    "model_card",
    "official_blog",
    "documentation",
    "code_repository",
)


def valid_record() -> dict:
    return {
        "id": "paper_one",
        "title": "A Valid Paper Record",
        "year": 2024,
        "version_date": "2024-01-02",
        "source_type": "preprint",
        "url": "https://example.com/paper",
        "code_url": "",
        "topics": ["attention"],
        "tier": "frontier",
        "status": "unread",
        "prerequisites": [],
        "claims": ["A falsifiable claim."],
        "evidence": ["A named experiment."],
        "limitations": ["A stated limitation."],
        "reproduction": "conceptual",
        "relations": [],
        "as_of": "2026-07-19",
    }


def write_catalog(tmp_path: Path, record: object) -> Path:
    path = tmp_path / "catalog.yaml"
    path.write_text(
        yaml.safe_dump(
            {"metadata": {"count": 1}, "papers": [record]},
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize("source_type", SOURCE_TYPES)
def test_json_schema_accepts_declared_source_types(source_type: str, tmp_path: Path) -> None:
    record = valid_record()
    record["source_type"] = source_type
    assert validate_catalog(write_catalog(tmp_path, record)).ok


@pytest.mark.parametrize("nested_relation", [False, True])
def test_json_schema_rejects_additional_properties(nested_relation: bool, tmp_path: Path) -> None:
    record = valid_record()
    if nested_relation:
        record["relations"] = [
            {"type": "builds_on", "target": "paper_one", "unsupported_note": "extra"}
        ]
    else:
        record["unsupported_note"] = "extra"

    report = validate_catalog(write_catalog(tmp_path, record))

    assert not report.ok
    assert any("Additional properties" in error for error in report.errors)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("id", "Bad ID"),
        ("year", 1899),
        ("version_date", "2024-02-30"),
        ("source_type", "personal_blog"),
        ("claims", []),
        ("url", "not a URI"),
    ],
)
def test_json_schema_rejects_invalid_paper_records(
    field: str, value: object, tmp_path: Path
) -> None:
    record = valid_record()
    record[field] = value

    report = validate_catalog(write_catalog(tmp_path, record))

    assert not report.ok
    assert any("Schema" in error for error in report.errors)


def test_catalog_rejects_non_mapping_record(tmp_path: Path) -> None:
    report = validate_catalog(write_catalog(tmp_path, "not a mapping"))
    assert not report.ok
    assert any("不是 mapping" in error for error in report.errors)


class FakeResponse:
    def __init__(self, status: int) -> None:
        self.status = status

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None


def test_link_check_falls_back_from_head_to_range_get(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paper = PaperRecord.from_dict(valid_record())
    requests: list[tuple[str, str | None, float]] = []

    def fake_urlopen(request: object, *, timeout: float) -> FakeResponse:
        method = request.get_method()
        requests.append((method, request.get_header("Range"), timeout))
        if method == "HEAD":
            raise urllib.error.URLError("HEAD not supported")
        return FakeResponse(206)

    monkeypatch.setattr(papers_module.urllib.request, "urlopen", fake_urlopen)
    report = check_links([paper], timeout=0.25)

    assert report.ok
    assert report.warnings == []
    assert requests == [
        ("HEAD", None, 0.25),
        ("GET", "bytes=0-1023", 0.25),
    ]


def test_pagedattention_does_not_claim_to_build_on_gqa() -> None:
    papers = {paper.id: paper for paper in load_catalog()}
    assert "pagedattention" in papers
    assert all(relation.target != "gqa" for relation in papers["pagedattention"].relations)
