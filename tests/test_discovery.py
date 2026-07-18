from pathlib import Path

import pytest
import yaml

import llm_course.discovery as discovery
from llm_course.discovery import _arxiv_id, deduplicate_candidates


def candidate(
    source: str,
    arxiv_id: str,
    title: str,
    *,
    published: str = "2026-07-19",
) -> dict:
    return {
        "source": source,
        "arxiv_id": arxiv_id,
        "url": f"https://arxiv.org/abs/{arxiv_id}",
        "doi": "",
        "title": title,
        "published": published,
    }


def test_arxiv_id_normalization() -> None:
    assert _arxiv_id("https://arxiv.org/pdf/2405.04434v2") == "2405.04434"
    assert _arxiv_id("arXiv:2412.06464") == "2412.06464"


def test_deduplicate_across_sources_by_arxiv_id() -> None:
    items = [
        candidate("arxiv", "2405.04434", "DeepSeek V2"),
        candidate("huggingface", "2405.04434", "DeepSeek-V2"),
        candidate("semantic-scholar", "2412.06464", "Gated Delta Networks"),
    ]
    unique = deduplicate_candidates(items)
    assert len(unique) == 2
    assert unique[0]["source"] == "arxiv"


def test_deduplicate_by_normalized_title() -> None:
    first = candidate("arxiv", "", "Attention Is All You Need")
    second = candidate("semantic-scholar", "", "Attention: Is All You Need!")
    assert len(deduplicate_candidates([first, second])) == 1


def test_block_known_catalog_key() -> None:
    item = candidate("huggingface", "1706.03762", "Attention Is All You Need")
    unique = deduplicate_candidates([item], blocked_keys={"arxiv:1706.03762"})
    assert unique == []


def test_fetch_arxiv_forwards_query_to_arxiv_api(monkeypatch: pytest.MonkeyPatch) -> None:
    received: dict[str, object] = {}

    def fake_fetch_arxiv_candidates(*, max_results: int, query: str) -> list[dict]:
        received.update(max_results=max_results, query=query)
        return [
            {
                "arxiv_id": "2501.00001",
                "title": "A Language Model Routing Study",
                "authors": ["A. Researcher"],
                "published": "2025-01-02",
                "url": "https://arxiv.org/abs/2501.00001",
                "summary": "A language model mixture of experts study.",
            }
        ]

    monkeypatch.setattr(discovery, "fetch_arxiv_candidates", fake_fetch_arxiv_candidates)
    result = discovery.fetch_arxiv(7, query="language model expert routing")

    assert received == {"max_results": 7, "query": "language model expert routing"}
    assert result[0]["source"] == "arxiv"


def test_profile_query_and_since_filter_are_recorded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    received: list[tuple[int, str]] = []

    def fake_arxiv(max_results: int, query: str) -> list[dict]:
        received.append((max_results, query))
        return [
            candidate("arxiv", "2501.00001", "Old MoE", published="2025-01-01"),
            candidate("arxiv", "2601.00001", "New MoE", published="2026-01-01"),
        ]

    monkeypatch.setitem(discovery.FETCHERS, "arxiv", fake_arxiv)
    monkeypatch.setattr(discovery, "_catalog_keys", lambda: set())
    inbox = tmp_path / "inbox.yaml"

    added, total, errors = discovery.update_inbox_multisource(
        5,
        source="arxiv",
        profile="moe",
        since="2025-06-01",
        path=inbox,
    )

    assert received == [(5, discovery.PROFILE_QUERIES["moe"])]
    assert (added, total, errors) == (1, 1, [])
    payload = yaml.safe_load(inbox.read_text(encoding="utf-8"))
    assert payload["metadata"]["profile"] == "moe"
    assert payload["metadata"]["since"] == "2025-06-01"
    assert payload["candidates"][0]["arxiv_id"] == "2601.00001"


def test_explicit_query_overrides_profile_for_arxiv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    received: list[str] = []

    def fake_arxiv(max_results: int, query: str) -> list[dict]:
        assert max_results == 3
        received.append(query)
        return []

    monkeypatch.setitem(discovery.FETCHERS, "arxiv", fake_arxiv)
    monkeypatch.setattr(discovery, "_catalog_keys", lambda: set())
    discovery.update_inbox_multisource(
        3,
        source="arxiv",
        profile="attention",
        query="  custom retrieval query  ",
        path=tmp_path / "inbox.yaml",
    )
    assert received == ["custom retrieval query"]


@pytest.mark.parametrize(
    ("kwargs", "exception", "message"),
    [
        ({"max_results": 0}, ValueError, "max_results"),
        ({"source": "crossref"}, ValueError, "source"),
        ({"profile": "unknown"}, ValueError, "profile"),
        ({"query": "   "}, ValueError, "query"),
        ({"since": "2025-02-30"}, ValueError, "since"),
        ({"since": 20250101}, TypeError, "since"),
    ],
)
def test_update_inbox_rejects_invalid_inputs(
    kwargs: dict[str, object],
    exception: type[Exception],
    message: str,
    tmp_path: Path,
) -> None:
    with pytest.raises(exception, match=message):
        discovery.update_inbox_multisource(path=tmp_path / "inbox.yaml", **kwargs)
