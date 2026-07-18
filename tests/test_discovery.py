from llm_course.discovery import _arxiv_id, deduplicate_candidates


def candidate(source: str, arxiv_id: str, title: str) -> dict:
    return {
        "source": source,
        "arxiv_id": arxiv_id,
        "url": f"https://arxiv.org/abs/{arxiv_id}",
        "doi": "",
        "title": title,
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
