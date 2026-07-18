"""Multi-source paper discovery with conservative, review-first ingestion."""

from __future__ import annotations

import json
import os
import re
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from llm_course.papers import fetch_arxiv_candidates, load_catalog
from llm_course.paths import PAPER_INBOX_PATH

SOURCES = ("arxiv", "semantic-scholar", "huggingface")
DEFAULT_QUERY = "large language model efficient attention mixture of experts"
PROFILE_QUERIES = {
    "all": DEFAULT_QUERY,
    "data": "language model pretraining data deduplication filtering mixture contamination",
    "training-systems": "language model distributed training parallelism FSDP ZeRO checkpoint",
    "attention": "language model attention RoPE FlashAttention MLA linear attention",
    "moe": "language model mixture of experts routing expert parallel",
    "posttraining": "language model post-training SFT preference DPO GRPO RLVR",
    "evaluation-safety": "language model evaluation contamination safety reward benchmark",
}
ARXIV_PATTERN = re.compile(r"(?:arXiv:|arxiv\.org/(?:abs|pdf)/)?(\d{4}\.\d{4,5})", re.I)
RELEVANCE_TERMS = {
    "language model",
    "llm",
    "transformer",
    "attention",
    "mixture of experts",
    "moe",
    "state space",
    "mamba",
    "tokenizer",
    "pretraining",
    "pre-training",
    "post-training",
    "alignment",
    "preference optimization",
    "reasoning model",
    "speculative decoding",
    "kv cache",
    "quantization",
}


def _request_json(url: str, *, headers: dict[str, str] | None = None) -> Any:
    request_headers = {"User-Agent": "llm-course/0.1 (educational paper radar)"}
    if headers:
        request_headers.update(headers)
    request = urllib.request.Request(url, headers=request_headers)
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _clean_title(value: str) -> str:
    return " ".join(value.split())


def _title_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def _arxiv_id(value: str | None) -> str:
    if not value:
        return ""
    match = ARXIV_PATTERN.search(value)
    return match.group(1) if match else ""


def _candidate_id(source: str, source_id: str) -> str:
    safe_id = re.sub(r"[^a-zA-Z0-9]+", "-", source_id).strip("-").lower()
    return f"candidate-{source}-{safe_id}"


def _candidate(
    *,
    source: str,
    source_id: str,
    title: str,
    authors: list[str],
    published: str,
    url: str,
    summary: str,
    arxiv_id: str = "",
    doi: str = "",
    code_url: str = "",
    citation_count: int | None = None,
    reference_count: int | None = None,
    topics: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "id": _candidate_id(source, source_id),
        "source": source,
        "source_id": source_id,
        "arxiv_id": arxiv_id,
        "doi": doi,
        "title": _clean_title(title),
        "authors": authors,
        "published": published,
        "url": url,
        "code_url": code_url,
        "summary": _clean_title(summary),
        "topics": topics or [],
        "citation_count": citation_count,
        "reference_count": reference_count,
        "tier": "frontier",
        "decision": "pending",
        "discovered_on": date.today().isoformat(),
        "note": "候选项：需人工核验技术贡献、证据、限制、官方代码和课程关系。",
    }


def _looks_relevant(title: str, summary: str) -> bool:
    text = f"{title} {summary}".casefold()
    return any(term in text for term in RELEVANCE_TERMS)


def _matches_query(title: str, summary: str, query: str) -> bool:
    text = f"{title} {summary}".casefold()
    terms = {term for term in re.findall(r"[a-z0-9-]+", query.casefold()) if len(term) >= 4}
    return not terms or bool(terms & set(re.findall(r"[a-z0-9-]+", text)))


def _is_on_or_after(published: str, since: str | None) -> bool:
    if since is None:
        return True
    return bool(published) and published[:10] >= since


def fetch_arxiv(max_results: int, query: str = DEFAULT_QUERY) -> list[dict[str, Any]]:
    normalized = []
    for item in fetch_arxiv_candidates(max_results=max_results, query=query):
        arxiv_id = item["arxiv_id"]
        candidate = _candidate(
            source="arxiv",
            source_id=arxiv_id,
            arxiv_id=arxiv_id,
            title=item["title"],
            authors=item["authors"],
            published=item["published"],
            url=item["url"],
            summary=item["summary"],
        )
        normalized.append(candidate)
    return normalized


def fetch_semantic_scholar(max_results: int, query: str = DEFAULT_QUERY) -> list[dict[str, Any]]:
    fields = ",".join(
        [
            "paperId",
            "title",
            "authors",
            "year",
            "publicationDate",
            "url",
            "abstract",
            "externalIds",
            "openAccessPdf",
            "citationCount",
            "referenceCount",
            "fieldsOfStudy",
        ]
    )
    parameters = urllib.parse.urlencode({"query": query, "limit": max_results, "fields": fields})
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?{parameters}"
    headers: dict[str, str] = {}
    api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
    if api_key:
        headers["x-api-key"] = api_key
    payload = _request_json(url, headers=headers)
    normalized = []
    for item in payload.get("data", []):
        external_ids = item.get("externalIds") or {}
        arxiv_id = _arxiv_id(external_ids.get("ArXiv"))
        doi = external_ids.get("DOI") or ""
        paper_url = (
            f"https://arxiv.org/abs/{arxiv_id}"
            if arxiv_id
            else item.get("url") or (item.get("openAccessPdf") or {}).get("url") or ""
        )
        title = item.get("title") or ""
        summary = item.get("abstract") or ""
        if not title or not paper_url or not _looks_relevant(title, summary):
            continue
        normalized.append(
            _candidate(
                source="semantic-scholar",
                source_id=item["paperId"],
                arxiv_id=arxiv_id,
                doi=doi,
                title=title,
                authors=[author.get("name", "") for author in item.get("authors", [])],
                published=item.get("publicationDate") or str(item.get("year") or ""),
                url=paper_url,
                summary=summary,
                citation_count=item.get("citationCount"),
                reference_count=item.get("referenceCount"),
                topics=item.get("fieldsOfStudy") or [],
            )
        )
    return normalized


def fetch_huggingface(max_results: int, query: str = DEFAULT_QUERY) -> list[dict[str, Any]]:
    api_limit = min(100, max(max_results * 4, 20))
    parameters = urllib.parse.urlencode({"p": 0, "limit": api_limit, "sort": "publishedAt"})
    headers: dict[str, str] = {}
    token = os.environ.get("HF_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    payload = _request_json(
        f"https://huggingface.co/api/daily_papers?{parameters}", headers=headers
    )
    normalized = []
    for item in payload:
        raw_paper = item.get("paper") or item
        arxiv_id = _arxiv_id(raw_paper.get("id"))
        title = raw_paper.get("title") or item.get("title") or ""
        summary = raw_paper.get("summary") or item.get("summary") or ""
        if (
            not arxiv_id
            or not title
            or not _looks_relevant(title, summary)
            or not _matches_query(title, summary, query)
        ):
            continue
        authors = raw_paper.get("authors") or []
        normalized.append(
            _candidate(
                source="huggingface",
                source_id=arxiv_id,
                arxiv_id=arxiv_id,
                title=title,
                authors=[author.get("name", "") for author in authors],
                published=(raw_paper.get("publishedAt") or item.get("publishedAt") or "")[:10],
                url=f"https://huggingface.co/papers/{arxiv_id}",
                code_url=raw_paper.get("githubRepo") or "",
                summary=summary,
                topics=raw_paper.get("ai_keywords") or [],
            )
        )
        if len(normalized) >= max_results:
            break
    return normalized


FETCHERS: dict[str, Callable[[int, str], list[dict[str, Any]]]] = {
    "arxiv": fetch_arxiv,
    "semantic-scholar": fetch_semantic_scholar,
    "huggingface": fetch_huggingface,
}


def _keys(item: dict[str, Any]) -> set[str]:
    keys = set()
    arxiv_id = _arxiv_id(item.get("arxiv_id") or item.get("url"))
    if arxiv_id:
        keys.add(f"arxiv:{arxiv_id}")
    doi = (item.get("doi") or "").casefold().strip()
    if doi:
        keys.add(f"doi:{doi}")
    title = _title_key(item.get("title") or "")
    if title:
        keys.add(f"title:{title}")
    return keys


def deduplicate_candidates(
    candidates: Iterable[dict[str, Any]],
    *,
    blocked_keys: set[str] | None = None,
) -> list[dict[str, Any]]:
    seen = set(blocked_keys or set())
    unique = []
    for item in candidates:
        item_keys = _keys(item)
        if item_keys & seen:
            continue
        item["dedup_keys"] = sorted(item_keys)
        unique.append(item)
        seen.update(item_keys)
    return unique


def _catalog_keys() -> set[str]:
    keys = set()
    for paper in load_catalog():
        keys.update(
            _keys(
                {
                    "title": paper.title,
                    "url": paper.url,
                    "doi": paper.doi,
                }
            )
        )
    return keys


def update_inbox_multisource(
    max_results: int = 20,
    *,
    source: str = "all",
    query: str | None = None,
    profile: str = "all",
    since: str | None = None,
    path: Path = PAPER_INBOX_PATH,
) -> tuple[int, int, list[str]]:
    if isinstance(max_results, bool) or not isinstance(max_results, int) or max_results < 1:
        raise ValueError("max_results 必须是正整数")
    if source != "all" and source not in SOURCES:
        raise ValueError(f"未知 source: {source!r}")
    if profile not in PROFILE_QUERIES:
        raise ValueError(f"未知 profile: {profile!r}")
    if query is None:
        resolved_query = PROFILE_QUERIES[profile]
    elif not isinstance(query, str) or not query.strip():
        raise ValueError("query 必须是非空字符串")
    else:
        resolved_query = query.strip()
    if since is not None:
        if not isinstance(since, str):
            raise TypeError("since 必须是 ISO 日期字符串")
        try:
            date.fromisoformat(since)
        except ValueError as exc:
            raise ValueError("since 必须是 YYYY-MM-DD 格式的有效日期") from exc
    selected = list(SOURCES if source == "all" else (source,))
    errors = []
    fetched: list[dict[str, Any]] = []
    succeeded = []
    for source_name in selected:
        try:
            candidates = FETCHERS[source_name](max_results, resolved_query)
            fetched.extend(
                item for item in candidates if _is_on_or_after(item.get("published", ""), since)
            )
            succeeded.append(source_name)
        except Exception as exc:  # Network/API failures must not corrupt the inbox.
            errors.append(f"{source_name}: {type(exc).__name__}: {exc}")
    if not succeeded:
        raise RuntimeError("所有论文来源均更新失败：" + "；".join(errors))

    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    else:
        data = {"metadata": {}, "candidates": []}
    existing = data.get("candidates", [])
    if not isinstance(existing, list):
        raise ValueError("inbox candidates 必须是列表")
    blocked = _catalog_keys()
    for item in existing:
        if isinstance(item, dict):
            blocked.update(_keys(item))
    additions = deduplicate_candidates(fetched, blocked_keys=blocked)
    data["metadata"] = {
        "updated_on": date.today().isoformat(),
        "sources_requested": selected,
        "sources_succeeded": succeeded,
        "source_errors": errors,
        "query": resolved_query,
        "profile": profile,
        "since": since,
        "max_results_per_source": max_results,
        "policy": "只进入候选池；按 arXiv ID、DOI、规范化标题去重；人工确认后才可加入 catalog。",
    }
    data["candidates"] = additions + existing
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(data, handle, allow_unicode=True, sort_keys=False, width=100)
    return len(additions), len(data["candidates"]), errors
