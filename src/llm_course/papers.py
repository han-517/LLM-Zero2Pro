from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from datetime import date
from pathlib import Path

import yaml
from jsonschema import Draft202012Validator, FormatChecker

from llm_course.paths import PAPER_CATALOG_PATH, PAPER_GRAPH_PATH, PAPER_INBOX_PATH
from llm_course.schemas import PaperRecord, ValidationReport

REQUIRED_FIELDS = {
    "id",
    "title",
    "year",
    "version_date",
    "source_type",
    "url",
    "code_url",
    "topics",
    "tier",
    "status",
    "prerequisites",
    "claims",
    "evidence",
    "limitations",
    "reproduction",
    "relations",
    "as_of",
}
RELATION_TYPES = {"builds_on", "improves", "contrasts_with", "used_by"}
TIERS = {"core", "deep_dive", "frontier"}
STATUSES = {"unread", "pass1", "pass2", "pass3", "reproduced"}
PAPER_SCHEMA_PATH = PAPER_CATALOG_PATH.parent / "schema.json"
ARXIV_ID = re.compile(r"arxiv\.org/(?:abs|pdf)/([^/?#]+)", re.IGNORECASE)


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} 根节点必须是 mapping")
    return value


def load_catalog(path: Path = PAPER_CATALOG_PATH) -> list[PaperRecord]:
    data = _load_yaml(path)
    raw_papers = data.get("papers", [])
    if not isinstance(raw_papers, list):
        raise ValueError("papers 必须是列表")
    return [PaperRecord.from_dict(item) for item in raw_papers]


def _arxiv_key(url: str) -> str:
    match = ARXIV_ID.search(url)
    if not match:
        return ""
    return match.group(1).removesuffix(".pdf").split("v", maxsplit=1)[0]


def validate_catalog(path: Path = PAPER_CATALOG_PATH) -> ValidationReport:
    report = ValidationReport()
    try:
        data = _load_yaml(path)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        report.errors.append(f"无法读取论文目录: {exc}")
        return report

    raw_papers = data.get("papers")
    if not isinstance(raw_papers, list):
        report.errors.append("papers 必须是列表")
        return report
    try:
        schema = json.loads(PAPER_SCHEMA_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        report.errors.append(f"无法读取论文 JSON Schema: {exc}")
        return report
    validator = Draft202012Validator(schema, format_checker=FormatChecker())

    ids: set[str] = set()
    dois: set[str] = set()
    arxiv_ids: set[str] = set()
    parsed: list[PaperRecord] = []
    for index, item in enumerate(raw_papers, start=1):
        if not isinstance(item, dict):
            report.errors.append(f"第 {index} 条论文不是 mapping")
            continue
        missing = REQUIRED_FIELDS - item.keys()
        schema_errors = sorted(validator.iter_errors(item), key=lambda error: list(error.path))
        if schema_errors:
            paper_id = item.get("id", index)
            for error in schema_errors:
                location = ".".join(str(part) for part in error.path) or "<root>"
                report.errors.append(f"论文 {paper_id} Schema {location}: {error.message}")
            continue
        if missing:
            report.errors.append(f"论文 {item.get('id', index)} 缺字段: {sorted(missing)}")
            continue
        try:
            paper = PaperRecord.from_dict(item)
        except (KeyError, TypeError, ValueError) as exc:
            report.errors.append(f"论文 {item.get('id', index)} 无法解析: {exc}")
            continue
        parsed.append(paper)
        if paper.id in ids:
            report.errors.append(f"重复论文 id: {paper.id}")
        ids.add(paper.id)
        if paper.tier not in TIERS:
            report.errors.append(f"{paper.id} tier 非法: {paper.tier}")
        if paper.status not in STATUSES:
            report.errors.append(f"{paper.id} status 非法: {paper.status}")
        if not paper.url.startswith(("https://", "http://")):
            report.errors.append(f"{paper.id} URL 非法: {paper.url}")
        relation_types = {relation.type for relation in paper.relations}
        invalid_relations = relation_types - RELATION_TYPES
        if invalid_relations:
            report.errors.append(f"{paper.id} relation type 非法: {sorted(invalid_relations)}")
        if not paper.claims or not paper.evidence or not paper.limitations:
            report.errors.append(f"{paper.id} 的 claims/evidence/limitations 不能为空")
        if paper.doi:
            normalized_doi = paper.doi.lower()
            if normalized_doi in dois:
                report.errors.append(f"重复 DOI: {paper.doi}")
            dois.add(normalized_doi)
        arxiv_id = _arxiv_key(paper.url)
        if arxiv_id:
            if arxiv_id in arxiv_ids:
                report.errors.append(f"重复 arXiv ID: {arxiv_id}")
            arxiv_ids.add(arxiv_id)

    for paper in parsed:
        for relation in paper.relations:
            if relation.target not in ids:
                report.errors.append(f"{paper.id} 关系指向未知论文: {relation.target}")

    declared = data.get("metadata", {}).get("count")
    if declared is not None and declared != len(raw_papers):
        report.errors.append(f"metadata.count={declared}，实际论文数={len(raw_papers)}")
    core_count = sum(paper.tier == "core" for paper in parsed)
    if core_count < 20:
        report.warnings.append(f"Core 论文只有 {core_count} 篇，建议保持约 25 篇")
    return report


def check_links(papers: Iterable[PaperRecord], timeout: float = 8.0) -> ValidationReport:
    report = ValidationReport()
    for paper in papers:
        head_request = urllib.request.Request(
            paper.url, method="HEAD", headers={"User-Agent": "llm-course/0.1"}
        )
        head_error: Exception | None = None
        try:
            with urllib.request.urlopen(head_request, timeout=timeout) as response:
                if response.status >= 400:
                    report.warnings.append(f"{paper.id} 链接返回 HTTP {response.status}")
        except (urllib.error.URLError, TimeoutError) as exc:
            head_error = exc
        if head_error is None:
            continue
        get_request = urllib.request.Request(
            paper.url,
            method="GET",
            headers={
                "User-Agent": "llm-course/0.1",
                "Range": "bytes=0-1023",
                "Connection": "close",
            },
        )
        try:
            with urllib.request.urlopen(get_request, timeout=timeout) as response:
                if response.status >= 400:
                    report.warnings.append(f"{paper.id} GET 回退返回 HTTP {response.status}")
        except (urllib.error.URLError, TimeoutError) as get_error:
            report.warnings.append(f"{paper.id} 链接检查失败: HEAD={head_error}; GET={get_error}")
    return report


def _normalize_arxiv_entry(entry: ET.Element, namespace: dict[str, str], as_of: str) -> dict:
    raw_id = entry.findtext("atom:id", default="", namespaces=namespace)
    arxiv_id = raw_id.rstrip("/").split("/")[-1]
    title = " ".join(entry.findtext("atom:title", default="", namespaces=namespace).split())
    summary = " ".join(entry.findtext("atom:summary", default="", namespaces=namespace).split())
    published = entry.findtext("atom:published", default=as_of, namespaces=namespace)[:10]
    authors = [
        node.findtext("atom:name", default="", namespaces=namespace)
        for node in entry.findall("atom:author", namespace)
    ]
    return {
        "id": f"candidate-{arxiv_id.replace('.', '-')}",
        "arxiv_id": arxiv_id,
        "title": title,
        "authors": authors,
        "published": published,
        "url": f"https://arxiv.org/abs/{arxiv_id}",
        "summary": summary,
        "tier": "frontier",
        "decision": "pending",
        "discovered_on": as_of,
        "note": "候选项：需人工核验技术贡献、证据、代码和与课程关系。",
    }


def fetch_arxiv_candidates(max_results: int = 20, query: str | None = None) -> list[dict]:
    if query:
        terms = [term for term in re.findall(r"[a-z0-9-]+", query.casefold()) if len(term) >= 4][
            :10
        ]
        search_query = "(" + " OR ".join(f'all:"{term}"' for term in terms) + ")"
    else:
        search_query = (
            '(all:"mixture of experts" OR all:"efficient attention" OR all:"language model")'
        )
    parameters = urllib.parse.urlencode(
        {
            "search_query": search_query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
    )
    url = f"https://export.arxiv.org/api/query?{parameters}"
    request = urllib.request.Request(url, headers={"User-Agent": "llm-course/0.1"})
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = response.read()
    root = ET.fromstring(payload)
    namespace = {"atom": "http://www.w3.org/2005/Atom"}
    today = date.today().isoformat()
    return [
        _normalize_arxiv_entry(entry, namespace, today)
        for entry in root.findall("atom:entry", namespace)
    ]


def update_inbox(max_results: int = 20, path: Path = PAPER_INBOX_PATH) -> tuple[int, int]:
    candidates = fetch_arxiv_candidates(max_results=max_results)
    if path.exists():
        data = _load_yaml(path)
    else:
        data = {"metadata": {}, "candidates": []}
    existing = data.get("candidates", [])
    if not isinstance(existing, list):
        raise ValueError("inbox candidates 必须是列表")
    known = {item.get("arxiv_id") for item in existing if isinstance(item, dict)}
    added = [item for item in candidates if item["arxiv_id"] not in known]
    data["metadata"] = {
        "updated_on": date.today().isoformat(),
        "source": "arXiv API",
        "policy": "只进入候选池；人工确认后才可加入 catalog.yaml。",
    }
    data["candidates"] = added + existing
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(data, handle, allow_unicode=True, sort_keys=False, width=100)
    return len(added), len(data["candidates"])


def generate_graph(papers: Iterable[PaperRecord], output_path: Path = PAPER_GRAPH_PATH) -> Path:
    paper_list = list(papers)
    lines = [
        "# 论文关系图",
        "",
        "> 此文件由 `llm-course papers graph` 生成。箭头从新工作指向它依赖或改进的工作。",
        "",
        "```mermaid",
        "flowchart LR",
    ]
    for paper in paper_list:
        label = paper.title.replace('"', "'")
        lines.append(f'  {paper.id}["{label}"]')
    edge_style = {
        "builds_on": "-->|builds on|",
        "improves": "-->|improves|",
        "contrasts_with": "-.->|contrasts|",
        "used_by": "-->|used by|",
    }
    for paper in paper_list:
        for relation in paper.relations:
            lines.append(f"  {paper.id} {edge_style[relation.type]} {relation.target}")
    lines.extend(["```", ""])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path
