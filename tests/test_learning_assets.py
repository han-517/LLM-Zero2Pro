from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit

import yaml

ROOT = Path(__file__).resolve().parents[1]
INTERACTIVE = ROOT / "docs" / "interactive"

LAB_PAGES = {
    "foundations-lab.html": (
        "分支梯度必须求和",
        "Byte BPE",
        "Bigram",
        "MLP",
        "RNN",
        "∂f/∂x",
    ),
    "architecture-lab.html": (
        "RoPE",
        "MHA",
        "MQA",
        "GQA",
        "MLA baseline",
        "Linear state",
        "滑窗感受野",
    ),
    "training-and-alignment.html": (
        "数据过滤漏斗",
        "DDP",
        "ZeRO-3 / FSDP",
        "SFT",
        "DPO",
        "GRPO / RLVR",
    ),
    "serving-lab.html": (
        "Paged KV",
        "copy-on-write",
        "(p−q)₊",
        "TTFT",
        "TPOT",
        "bonus token",
    ),
    "multimodal-flow.html": (
        "patchify",
        "Projector + token prefix",
        "Cross-attention / resampler",
        "M-RoPE",
        "grounded",
    ),
}

ENTRY_PAGES = (
    "foundations-lab.html",
    "core-concepts.html",
    "architecture-lab.html",
    "architecture-evolution.html",
    "training-and-alignment.html",
    "serving-lab.html",
    "multimodal-flow.html",
)

DYNAMIC_REGIONS = {
    "foundations-lab.html": ("graph", "tokens", "context"),
    "architecture-lab.html": ("heads", "reach-note"),
    "training-and-alignment.html": ("data-note", "memory", "objective-flow"),
    "serving-lab.html": ("pages", "spec", "metrics"),
    "multimodal-flow.html": ("flow", "shape"),
}


class PageAuditParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.html_lang: str | None = None
        self.has_viewport = False
        self.stylesheets: list[str] = []
        self.links: list[str] = []
        self.controls: list[tuple[str, dict[str, str], bool]] = []
        self.labels_for: set[str] = set()
        self.nodes_by_id: dict[str, dict[str, str]] = {}
        self.images: list[dict[str, str]] = []
        self._label_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {key: value or "" for key, value in attrs}
        if tag == "html":
            self.html_lang = values.get("lang")
        elif tag == "meta" and values.get("name") == "viewport":
            self.has_viewport = "width=device-width" in values.get("content", "")
        elif tag == "link" and "stylesheet" in values.get("rel", "").split():
            self.stylesheets.append(values.get("href", ""))
        elif tag == "a" and "href" in values:
            self.links.append(values["href"])
        if tag == "label":
            self._label_depth += 1
            if values.get("for"):
                self.labels_for.add(values["for"])
        if tag in {"input", "select", "button"}:
            self.controls.append((tag, values, self._label_depth > 0))
        if values.get("id"):
            self.nodes_by_id[values["id"]] = values
        if tag == "svg" or values.get("role") == "img":
            self.images.append(values)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag: str) -> None:
        if tag == "label":
            self._label_depth -= 1


def read_page(name: str) -> str:
    return (INTERACTIVE / name).read_text(encoding="utf-8")


def parse_page(name: str) -> PageAuditParser:
    parser = PageAuditParser()
    parser.feed(read_page(name))
    return parser


def test_interactive_index_links_every_lab_and_local_target_exists() -> None:
    parser = parse_page("index.html")
    assert parser.html_lang == "zh-CN"
    assert parser.has_viewport
    assert parser.stylesheets == ["labs.css"]
    for page in ENTRY_PAGES:
        assert page in parser.links
    for href in parser.links:
        parsed = urlsplit(href)
        if parsed.scheme or not parsed.path or parsed.path.startswith("/"):
            continue
        assert (INTERACTIVE / parsed.path).resolve().exists(), f"入口链接失效: {href}"


def test_requested_labs_have_key_knowledge_markers() -> None:
    for page, markers in LAB_PAGES.items():
        html = read_page(page)
        for marker in markers:
            assert marker in html, f"{page} 缺少知识 marker: {marker}"


def test_all_interactive_pages_are_offline_safe() -> None:
    forbidden = (r"fetch\s*\(", r"XMLHttpRequest", r"WebSocket")
    for path in INTERACTIVE.glob("*.html"):
        html = path.read_text(encoding="utf-8")
        for pattern in forbidden:
            assert re.search(pattern, html, flags=re.IGNORECASE) is None, (
                f"{path.name} 不应在页面运行时访问网络: {pattern}"
            )


def test_lab_controls_have_accessible_names_and_dynamic_feedback() -> None:
    for page in LAB_PAGES:
        parser = parse_page(page)
        assert parser.html_lang == "zh-CN"
        assert parser.has_viewport
        assert parser.stylesheets == ["labs.css"]
        assert parser.controls, f"{page} 应包含交互控件"
        for tag, attrs, nested_in_label in parser.controls:
            control_id = attrs.get("id")
            assert control_id, f"{page} 的 {tag} 缺少 id"
            if tag == "button":
                assert attrs.get("type") == "button", f"{page} 按钮应显式 type=button"
                continue
            has_aria_name = bool(attrs.get("aria-label") or attrs.get("aria-labelledby"))
            assert nested_in_label or control_id in parser.labels_for or has_aria_name, (
                f"{page} 控件 {control_id} 没有可访问名称"
            )
        for region_id in DYNAMIC_REGIONS[page]:
            attrs = parser.nodes_by_id[region_id]
            assert attrs.get("aria-live") == "polite", (
                f"{page} 动态区域 {region_id} 应提供 aria-live=polite"
            )
        for image in parser.images:
            assert image.get("aria-label"), f"{page} role=img/SVG 缺少 aria-label"


def test_multimodal_patch_grid_is_exposed_as_an_image() -> None:
    parser = parse_page("multimodal-flow.html")
    patches = parser.nodes_by_id["patches"]
    assert patches.get("role") == "img"
    assert patches.get("aria-label") == "图像 patch 网格"


def test_shared_css_supports_320px_and_keyboard_focus() -> None:
    css = (INTERACTIVE / "labs.css").read_text(encoding="utf-8")
    breakpoints = [
        int(value) for value in re.findall(r"@media\s*\(\s*max-width\s*:\s*(\d+)px\s*\)", css)
    ]
    assert any(value >= 320 for value in breakpoints)
    for marker in (
        "*{box-sizing:border-box}",
        "grid-template-columns:repeat(auto-fit,minmax(250px,1fr))",
        "input[type=range]{width:100%}",
        "svg{width:100%",
        ".control{min-width:100%}",
        ":focus-visible",
    ):
        assert marker in css


def test_interactive_core_concepts_has_all_required_labs() -> None:
    html = read_page("core-concepts.html")
    for section_id in ("softmax", "attention", "kv-cache", "moe"):
        assert f'id="{section_id}"' in html
    assert html.count('type="range"') >= 12


def test_architecture_evolution_covers_three_tracks_and_current_models() -> None:
    html = read_page("architecture-evolution.html")
    for marker in ('lane:"attention"', 'lane:"position"', 'lane:"moe"'):
        assert marker in html
    for model in ("Qwen3.5", "DeepSeek-V3.2", "Llama 4", "Gated DeltaNet"):
        assert model in html
    assert html.count('lane:"') >= 24


def test_interactive_map_is_linked_from_readme() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "docs/interactive/index.html" in readme


def test_architecture_evolution_source_map_has_current_2026_review_date() -> None:
    guide = (ROOT / "docs" / "architecture_evolution.md").read_text(encoding="utf-8")
    for topic in ("RoPE", "注意力机制演化", "MoE 演化", "2026 年公开模型"):
        assert topic in guide
    assert "2026-07-19" in guide


def test_progress_template_has_all_48_weeks() -> None:
    progress = yaml.safe_load((ROOT / "progress.yaml").read_text(encoding="utf-8"))
    assert [item["week"] for item in progress["weeks"]] == list(range(1, 49))
