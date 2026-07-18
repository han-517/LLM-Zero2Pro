from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_interactive_core_concepts_has_all_required_labs() -> None:
    html = (ROOT / "docs" / "interactive" / "core-concepts.html").read_text(encoding="utf-8")
    for section_id in ("softmax", "attention", "kv-cache", "moe"):
        assert f'id="{section_id}"' in html
    assert html.count('type="range"') >= 12
    assert "fetch(" not in html
    assert "XMLHttpRequest" not in html


def test_architecture_evolution_covers_three_tracks_and_current_models() -> None:
    html = (ROOT / "docs" / "interactive" / "architecture-evolution.html").read_text(
        encoding="utf-8"
    )
    for marker in ('lane:"attention"', 'lane:"position"', 'lane:"moe"'):
        assert marker in html
    for model in ("Qwen3.5", "DeepSeek-V3.2", "Llama 4", "Gated DeltaNet"):
        assert model in html
    assert html.count('lane:"') >= 24
    assert "fetch(" not in html
    assert "XMLHttpRequest" not in html


def test_interactive_labs_are_linked_from_course_entry_points() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    core_path = (ROOT / "docs" / "core_learning_path.md").read_text(encoding="utf-8")
    assert "docs/interactive/core-concepts.html" in readme
    assert "docs/interactive/architecture-evolution.html" in readme
    assert "interactive/architecture-evolution.html" in core_path
    for anchor in ("#softmax", "#attention", "#kv-cache", "#moe"):
        assert anchor in core_path



def test_architecture_evolution_has_human_readable_source_map() -> None:
    guide = (ROOT / "docs" / "architecture_evolution.md").read_text(encoding="utf-8")
    for topic in ("RoPE", "注意力机制演化", "MoE 演化", "2026 年公开模型"):
        assert topic in guide
    assert "2026-07-18" in guide


def test_progress_template_has_all_48_weeks() -> None:
    progress = yaml.safe_load((ROOT / "progress.yaml").read_text(encoding="utf-8"))
    assert [item["week"] for item in progress["weeks"]] == list(range(1, 49))
