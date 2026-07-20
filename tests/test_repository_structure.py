from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
DOCUMENT_ROOTS = (
    ROOT / "README.md",
    ROOT / "learning",
    ROOT / "setup",
    ROOT / "course",
    ROOT / "solutions",
)


def _markdown_files() -> list[Path]:
    result: list[Path] = []
    for root in DOCUMENT_ROOTS:
        if root.is_file():
            result.append(root)
        elif root.is_dir():
            result.extend(root.rglob("*.md"))
    return sorted(result)


def test_all_local_markdown_links_resolve() -> None:
    broken: list[str] = []
    for path in _markdown_files():
        text = path.read_text(encoding="utf-8")
        for raw_target in MARKDOWN_LINK.findall(text):
            target = raw_target.strip().strip("<>")
            if target.startswith(("http://", "https://", "mailto:", "#")) or any(
                marker in target for marker in "[]"
            ):
                continue
            local_target = unquote(target.split("#", maxsplit=1)[0])
            if not local_target:
                continue
            resolved = (path.parent / local_target).resolve()
            if not resolved.is_relative_to(ROOT.resolve()) or not resolved.exists():
                broken.append(f"{path.relative_to(ROOT)} -> {raw_target}")
    assert not broken, "失效本地 Markdown 链接:\n" + "\n".join(broken)


def test_learning_assets_have_one_root_and_no_legacy_directories() -> None:
    for name in ("docs", "notebooks", "exercises", "papers", "knowledge", "examples"):
        assert not (ROOT / name).exists(), name
    catalog = ROOT / "learning" / "README.md"
    assert catalog.is_file()
    assert catalog.read_text(encoding="utf-8").count("### 第 ") == 48


def test_every_required_notebook_has_a_vscode_percent_pair() -> None:
    labs = ROOT / "learning" / "labs"
    notebooks = sorted(path for path in labs.glob("*.ipynb"))
    assert len(notebooks) == 11
    for notebook in notebooks:
        python_lab = notebook.with_suffix(".py")
        assert python_lab.is_file()
        assert "# %%" in python_lab.read_text(encoding="utf-8")


def test_vscode_configuration_is_valid_json() -> None:
    files = (
        ROOT / "LLM-Zero2Pro.code-workspace",
        ROOT / ".vscode" / "extensions.json",
        ROOT / ".vscode" / "settings.json",
        ROOT / ".vscode" / "tasks.json",
    )
    for path in files:
        assert isinstance(json.loads(path.read_text(encoding="utf-8")), dict)
