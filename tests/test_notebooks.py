from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
LAB_ROOT = ROOT / "learning" / "labs"
REQUIRED_CPU_NOTEBOOKS = [
    "01_shapes_and_autograd.ipynb",
    "02_neural_language_models.ipynb",
    "03_tokenization_and_bpe.ipynb",
    "04_attention_mechanics.ipynb",
    "05_tiny_gpt.ipynb",
    "06_modern_decoder.ipynb",
    "07_pretraining_systems.ipynb",
    "08_attention_frontiers.ipynb",
    "09_moe.ipynb",
    "10_posttraining.ipynb",
    "11_inference_serving.ipynb",
]
NETWORK_OR_INSTALL_MARKERS = (
    "!pip ",
    "%pip ",
    "pip install",
    "requests.get(",
    "httpx.get(",
    "urllib.request",
    "!wget ",
    "!curl ",
    "!git clone",
)


def _load_valid_notebook(name: str):
    path = LAB_ROOT / name
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    assert notebook.nbformat == 4
    assert any(cell.cell_type == "code" for cell in notebook.cells)
    return path, notebook


def _assert_code_is_offline(notebook) -> None:
    code = "\n".join(
        "".join(cell.source) if isinstance(cell.source, list) else cell.source
        for cell in notebook.cells
        if cell.cell_type == "code"
    ).lower()
    for marker in NETWORK_OR_INSTALL_MARKERS:
        assert marker not in code


def _execute_on_cpu(path: Path, notebook) -> None:
    client = NotebookClient(
        notebook,
        timeout=90,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    )
    client.execute()


@pytest.mark.parametrize("name", REQUIRED_CPU_NOTEBOOKS)
def test_required_notebook_is_valid_offline_and_executes_on_cpu(name: str) -> None:
    path, notebook = _load_valid_notebook(name)
    _assert_code_is_offline(notebook)
    _execute_on_cpu(path, notebook)


@pytest.mark.parametrize("name", REQUIRED_CPU_NOTEBOOKS)
def test_paired_vscode_python_lab_executes_on_cpu(name: str) -> None:
    path = (LAB_ROOT / name).with_suffix(".py")
    source = path.read_text(encoding="utf-8")
    assert "# %%" in source
    compile(source, str(path), "exec")
    env = {**os.environ, "MPLBACKEND": "Agg", "PYTHONUTF8": "1"}
    completed = subprocess.run(
        [sys.executable, str(path)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=90,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_multimodal_bridge_is_valid_offline_and_executes_on_cpu() -> None:
    path, notebook = _load_valid_notebook("optional/80_multimodal_bridge.ipynb")
    _assert_code_is_offline(notebook)
    _execute_on_cpu(path, notebook)


def test_optional_gpu_notebook_is_valid_but_not_part_of_cpu_execution() -> None:
    _, notebook = _load_valid_notebook("optional/90_gpu_environment_check.ipynb")
    assert notebook.nbformat == 4
    assert (LAB_ROOT / "optional/90_gpu_environment_check.py").is_file()


def test_core_notebooks_have_a_complete_learning_loop() -> None:
    for name in REQUIRED_CPU_NOTEBOOKS:
        _, notebook = _load_valid_notebook(name)
        markdown = [cell for cell in notebook.cells if cell.cell_type == "markdown"]
        code = [cell for cell in notebook.cells if cell.cell_type == "code"]
        prose = "\n".join(cell.source for cell in markdown)
        assert len(notebook.cells) >= 12, name
        assert len(markdown) >= 7, name
        assert len(code) >= 4, name
        assert "验收" in prose or "完成断言" in prose, name
        assert "来源" in prose, name
        assert notebook.metadata.get("llm_course", {}).get("offline_cpu") is True, name