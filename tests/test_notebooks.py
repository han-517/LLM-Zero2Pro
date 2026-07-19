from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_CPU_NOTEBOOKS = [
    "00_START_HERE.ipynb",
    "core/01_shapes_and_autograd.ipynb",
    "core/02_neural_language_models.ipynb",
    "core/03_tokenization_and_bpe.ipynb",
    "core/04_attention_mechanics.ipynb",
    "core/05_tiny_gpt.ipynb",
    "core/06_modern_decoder.ipynb",
    "core/07_pretraining_systems.ipynb",
    "core/08_attention_frontiers.ipynb",
    "core/09_moe.ipynb",
    "core/10_posttraining.ipynb",
    "core/11_inference_serving.ipynb",
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
    path = ROOT / "notebooks" / name
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


def test_multimodal_bridge_is_valid_offline_and_executes_on_cpu() -> None:
    path, notebook = _load_valid_notebook("optional/80_multimodal_bridge.ipynb")
    _assert_code_is_offline(notebook)
    _execute_on_cpu(path, notebook)


def test_optional_gpu_notebook_is_valid_but_not_part_of_cpu_execution() -> None:
    _, notebook = _load_valid_notebook("optional/90_gpu_environment_check.ipynb")
    assert notebook.nbformat == 4


def test_core_notebooks_have_a_complete_learning_loop() -> None:
    for name in REQUIRED_CPU_NOTEBOOKS[1:]:
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
