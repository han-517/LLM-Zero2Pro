from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_CPU_NOTEBOOKS = [
    "00_START_HERE.ipynb",
    "00_shapes_and_autograd.ipynb",
    "neural_lm_lab.ipynb",
    "tokenization_lab.ipynb",
    "01_attention_lab.ipynb",
    "03_tiny_gpt.ipynb",
    "modern_decoder_lab.ipynb",
    "pretraining_lab.ipynb",
    "attention_frontiers_lab.ipynb",
    "02_moe_lab.ipynb",
    "posttraining_lab.ipynb",
    "inference_serving_lab.ipynb",
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
    path, notebook = _load_valid_notebook("80_multimodal_bridge.ipynb")
    _assert_code_is_offline(notebook)
    _execute_on_cpu(path, notebook)


def test_optional_gpu_notebook_is_valid_but_not_part_of_cpu_execution() -> None:
    _, notebook = _load_valid_notebook("90_optional_gpu_check.ipynb")
    assert notebook.nbformat == 4
