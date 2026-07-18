from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_NOTEBOOKS = [
    "00_shapes_and_autograd.ipynb",
    "01_attention_lab.ipynb",
    "02_moe_lab.ipynb",
    "03_tiny_gpt.ipynb",
]


@pytest.mark.parametrize("name", REQUIRED_NOTEBOOKS)
def test_required_notebook_executes_on_cpu(name: str) -> None:
    path = ROOT / "notebooks" / name
    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=90,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    )
    client.execute()


def test_optional_gpu_notebook_is_valid() -> None:
    path = ROOT / "notebooks" / "90_optional_gpu_check.ipynb"
    notebook = nbformat.read(path, as_version=4)
    assert notebook.nbformat == 4
    assert any(cell.cell_type == "code" for cell in notebook.cells)
