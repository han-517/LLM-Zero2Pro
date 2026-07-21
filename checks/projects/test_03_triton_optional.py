from __future__ import annotations

import importlib.util
import os

import pytest
import torch

RUN_GPU = os.environ.get("RUN_GPU_PROJECT_TESTS") == "1"
TRITON_AVAILABLE = importlib.util.find_spec("triton") is not None
pytestmark = pytest.mark.skipif(
    not RUN_GPU or not torch.cuda.is_available() or not TRITON_AVAILABLE,
    reason="set RUN_GPU_PROJECT_TESTS=1 in a Linux/NVIDIA environment with Triton",
)


def test_triton_matmul_matches_pytorch_for_partial_tiles() -> None:
    from triton_matmul import triton_matmul

    torch.manual_seed(0)
    left = torch.randn(129, 70, device="cuda", dtype=torch.float16)
    right = torch.randn(70, 113, device="cuda", dtype=torch.float16)
    expected = left @ right
    actual = triton_matmul(left, right)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
