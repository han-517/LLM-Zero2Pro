"""Optional Linux/NVIDIA Triton kernel starter.

Install the hosted-GPU extras first.  The CPU course and public project checks do not import
Triton.  Fill both the kernel body and launch grid, compare output/gradients with PyTorch, then
benchmark only after correctness passes.
"""

from __future__ import annotations

try:
    import triton
    import triton.language as tl
except ImportError:  # Windows/macOS and CPU-only environments
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _matmul_kernel(
        left,
        right,
        output,
        rows: tl.constexpr,
        inner: tl.constexpr,
        columns: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        # TODO: calculate program offsets, masked K loads, FP32 accumulation and masked stores.
        pass


def triton_matmul(left, right):
    """Launch the learner's Triton matrix multiplication kernel."""

    if triton is None:
        raise RuntimeError("Triton is unavailable; use a Linux/NVIDIA hosted GPU environment")
    # TODO: validate contiguous rank-2 tensors, allocate output, choose a grid, and launch.
    raise NotImplementedError
