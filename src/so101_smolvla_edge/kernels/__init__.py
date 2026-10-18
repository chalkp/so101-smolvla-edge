"""Triton kernels for the CUDA engine's prefix pass and denoising loop (`kernels="triton"`).

    norm.rms_norm          RMSNorm, float32 statistics, output in the matmul dtype
    gemm.linear            skinny-M GEMM with fused residual / SiLU-gate / bias epilogues
    attention.attention    grouped-query attention over the 5 kv heads, RoPE and mask in-kernel
    rope.rope_kv           RoPE on the keys and the K/V cache write

Every kernel is deterministic, so a CUDA-graph replay gives the same bits as an eager call, and
each is tested against the torch expression it replaces (tests/test_kernels.py). Triton is
optional: without it `available()` is False and the engine keeps its torch kernels.
"""

from __future__ import annotations

import functools


@functools.cache
def available() -> bool:
    """Triton importable and a CUDA device to run it on."""
    try:
        import torch
        import triton  # noqa: F401
    except Exception:  # noqa: BLE001 (any import failure means no)
        return False
    return torch.cuda.is_available()


@functools.cache
def sm_count(device_index: int = 0) -> int:
    import torch

    return torch.cuda.get_device_properties(device_index).multi_processor_count
