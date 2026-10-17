"""RMSNorm in one kernel.

Statistics in float32 like `smolvla.rms_norm`, output written straight in the dtype of the matmul
that reads it, so the norm costs one pass instead of a float32 round trip and a cast.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def so101_rms_norm(X, W, Y, stride_x, stride_y, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(x * x, axis=0) / N + eps)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(Y + row * stride_y + cols, (x * rstd * w).to(Y.dtype.element_ty), mask=mask)


def rms_norm(
    x: torch.Tensor,
    w: torch.Tensor,
    eps: float,
    out_dtype: torch.dtype = torch.bfloat16,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Normalize x (..., N), float32 or bf16, by w (N,) into `out_dtype`."""
    n = x.shape[-1]
    x2 = x.reshape(-1, n)
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    y = out if out is not None else torch.empty(x2.shape, dtype=out_dtype, device=x.device)
    so101_rms_norm[(x2.shape[0],)](
        x2, w, y, x2.stride(0), y.stride(0), n, eps,
        BLOCK=triton.next_power_of_2(n), num_warps=4 if n <= 1024 else 8,
    )  # fmt: skip
    return y.view(*x.shape[:-1], n)
