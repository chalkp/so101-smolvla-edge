"""RoPE on the keys and the K/V cache write, one kernel per layer.

The fused q/k/v projection gives (rows, (n_heads + 2 * n_kv) * hd). This rotates the key columns in
float32, rounds them to the cache dtype and stores them, and copies the value columns, straight
into the cache rows attention reads later. Triton contracts the rotation into fused multiply-adds,
so a bf16 key can differ from `apply_rope`'s in the last bit.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def so101_rope_kv(SRC, COS, SIN, KD, VD, R, s_src, k_col, v_col, s_kd, s_vd,
                  HD: tl.constexpr, HAS_V: tl.constexpr, BLOCK_R: tl.constexpr):  # fmt: skip
    HALF: tl.constexpr = HD // 2
    pid = tl.program_id(0)
    h = tl.program_id(1)
    rows = pid * BLOCK_R + tl.arange(0, BLOCK_R)
    d = tl.arange(0, HALF)
    ok = rows[:, None] < R
    kp = SRC + rows[:, None] * s_src + k_col + h * HD + d[None, :]
    k1 = tl.load(kp, mask=ok, other=0.0).to(tl.float32)
    k2 = tl.load(kp + HALF, mask=ok, other=0.0).to(tl.float32)
    c = tl.load(COS + rows[:, None] * HALF + d[None, :], mask=ok, other=0.0)
    s = tl.load(SIN + rows[:, None] * HALF + d[None, :], mask=ok, other=0.0)
    kd = KD + rows[:, None] * s_kd + h * HD + d[None, :]
    tl.store(kd, (k1 * c - k2 * s).to(KD.dtype.element_ty), mask=ok)
    tl.store(kd + HALF, (k2 * c + k1 * s).to(KD.dtype.element_ty), mask=ok)
    if HAS_V:
        dv = tl.arange(0, HD)
        v = tl.load(SRC + rows[:, None] * s_src + v_col + h * HD + dv[None, :], mask=ok, other=0.0)
        tl.store(
            VD + rows[:, None] * s_vd + h * HD + dv[None, :], v.to(VD.dtype.element_ty), mask=ok
        )


def rope_kv(
    qkv: torch.Tensor,
    n_heads: int,
    n_kv: int,
    cos: torch.Tensor,
    sin: torch.Tensor,
    k_out: torch.Tensor,
    v_out: torch.Tensor | None = None,
) -> None:
    """Rotate the key columns of qkv into k_out and copy its value columns into v_out.

    qkv is (R, (n_heads + 2 * n_kv) * hd), k_out and v_out (R, n_kv * hd) and may be row slices of
    a larger cache, cos and sin (R, hd / 2).
    """
    r = qkv.shape[0]
    hd = qkv.shape[1] // (n_heads + 2 * n_kv)
    for name, t in (("qkv", qkv), ("k_out", k_out), ("v_out", v_out)):
        if t is not None and t.stride(-1) != 1:
            raise ValueError(f"{name} needs unit column stride")
    v = v_out if v_out is not None else k_out
    rows = 32
    so101_rope_kv[(triton.cdiv(r, rows), n_kv)](
        qkv, cos.reshape(r, hd // 2), sin.reshape(r, hd // 2), k_out, v, r,
        qkv.stride(0), n_heads * hd, (n_heads + n_kv) * hd, k_out.stride(0), v.stride(0),
        HD=hd, HAS_V=v_out is not None, BLOCK_R=rows, num_warps=4,
    )  # fmt: skip
