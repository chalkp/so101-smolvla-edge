"""Grouped-query attention for SmolVLA's prefix and action expert, one kernel per layer.

K and V stay at their 5 kv heads in the token-major layout the projections write, (Lk, n_kv * hd),
and query head h reads kv head h // 3, so nothing is repeated or transposed. RoPE is applied to q
as it is loaded. The mask is computed rather than loaded: every mask SmolVLA uses is block-causal
with one free block, key j visible to query i iff j < causal_from or j <= i + q_offset.

    prefix (state token last)   causal_from = L - 1   q_offset = 0
    expert self-attention       causal_from = L       q_offset = L
    expert cross-attention      causal_from = Lk      (no mask)

Softmax is online in float32 and P is rounded to bf16 for the PV product, as flash attention does.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

from . import sm_count


@triton.jit
def so101_attention(Q, K, V, OUT, COS, SIN, Lq, Lk, CK, QOFF, qk_scale, sqm, skn, svn, som,
                    GROUPS: tl.constexpr, HD: tl.constexpr, ROPE: tl.constexpr,
                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):  # fmt: skip
    HALF: tl.constexpr = HD // 2
    h = tl.program_id(0)
    pid_m = tl.program_id(1)
    kvh = h // GROUPS
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    d = tl.arange(0, HALF)
    m_ok = rm[:, None] < Lq
    q_ptr = Q + rm[:, None] * sqm + h * HD + d[None, :]
    q1 = tl.load(q_ptr, mask=m_ok, other=0.0).to(tl.float32)
    q2 = tl.load(q_ptr + HALF, mask=m_ok, other=0.0).to(tl.float32)
    if ROPE:
        c = tl.load(COS + rm[:, None] * HALF + d[None, :], mask=m_ok, other=0.0)
        sn = tl.load(SIN + rm[:, None] * HALF + d[None, :], mask=m_ok, other=0.0)
        q1, q2 = q1 * c - q2 * sn, q2 * c + q1 * sn
    q1 = q1.to(K.dtype.element_ty)
    q2 = q2.to(K.dtype.element_ty)
    m_i = tl.full((BLOCK_M,), float("-inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc1 = tl.zeros((BLOCK_M, HALF), tl.float32)
    acc2 = tl.zeros((BLOCK_M, HALF), tl.float32)
    last = rm + QOFF  # the last causal key each query sees
    k_base = K + kvh * HD + d[None, :]
    v_base = V + kvh * HD + d[None, :]
    for start in range(0, Lk, BLOCK_N):
        rn = start + tl.arange(0, BLOCK_N)
        n_ok = rn[:, None] < Lk
        k1 = tl.load(k_base + rn[:, None] * skn, mask=n_ok, other=0.0)
        k2 = tl.load(k_base + rn[:, None] * skn + HALF, mask=n_ok, other=0.0)
        s = tl.dot(q1, tl.trans(k1))
        s = tl.dot(q2, tl.trans(k2), s) * qk_scale
        vis = (rn[None, :] < Lk) & ((rn[None, :] < CK) | (rn[None, :] <= last[:, None]))
        s = tl.where(vis, s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        p = tl.exp2(s - m_new[:, None])
        alpha = tl.exp2(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        v1 = tl.load(v_base + rn[:, None] * svn, mask=n_ok, other=0.0)
        v2 = tl.load(v_base + rn[:, None] * svn + HALF, mask=n_ok, other=0.0)
        pb = p.to(V.dtype.element_ty)
        acc1 = tl.dot(pb, v1, acc1 * alpha[:, None])
        acc2 = tl.dot(pb, v2, acc2 * alpha[:, None])
        m_i = m_new
    o_ptr = OUT + rm[:, None] * som + h * HD + d[None, :]
    tl.store(o_ptr, (acc1 / l_i[:, None]).to(OUT.dtype.element_ty), mask=m_ok)
    tl.store(o_ptr + HALF, (acc2 / l_i[:, None]).to(OUT.dtype.element_ty), mask=m_ok)


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    n_heads: int,
    n_kv: int,
    *,
    cos: torch.Tensor | None = None,
    sin: torch.Tensor | None = None,
    causal_from: int | None = None,
    q_offset: int = 0,
    out: torch.Tensor | None = None,
    block_m: int | None = None,
    block_n: int | None = None,
) -> torch.Tensor:
    """Attention of q's heads over k and v, written as (Lq, n_heads * hd) in k's dtype.

    q is (Lq, >= n_heads * hd) with head h in columns [h * hd, (h + 1) * hd) and any row stride, so
    it can be a slice of the fused qkv output. k and v are (Lk, n_kv * hd). cos and sin (Lq, hd / 2)
    rotate q; leave them out for no RoPE. `causal_from` and `q_offset` give the mask (see the
    module docstring); None means no mask.
    """
    lq, lk = q.shape[0], k.shape[0]
    hd = k.shape[1] // n_kv
    for name, t in (("q", q), ("k", k), ("v", v)):
        if t.stride(1) != 1:
            raise ValueError(f"{name} needs unit column stride, has {t.stride()}")
    if out is None:
        out = torch.empty(lq, n_heads * hd, dtype=k.dtype, device=q.device)
    o = out
    rope = cos is not None
    if rope:
        cos, sin = cos.reshape(lq, hd // 2), sin.reshape(lq, hd // 2)
    ck = lk if causal_from is None else causal_from
    bm, bn, warps = _blocks(lq, q.device)
    bm, bn = block_m or bm, block_n or bn
    grid = (n_heads, triton.cdiv(lq, bm))
    so101_attention[grid](
        q, k, v, o, cos if rope else q, sin if rope else q, lq, lk, ck, q_offset,
        math.log2(math.e) / math.sqrt(hd), q.stride(0), k.stride(0), v.stride(0), o.stride(0),
        GROUPS=n_heads // n_kv, HD=hd, ROPE=rope, BLOCK_M=bm, BLOCK_N=bn,
        num_warps=warps, num_stages=2,
    )  # fmt: skip
    return o


def _blocks(lq: int, device: torch.device) -> tuple[int, int, int]:
    """Query block, key block and warps, from sweeps on an RTX 3080 and an Orin Nano.

    A desktop GPU wants many small query blocks to fill its SMs. The Orin has 8, so bigger query
    blocks that reuse every K/V load win there (the expert's 50 queries in one block).
    """
    if sm_count(device.index or 0) <= 16:
        return (64, 64, 4) if lq <= 64 else (128, 32, 2)
    return (16, 128, 4) if lq <= 64 else (64, 128, 4)
