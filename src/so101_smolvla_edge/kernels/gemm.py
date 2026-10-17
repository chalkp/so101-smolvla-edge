"""`y = x @ w.T` for few rows, with the epilogue fused.

At 50 rows (the action expert) the matmul is weight streaming, and cuBLAS's square tiles leave
most SMs idle. Here one CTA owns all the rows and a slice of the output columns, and K is split
across CTAs when the columns alone cannot fill the GPU, so every SM streams its share of the
weights once. The prefix pass (200 rows) is closer to compute-bound and gets wider tiles.

Epilogues work on the float32 accumulator before the one rounding to the output dtype:

    residual    y = residual + x @ w.T            o_proj, down
    silu_gate   y = silu(x @ wg.T) * (x @ wu.T)   w = [wg; wu], the fused gate_up weight
    bias        y = x @ w.T + b

Split-K stays deterministic: each split stores its float32 partial and takes a ticket from a
per-tile counter, and whichever split arrives last sums the partials in split order, applies the
epilogue and resets the counter. A CUDA-graph replay gives the same bits as an eager call.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from . import sm_count

EPI_NONE = 0
EPI_RESIDUAL = 1
EPI_SILU_GATE = 2


@triton.jit
def _epilogue(acc, acc_u, rm, rn, B, R, Y, M, N, srm, srn, sym, syn,
              HAS_BIAS: tl.constexpr, EPI: tl.constexpr):  # fmt: skip
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    if HAS_BIAS:
        acc += tl.load(B + rn, mask=rn < N, other=0.0).to(tl.float32)[None, :]
    if EPI == 1:
        r = tl.load(R + rm[:, None] * srm + rn[None, :] * srn, mask=mask, other=0.0)
        acc += r.to(tl.float32)
    if EPI == 2:
        acc = acc * tl.sigmoid(acc) * acc_u
    tl.store(Y + rm[:, None] * sym + rn[None, :] * syn, acc.to(Y.dtype.element_ty), mask=mask)


@triton.jit
def so101_gemm(X, W, B, R, Y, P, CNT, M, N, K, K_SPLIT,
               sxm, sxk, swn, swk, srm, srn, sym, syn,
               HAS_BIAS: tl.constexpr, EPI: tl.constexpr, SPLIT: tl.constexpr,
               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):  # fmt: skip
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_m = tl.program_id(2)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    k_lo = pid_k * K_SPLIT
    k_hi = tl.minimum(k_lo + K_SPLIT, K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_u = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    x_ptr = X + rm[:, None] * sxm
    w_ptr = W + rn[:, None] * swn
    m_ok = rm[:, None] < M
    n_ok = rn[:, None] < N
    for k in range(k_lo, k_hi, BLOCK_K):
        kk = k + rk
        k_ok = kk[None, :] < k_hi
        x = tl.load(x_ptr + kk[None, :] * sxk, mask=m_ok & k_ok, other=0.0)
        x = x.to(W.dtype.element_ty)
        w = tl.load(w_ptr + kk[None, :] * swk, mask=n_ok & k_ok, other=0.0)
        acc = tl.dot(x, tl.trans(w), acc)
        if EPI == 2:
            wu = tl.load(w_ptr + N * swn + kk[None, :] * swk, mask=n_ok & k_ok, other=0.0)
            acc_u = tl.dot(x, tl.trans(wu), acc_u)
    if SPLIT == 1:
        _epilogue(acc, acc_u, rm, rn, B, R, Y, M, N, srm, srn, sym, syn, HAS_BIAS, EPI)
    else:  # float32 partials, (SPLIT, 2, M, N)
        pmask = m_ok & (rn[None, :] < N)
        off = rm[:, None] * N + rn[None, :]
        tl.store(P + pid_k * 2 * M * N + off, acc, mask=pmask)
        if EPI == 2:
            tl.store(P + pid_k * 2 * M * N + M * N + off, acc_u, mask=pmask)
        tl.debug_barrier()  # every thread's partial is stored before the ticket
        tile = pid_m * tl.num_programs(0) + pid_n
        if tl.atomic_add(CNT + tile, 1, sem="acq_rel", scope="gpu") == SPLIT - 1:
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            acc_u = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for s in tl.static_range(SPLIT):  # in split order, whoever arrived last
                part = P + s * 2 * M * N + off
                acc += tl.load(part, mask=pmask, other=0.0, cache_modifier=".cg")
                if EPI == 2:
                    acc_u += tl.load(part + M * N, mask=pmask, other=0.0, cache_modifier=".cg")
            _epilogue(acc, acc_u, rm, rn, B, R, Y, M, N, srm, srn, sym, syn, HAS_BIAS, EPI)
            tl.atomic_xchg(CNT + tile, 0)  # ready for the next launch or graph replay


@dataclass(frozen=True)
class Config:
    block_m: int
    block_n: int
    block_k: int
    split: int  # CTAs along K
    num_warps: int = 4
    num_stages: int = 4


def pick_config(m: int, n: int, k: int, gate: bool = False, sms: int | None = None) -> Config:
    """Tiles for an (m x k) @ (k x n) GEMM, from a scripts/tune_gemm.py sweep on an RTX 3080."""
    return _desktop(m, n, k, gate, sms or sm_count())


def _desktop(m: int, n: int, k: int, gate: bool, sms: int) -> Config:
    if m > 64:  # the prefix: four row tiles that re-read each weight tile from L2
        if gate:
            return Config(64, 32, 32, 1, num_warps=4, num_stages=3)
        if n <= 512:
            return Config(64, 16, 64, 1, num_warps=4, num_stages=3)
        if n <= 1024 and k >= 1536:
            return Config(64, 64, 128, 1, num_warps=4, num_stages=4)
        if n <= 1024:
            return Config(64, 64, 32, 1, num_warps=4, num_stages=3)
        return Config(64, 32, 64, 1, num_warps=4, num_stages=3)
    bm = _rows(m)
    if k >= 1536:  # split K only where the column tiles leave SMs idle (the expert's down)
        bn, bk = 32, 64
        tiles = triton.cdiv(n, bn) * triton.cdiv(m, bm) * (2 if gate else 1)
        split = max(1, min(triton.cdiv(k, bk) // 8, round(sms / tiles)))
        return Config(bm, bn, bk, split, num_warps=8, num_stages=4)
    if n <= 1024:
        return Config(bm, 16, 64, 1, num_warps=4, num_stages=4)
    if gate:
        return Config(bm, 32, 128, 1, num_warps=4, num_stages=3)
    return Config(bm, 32, 32, 1, num_warps=8, num_stages=4)


def _rows(m: int) -> int:
    """Smallest row block that holds all m rows (m <= 64)."""
    return 16 if m <= 16 else 32 if m <= 32 else 64


_COUNTERS: dict[torch.device, torch.Tensor] = {}


def _counters(device: torch.device) -> torch.Tensor:
    """Split-K tickets, one per output tile and zero between launches.

    One buffer per device serves every GEMM, since they run one after another on the stream. It
    must exist before a CUDA graph is captured.
    """
    c = _COUNTERS.get(device)
    if c is None:
        c = _COUNTERS[device] = torch.zeros(1 << 16, dtype=torch.int32, device=device)
    return c


def linear(
    x: torch.Tensor,
    w: torch.Tensor,
    *,
    bias: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
    silu_gate: bool = False,
    out_dtype: torch.dtype | None = None,
    out: torch.Tensor | None = None,
    config: Config | None = None,
) -> torch.Tensor:
    """Compute x (..., K) @ w (N, K).T with a fused epilogue.

    With `silu_gate`, w is (2F, K) = [gate; up] and the result (..., F). `residual` (..., N) is
    added in float32 and the result takes its dtype unless `out_dtype` says otherwise. `out` may be
    the residual itself: each element is read before it is written, by the same thread.
    """
    if silu_gate and (residual is not None or bias is not None):
        raise ValueError("silu_gate takes neither a residual nor a bias")
    k = x.shape[-1]
    x2 = x.reshape(-1, k)
    m = x2.shape[0]
    n = w.shape[0] // 2 if silu_gate else w.shape[0]
    if w.shape[1] != k:
        raise ValueError(f"x has {k} features, w {tuple(w.shape)}")
    epi = EPI_SILU_GATE if silu_gate else EPI_RESIDUAL if residual is not None else EPI_NONE
    if out_dtype is None:
        out_dtype = residual.dtype if residual is not None else w.dtype
    if out is not None:
        y = out.view(m, n)
    else:
        y = torch.empty(m, n, dtype=out_dtype, device=x.device)
    r2 = residual.reshape(m, n) if residual is not None else y
    cfg = config or pick_config(m, n, k, silu_gate, sm_count(x.device.index or 0))
    k_split = triton.cdiv(triton.cdiv(k, cfg.block_k), cfg.split) * cfg.block_k
    split = triton.cdiv(k, k_split)
    p = torch.empty(split * 2 * m * n, dtype=torch.float32, device=x.device) if split > 1 else y
    cnt = _counters(x.device) if split > 1 else y
    b = bias if bias is not None else y
    grid = (triton.cdiv(n, cfg.block_n), split, triton.cdiv(m, cfg.block_m))
    so101_gemm[grid](
        x2, w, b, r2, y, p, cnt, m, n, k, k_split,
        x2.stride(0), x2.stride(1), w.stride(0), w.stride(1),
        r2.stride(0), r2.stride(1), y.stride(0), y.stride(1),
        HAS_BIAS=bias is not None, EPI=epi, SPLIT=split,
        BLOCK_M=cfg.block_m, BLOCK_N=cfg.block_n, BLOCK_K=cfg.block_k,
        num_warps=cfg.num_warps, num_stages=cfg.num_stages,
    )  # fmt: skip
    return y.view(*x.shape[:-1], n)
