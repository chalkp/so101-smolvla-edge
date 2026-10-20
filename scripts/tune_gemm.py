#!/usr/bin/env python3
"""Sweep the Triton GEMM tiles on this GPU for the model's shapes.

    pixi run python scripts/tune_gemm.py [--shapes expert|prefix] [--top 5]
    uv run python scripts/tune_gemm.py --shapes expert      # on a Jetson

Prints, per shape, cuBLAS with the epilogue as separate ops (what `SmolVLA.velocity` runs), the
heuristic's pick (kernels.gemm.pick_config) and the best configs of the sweep, with the effective
weight bandwidth. Times are CUDA-graph replays, so launch overhead is out of all of them. The
heuristic has tiles for desktop GPUs and for Orin-class GPUs (16 SMs or fewer) and should be within
a few percent of the best; if it is not on some GPU, this is where to see it.
"""

from __future__ import annotations

import argparse
import itertools
from functools import partial

import torch
import torch.nn.functional as F
import triton
from triton.testing import do_bench_cudagraph

from so101_smolvla_edge.kernels import gemm

# shape set -> (M, [(name, K, N, epilogue)])
# expert: M = chunk = 50, exp_dim 720, 15 x 64 query heads, 5 x 64 kv heads, mlp 2048
# prefix: M = prefix_len = 200, text_dim 960, mlp 2560, plus the expert's k/v projections
SHAPES = {
    "expert": (
        50,
        [
            ("qkv (self)", 720, 1600, "none"),
            ("q (cross)", 720, 960, "none"),
            ("o + residual", 960, 720, "residual"),
            ("gate_up + silu", 720, 4096, "silu_gate"),
            ("down + residual", 2048, 720, "residual"),
        ],
    ),
    "prefix": (
        200,
        [
            ("qkv", 960, 1600, "none"),
            ("o + residual", 960, 960, "residual"),
            ("gate_up + silu", 960, 5120, "silu_gate"),
            ("down + residual", 2560, 960, "residual"),
            ("expert k / v", 320, 320, "none"),
        ],
    ),
}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--shapes", choices=list(SHAPES), default="expert")
    ap.add_argument("--m", type=int, help="rows (default: the shape set's own)")
    ap.add_argument("--top", type=int, default=3, help="sweep configs to print per shape")
    ap.add_argument("--heuristic-only", action="store_true", help="skip the sweep")
    ap.add_argument("--only", help="sweep only the shapes whose name starts with this")
    a = ap.parse_args()
    m0, shapes = SHAPES[a.shapes]
    a.m = a.m or m0
    dev = "cuda"
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    print(f"{torch.cuda.get_device_name()}, {sms} SMs, M = {a.m}")
    total = {"cublas": 0.0, "heuristic": 0.0, "best": 0.0}
    for name, k, n, epi in [sh for sh in shapes if not a.only or sh[0].startswith(a.only)]:
        x = torch.randn(a.m, k, device=dev).to(torch.bfloat16)
        w = (torch.randn(n, k, device=dev) * k**-0.5).to(torch.bfloat16)
        r = torch.randn(a.m, n if epi != "silu_gate" else n // 2, device=dev)

        def cublas(x=x, w=w, r=r, epi=epi):
            y = F.linear(x, w)
            if epi == "residual":
                return r + y
            if epi == "silu_gate":
                g, u = y.chunk(2, dim=-1)
                return F.silu(g) * u
            return y

        kw = {}  # the fused epilogue
        if epi == "residual":
            kw = {"residual": r}
        elif epi == "silu_gate":
            kw = {"silu_gate": True}
        t_cublas = do_bench_cudagraph(cublas) * 1e3
        heur = gemm.pick_config(a.m, n // 2 if epi == "silu_gate" else n, k, epi == "silu_gate")
        t_heur = do_bench_cudagraph(partial(gemm.linear, x, w, config=heur, **kw)) * 1e3
        res = [(t_heur, heur)]
        if a.m > 64:  # compute-bound: wide tiles, no split
            space = itertools.product(
                (64, 128, 256), (32, 64, 128, 256), (32, 64, 128), (1,), (4, 8), (2, 3, 4)
            )
        else:  # weight streaming: all rows in one tile, split K when the columns run out
            bm = 64 if a.m > 32 else 32 if a.m > 16 else 16
            space = itertools.product(
                (bm,), (16, 32, 64, 128), (32, 64, 128), (1, 2, 3, 4, 6), (4, 8), (2, 3, 4)
            )
        for bm, bn, bk, split, warps, stages in space:
            if (
                a.heuristic_only
                or bk * split > triton.next_power_of_2(k) * 2
                or bm * bn > 256 * 128
            ):
                continue
            cfg = gemm.Config(bm, bn, bk, split, warps, stages)
            try:
                t = do_bench_cudagraph(partial(gemm.linear, x, w, config=cfg, **kw)) * 1e3
            except Exception:  # noqa: BLE001  # some configs run out of shared memory
                continue
            res.append((t, cfg))
        res.sort(key=lambda t: t[0])
        mb = w.numel() * 2 / 1e6
        print(f"\n{name}: K={k} N={n} ({mb:.1f} MB of weights)")
        print(f"  cuBLAS + ops       {t_cublas:7.2f} us  {mb / t_cublas * 1e3:6.0f} GB/s")
        print(f"  heuristic          {t_heur:7.2f} us  {mb / t_heur * 1e3:6.0f} GB/s  {heur}")
        for t, cfg in res[: a.top]:
            print(f"  sweep              {t:7.2f} us  {mb / t * 1e3:6.0f} GB/s  {cfg}")
        total["cublas"] += t_cublas
        total["heuristic"] += t_heur
        total["best"] += res[0][0]
    print(
        f"\nall {a.shapes} shapes once: " + "  ".join(f"{k} {v:.1f} us" for k, v in total.items())
    )


if __name__ == "__main__":
    main()
