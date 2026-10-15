#!/usr/bin/env python3
r"""Run the engine for Nsight Systems, recording only the measured loop.

Normally started by scripts/nsys_profile.py, which also reads the trace back per stage. By hand:

    nsys profile -t cuda,nvtx --cuda-graph-trace=node \
        --capture-range=cudaProfilerApi --capture-range-end=stop -o prof \
        pixi run python scripts/profile_engine.py BUNDLE --mode stages

Loading, compiling, warm-up and graph capture stay outside the cudaProfilerStart/Stop range.

    --mode stages   what the robot runs, cut at the stage boundaries: the engine's vision, prefix
                    and denoise stages (compiled or TensorRT as configured), each captured
                    as its own CUDA graph and replayed inside an NVTX range `stage:<name>`, then
                    the engine's per-chunk graph inside `stage:chunk`. Node tracing tags every
                    kernel of a replay with its cudaGraphLaunch, which gives each kernel exactly
                    one stage.
    --mode graph    `Engine.infer` back to back, host copies included, one NVTX range per chunk.
    --mode eager    no graphs: an NVTX range around each stage and each denoising step, so the
                    launch gaps between kernels show as well.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("bundle")
    ap.add_argument("--mode", choices=["stages", "graph", "eager"], default="stages")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--residual", default="fp32")
    ap.add_argument("-n", type=int, default=10, help="iterations to record")
    ap.add_argument("--vision", default="torch", choices=["torch", "trt"])
    ap.add_argument("--trt-precision", default="bf16")
    ap.add_argument(
        "--compile",
        default="",
        choices=["", "default", "autotune"],
        help="torch.compile mode, off by default",
    )
    a = ap.parse_args()

    # preloads the CUDA libraries a Jetson torch needs, so it has to come before torch
    from so101_smolvla_edge.model import _preload  # noqa: F401

    # isort: split
    import torch
    from torch.cuda import nvtx

    from so101_smolvla_edge.model.engine import Engine

    g = dict(np.load(Path(a.bundle) / "golden.npz"))
    cams = json.loads((Path(a.bundle) / "meta.json").read_text())["cameras"]
    imgs = {c: g[f"img.{c}"][0] for c in cams}
    st = g["state"][0]
    eng = Engine(
        a.bundle,
        dtype=a.dtype,
        residual=a.residual,
        cuda_graph=a.mode != "eager",
        warmup=3,
        vision=a.vision,
        trt_precision=a.trt_precision,
        compile={"": False, "default": True, "autotune": "autotune"}[a.compile],
    )
    for _ in range(5):
        eng.infer(imgs, st)
    torch.cuda.synchronize()

    stage_graphs = []
    if a.mode == "stages":
        vision, prefix, denoise = eng._stages
        buf = {}

        def run_vision():
            buf["rows"] = vision()

        def run_prefix():
            return prefix(buf["rows"])

        fns = [("vision", run_vision), ("prefix", run_prefix), ("denoise", denoise)]
        with torch.no_grad():
            for name, fn in fns:
                # warm up on a side stream before capture, as CUDA graph capture wants
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    for _ in range(2):
                        fn()
                torch.cuda.current_stream().wait_stream(side)
                gr = torch.cuda.CUDAGraph()
                with torch.cuda.graph(gr):
                    fn()
                stage_graphs.append((name, gr))
            stage_graphs.append(("chunk", eng.graph))
        torch.cuda.synchronize()

    def eager_once():
        """One chunk without graphs, with an NVTX range per stage and per denoising step."""
        vision, prefix, _ = eng._stages
        with torch.no_grad():
            nvtx.range_push("vision")
            rows = vision()
            nvtx.range_pop()
            nvtx.range_push("prefix")
            prefix(rows)
            nvtx.range_pop()
            x_t = eng.noise
            for k in range(eng.spec.n_steps):
                nvtx.range_push(f"denoise_{k}")
                x_t = eng.step(k, x_t)
                nvtx.range_pop()
            eng._finish(x_t)
        torch.cuda.synchronize()

    torch.cuda.profiler.start()
    t0 = time.perf_counter()
    nvtx.range_push("profile")
    for i in range(a.n):
        if a.mode == "stages":
            for name, gr in stage_graphs:
                nvtx.range_push(f"stage:{name}")
                gr.replay()
                nvtx.range_pop()
                torch.cuda.synchronize()
        else:
            nvtx.range_push(f"chunk_{i}")
            if a.mode == "graph":
                eng.infer(imgs, st)
            else:
                eager_once()
            nvtx.range_pop()
    torch.cuda.synchronize()
    nvtx.range_pop()
    torch.cuda.profiler.stop()
    ms = (time.perf_counter() - t0) / a.n * 1e3
    print(f"{a.mode} {a.dtype}: {ms:.1f} ms per iteration over {a.n}")


if __name__ == "__main__":
    main()
