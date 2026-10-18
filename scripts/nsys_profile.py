#!/usr/bin/env python3
"""Record the engine under Nsight Systems and report where each stage's GPU time goes.

    pixi run -e profile nsys-profile bundles/standin --label baseline
    pixi run -e profile nsys-profile bundles/standin --label trt -- --vision trt
    pixi run -e profile nsys-profile --report profiles/<label>/trace.sqlite    # re-read a trace
    pixi run -e profile nsys-profile --compare profiles/<a> profiles/<b>

Everything after `--` goes to scripts/profile_engine.py, which runs under
`nsys profile --cuda-graph-trace=node` (in its `stages` mode by default). The trace is exported to
SQLite and reported into profiles/<label>/ (untracked output):

    summary.md    per stage: GPU span, busy and idle time, kernel count, time by class, top kernels
    stages.json   the same numbers, for --compare
    kernels.csv   every kernel per stage, with calls and time per iteration

A stage's NVTX range (`stage:vision`, `chunk_3`, `denoise_7`, ...) owns the CUDA API calls its
thread made inside it, and through their correlation ids the kernels those calls ran. With node
tracing every kernel of a graph replay carries its cudaGraphLaunch's id. Span runs from the first
kernel's start to the last one's end, busy is the sum of kernel durations, and the difference is
the GPU idling inside the stage (launch gaps, dependencies, host syncs).

Needs a current nsys: Ubuntu's 2023.4 silently drops cuBLAS kernels under recent drivers. On the
desktop use the pixi `profile` environment (conda-forge nsight-systems). On a Jetson use the one
JetPack ships, with /opt/nvidia/nsight-systems/<newest>/bin first on PATH.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sqlite3
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Kernel classes of the report, first match wins, most specific first. This package's own Triton
# kernels (so101_*) land in the class of what they do; "triton" is code Inductor generated.
CLASSES = [
    ("memcpy", r"^\[mem"),
    ("attention", r"flash|fmha|attention|attn|mha|softmax"),
    (
        "gemm",
        r"gemm|splitk|cutlass|xmma|nvjet|cublas|"
        r"sm\d+_.*(tensorop|s16816|h16816)|matmul|_mm_|addmm|bmm",
    ),
    ("triton", r"^triton_|^tl_"),
    ("norm", r"norm|rms|welford"),
    ("trt", r"^__myl|^trt_|^sm\d+_xmma|tensorrt|myelin"),
    ("copy/cat", r"copy|cat_|CatArray|memcpy|memset|transpose|permute|index|gather|scatter|pad"),
    ("reduce", r"reduce|sum|mean|max_|amax"),
    ("elementwise", r"elementwise|vectorized|unrolled|pointwise|silu|gelu|mul|add"),
]

# report order of the stages; others follow
ORDER = [
    "vision",
    "prefix",
    "denoise",
    "chunk",
    "preprocess",
    "siglip+connector",
    "prefix_text",
    "expert_cache",
    "denoise_step",
]


def classify(name: str) -> str:
    """Kernel class of a kernel name, "other" when no class matches."""
    low = name.lower()
    for cls, pat in CLASSES:
        if re.search(pat, low if cls != "triton" else name):
            return cls
    return "other"


def find_nsys() -> str:
    """Path of nsys; warns when it is a version that drops cuBLAS kernels."""
    nsys = shutil.which("nsys")
    if nsys is None:
        raise SystemExit(
            "nsys not found: pixi run -e profile ... (the profile environment brings "
            "Nsight Systems; on a Jetson put /opt/nvidia/nsight-systems/<newest>/bin first on PATH)"
        )
    out = subprocess.run([nsys, "--version"], capture_output=True, text=True).stdout
    m = re.search(r"(\d{4})\.(\d+)", out)
    if m and int(m.group(1)) < 2024:
        print(
            f"!! {out.strip()}: this nsys drops cuBLAS kernels under recent drivers; "
            "use the profile environment",
            file=sys.stderr,
        )
    return nsys


def record(bundle: str, out: Path, n: int, mode: str, engine_args: list[str]) -> Path:
    """Record profile_engine.py into out/trace.nsys-rep and return the exported trace.sqlite.

    Also writes out/command.txt, the engine's command line for the report's title.
    """
    nsys = find_nsys()
    out.mkdir(parents=True, exist_ok=True)
    rep = out / "trace"
    cmd = [
        nsys, "profile", "-t", "cuda,nvtx", "--cuda-graph-trace=node",
        "--capture-range=cudaProfilerApi", "--capture-range-end=stop",
        "--force-overwrite=true", "-o", str(rep),
        sys.executable, str(ROOT / "scripts" / "profile_engine.py"), bundle,
        "--mode", mode, "-n", str(n), *engine_args,
    ]  # fmt: skip
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    db = out / "trace.sqlite"
    export = [nsys, "export", "--type", "sqlite", "--force-overwrite=true", "-o", str(db)]
    subprocess.run([*export, str(rep) + ".nsys-rep"], check=True, capture_output=True)
    engine_cmd = cmd[cmd.index(sys.executable) + 1 :]
    shown = [
        "python",
        *(str(Path(x).relative_to(ROOT)) if x.startswith(str(ROOT)) else x for x in engine_cmd),
    ]
    (out / "command.txt").write_text(" ".join(shown) + "\n")
    return db


def stage_of(text: str) -> str | None:
    """Stage an NVTX range stands for, or None for a range that only groups others."""
    if text.startswith("stage:"):
        return text[6:]
    if re.fullmatch(r"chunk_\d+", text):
        return "chunk"
    if re.fullmatch(r"denoise_\d+", text):
        return "denoise_step"
    if text in ("profile",):
        return None
    return text


def analyse(db: Path) -> dict:
    """Per-stage numbers from an exported trace: median wall, span and busy time per iteration,
    kernels per iteration, time by kernel class, and every kernel sorted by time."""
    con = sqlite3.connect(db)
    push_pop_range = 59  # NVTX_EVENTS.eventType of a push/pop range
    names = dict(con.execute("SELECT id, value FROM StringIds"))
    ranges = con.execute(
        "SELECT start, end, text, globalTid FROM NVTX_EVENTS "
        "WHERE eventType = ? AND end IS NOT NULL ORDER BY start",
        (push_pop_range,),
    ).fetchall()
    calls = con.execute(
        "SELECT start, end, globalTid, correlationId FROM CUPTI_ACTIVITY_KIND_RUNTIME"
    ).fetchall()
    gpu_ops = defaultdict(list)  # correlation id -> [(start, end, name)]
    for start, end, corr, name in con.execute(
        "SELECT start, end, correlationId, demangledName FROM CUPTI_ACTIVITY_KIND_KERNEL"
    ):
        gpu_ops[corr].append((start, end, names[name]))
    tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    copy_kinds = {0: "memcpy", 1: "HtoD", 2: "DtoH", 8: "DtoD", 10: "PtoP"}  # CUPTI copyKind
    # copies and memsets count as GPU work of the range that issued them
    if "CUPTI_ACTIVITY_KIND_MEMCPY" in tables:
        for start, end, corr, kind, nbytes in con.execute(
            "SELECT start, end, correlationId, copyKind, bytes FROM CUPTI_ACTIVITY_KIND_MEMCPY"
        ):
            label = f"[memcpy {copy_kinds.get(kind, kind)} {nbytes / 2**20:.2f} MiB]"
            gpu_ops[corr].append((start, end, label))
    if "CUPTI_ACTIVITY_KIND_MEMSET" in tables:
        for start, end, corr, nbytes in con.execute(
            "SELECT start, end, correlationId, bytes FROM CUPTI_ACTIVITY_KIND_MEMSET"
        ):
            gpu_ops[corr].append((start, end, f"[memset {nbytes / 2**10:.1f} KiB]"))
    by_tid = defaultdict(list)  # thread -> [(call start, correlation id)], sorted
    for start, _end, tid, corr in calls:
        by_tid[tid].append((start, corr))
    for v in by_tid.values():
        v.sort()
    stages: dict[str, list[dict]] = defaultdict(list)
    for start, end, text, tid in ranges:
        stage = stage_of(text or "")
        if stage is None:
            continue
        ops = []
        for t, corr in by_tid.get(tid, ()):
            if t < start:
                continue
            if t > end:
                break
            ops.extend(gpu_ops.get(corr, ()))
        if not ops:
            continue
        ops.sort()
        stages[stage].append(
            {
                "wall_us": (end - start) / 1e3,
                "span_us": (max(k[1] for k in ops) - ops[0][0]) / 1e3,
                "busy_us": sum(k[1] - k[0] for k in ops) / 1e3,
                "kernels": ops,
            }
        )
    out = {}
    for stage, iters in stages.items():
        n = len(iters)
        per_name: dict[str, list[float]] = defaultdict(lambda: [0, 0.0])  # [calls, us], all iters
        for it in iters:
            for s, e, name in it["kernels"]:
                per_name[name][0] += 1
                per_name[name][1] += (e - s) / 1e3
        by_class: dict[str, float] = defaultdict(float)
        for name, (_calls, us) in per_name.items():
            by_class[classify(name)] += us / n
        out[stage] = {
            "iterations": n,
            "wall_ms": statistics.median(i["wall_us"] for i in iters) / 1e3,
            "span_ms": statistics.median(i["span_us"] for i in iters) / 1e3,
            "busy_ms": statistics.median(i["busy_us"] for i in iters) / 1e3,
            "kernels_per_iteration": statistics.median(len(i["kernels"]) for i in iters),
            "by_class_ms": {k: v / 1e3 for k, v in sorted(by_class.items(), key=lambda kv: -kv[1])},
            "top": [
                {"name": name, "class": classify(name), "calls": calls_ / n, "ms": us / n / 1e3}
                for name, (calls_, us) in sorted(per_name.items(), key=lambda kv: -kv[1][1])
            ],
        }
    return out


def short(name: str, width: int = 90) -> str:
    """Kernel name on one line, cut to `width` characters."""
    name = re.sub(r"\s+", " ", name)
    return name if len(name) <= width else name[: width - 1] + "…"


def write_report(res: dict, out: Path, title: str) -> None:
    """Write summary.md, stages.json and kernels.csv into out/."""
    stages = sorted(res, key=lambda s: ORDER.index(s) if s in ORDER else len(ORDER))
    class_names = [c for c, _ in CLASSES]
    lines = [
        f"# {title}",
        "",
        "wall: the NVTX range on the host (launch, copies, sync); "
        "span: first GPU op start to last end; "
        "busy: sum of GPU op durations; idle = span - busy. "
        "Class columns: GPU ms per iteration.",
        "",
        "| stage | wall ms | GPU span ms | busy ms | idle ms | GPU ops | "
        + " | ".join(class_names)
        + " | other |",
        "|---" * (7 + len(CLASSES)) + "|",
    ]
    for s in stages:
        r = res[s]
        cls = [f"{r['by_class_ms'].get(c, 0.0):.2f}" for c in [*class_names, "other"]]
        idle = r["span_ms"] - r["busy_ms"]
        lines.append(
            f"| {s} | {r['wall_ms']:.2f} | {r['span_ms']:.2f} | {r['busy_ms']:.2f} | {idle:.2f} | "
            f"{r['kernels_per_iteration']:.0f} | " + " | ".join(cls) + " |"
        )
    for s in stages:
        r = res[s]
        lines += [
            "",
            f"## {s} — top kernels per iteration ({r['iterations']} iterations)",
            "",
            "| ms | % of busy | calls | class | kernel |",
            "|---|---|---|---|---|",
        ]
        for k in r["top"][:15]:
            share = 100 * k["ms"] / r["busy_ms"]
            lines.append(
                f"| {k['ms']:.3f} | {share:.1f} | {k['calls']:.0f} | {k['class']} | "
                f"`{short(k['name'])}` |"
            )
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    summary = {
        s: {k: v for k, v in res[s].items() if k != "top"} | {"top": res[s]["top"][:30]}
        for s in stages
    }
    (out / "stages.json").write_text(json.dumps(summary, indent=1))
    with open(out / "kernels.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stage", "class", "calls_per_iteration", "ms_per_iteration", "kernel"])
        for s in stages:
            for k in res[s]["top"]:
                w.writerow([s, k["class"], f"{k['calls']:.0f}", f"{k['ms']:.4f}", k["name"]])


def compare(dirs: list[Path]) -> None:
    """Print span, busy time and kernel count per stage for several profile dirs side by side."""
    runs = [(d.name, json.loads((d / "stages.json").read_text())) for d in dirs]
    stages = [s for s in ORDER if all(s in r for _, r in runs)]
    print("| stage | " + " | ".join(f"{n} span / busy / kernels" for n, _ in runs) + " |")
    print("|---" * (1 + len(runs)) + "|")
    for s in stages:
        cells = [
            f"{r[s]['span_ms']:.2f} / {r[s]['busy_ms']:.2f} / {r[s]['kernels_per_iteration']:.0f}"
            for _, r in runs
        ]
        print(f"| {s} | " + " | ".join(cells) + " |")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "bundle", nargs="?", help="bundle to record (not needed for --report/--compare)"
    )
    ap.add_argument("--label", default="baseline", help="write the reports to profiles/<label>/")
    ap.add_argument(
        "--mode",
        default="stages",
        choices=["stages", "graph", "eager"],
        help="profile_engine.py mode",
    )
    ap.add_argument("-n", type=int, default=20, help="iterations inside the capture range")
    ap.add_argument("--report", type=Path, help="report an existing trace.sqlite, no recording")
    ap.add_argument(
        "--compare",
        type=Path,
        nargs="+",
        help="compare the stages.json of these profile dirs in one table",
    )
    argv = sys.argv[1:]
    split = argv.index("--") if "--" in argv else len(argv)
    engine_args = argv[split + 1 :]  # for profile_engine.py
    a = ap.parse_args(argv[:split])
    if a.compare:
        compare(a.compare)
        return
    if a.report:
        db, out = a.report, a.report.parent
    else:
        if not a.bundle:
            ap.error("a bundle is needed to record")
        out = ROOT / "profiles" / a.label
        db = record(a.bundle, out, a.n, a.mode, engine_args)
    res = analyse(db)
    command_file = out / "command.txt"
    command = command_file.read_text().strip() if command_file.exists() else db
    write_report(res, out, f"{out.name}: {command}")
    print((out / "summary.md").read_text().split("\n## ")[0])


if __name__ == "__main__":
    main()
