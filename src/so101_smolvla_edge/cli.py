"""so101-smolvla-edge: export, verify and benchmark the policy, and drive SO-101 arms with it.

`so101-smolvla-edge COMMAND --help` describes each command. `run CONFIG --sim` runs the whole
robot process on simulated servos and replayed images, without hardware.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np


def _engine(args, bundle: str | None = None):
    """Build the inference engine the command-line options describe."""
    from .model.engine import Engine

    return Engine(
        bundle or args.bundle,
        device=_torch_device(args.device),
        dtype=args.dtype,
        residual=args.residual,
        cuda_graph=not args.no_graph,
        compile="autotune" if args.autotune else args.compile,
        warmup=2,
        vision=args.vision,
        trt_precision=args.trt_precision,
    )


def _torch_device(device: str) -> str:
    """Resolve --device auto to cuda when torch sees a GPU, else cpu."""
    if device != "auto":
        return device
    # the CUDA library preload in .model has to run before torch is imported
    from . import model  # noqa: F401, I001
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _golden(bundle: str):
    """The bundle's golden observations and reference actions, and its camera names."""
    g = dict(np.load(Path(bundle) / "golden.npz"))
    cams = json.loads((Path(bundle) / "meta.json").read_text())["cameras"]
    return g, cams


# -------------------------------------------------------------------------------------------- model
def cmd_export(args) -> int:
    """Write an inference bundle from a training checkpoint."""
    from .model.bundle import export_bundle

    out = export_bundle(args.checkpoint, args.bundle, task=args.task, golden_obs=args.golden)
    print(f"bundle written to {out}")
    return 0


def cmd_verify(args) -> int:
    """Run the bundle's golden observations and compare with its reference actions.

    Exit 1 when the error against the float32 reference (or the served one, if that is all the
    bundle has) exceeds --tol. The default tolerance is tight for fp32 and loose for bf16 and fp16.
    """
    g, cams = _golden(args.bundle)
    eng = _engine(args)
    outs = np.stack(
        [
            eng.infer({c: g[f"img.{c}"][i] for c in cams}, g["state"][i], noise=g["noise"][i])
            for i in range(len(g["state"]))
        ]
    )
    tol = args.tol if args.tol is not None else (1e-4 if args.dtype == "fp32" else 3e-2)
    compile_mode = "autotune" if args.autotune else args.compile
    print(
        f"engine {args.dtype} residual={args.residual} graph={not args.no_graph} "
        f"compile={compile_mode} vision={args.vision}  ({len(outs)} observations)"
    )
    ok = True
    src = json.loads((Path(args.bundle) / "meta.json").read_text()).get("golden_source")
    if src:
        print(f"  this bundle's reference is {src}")
    refs = (
        ("actions_fp32", "reference, float32"),
        ("actions", "reference as served (bf16 weights)"),
    )
    for key, label in refs:
        if key not in g:
            continue
        d = np.abs(outs - g[key])
        print(f"  vs {label:38s} max|d| {d.max():.2e} rad  mean {d.mean():.2e}")
        if key == "actions_fp32" or "actions_fp32" not in g:
            ok &= bool(d.max() <= tol)
    print("PASS" if ok else f"FAIL (tolerance {tol:.1e} rad)")
    return 0 if ok else 1


def cmd_bench(args) -> int:
    """Time inference on the bundle's first golden observation."""
    g, cams = _golden(args.bundle)
    imgs = {c: g[f"img.{c}"][0] for c in cams}
    st = g["state"][0]
    settle = 5  # untimed runs before the timed ones
    # the CUDA library preload in .model has to run before torch is imported
    from . import model  # noqa: F401, I001
    import torch

    cuda = _torch_device(args.device) == "cuda"
    gpu = torch.cuda.get_device_name() if cuda else "CPU"
    print(
        f"torch {torch.__version__}  {gpu}  dtype={args.dtype} residual={args.residual} "
        f"graph={not args.no_graph} compile={args.compile} vision={args.vision}"
    )
    t0 = time.perf_counter()
    eng = _engine(args)
    print(f"load + warmup + capture: {time.perf_counter() - t0:.1f} s")
    for _ in range(settle):
        eng.infer(imgs, st)
    eng.timing.history.clear()
    for _ in range(args.n):
        eng.infer(imgs, st)
        if args.period:
            time.sleep(args.period)  # idle like the robot between replans; the GPU may clock down
    s = eng.timing.summary()
    print(
        f"per chunk (images in -> 50x{eng.action_dim} actions out): p50 {s['p50_ms']:.1f} ms  "
        f"p90 {s['p90_ms']:.1f}  p99 {s['p99_ms']:.1f}  max {s['max_ms']:.1f}  ({args.n} runs)"
    )
    if args.profile:
        if not cuda:
            print("--profile times CUDA events; not available on the CPU")
        else:
            prof = eng.profile(imgs, st)
            kind = "graph" if eng.graph is not None else "eager"
            print(f"stages ({kind}, GPU ms): " + "  ".join(f"{k} {v:.1f}" for k, v in prof.items()))
    if cuda:
        print(f"peak GPU memory {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB")
    return 0


def _act_clip(bundle: str, path: str | None) -> dict[str, float]:
    """Activation clips for int8 vision: `path`, else the bundle's own trt/act_clip.json."""
    path = Path(path) if path else Path(bundle) / "trt" / "act_clip.json"
    if not path.exists():
        raise SystemExit(
            f"int8 vision needs activation clips: {path} not found "
            "(wm_rank/scripts/w8a8_vision_proxy.py writes them; pass --act-clip or put it there)"
        )
    clip = json.loads(path.read_text())
    return clip.get("so101_act_clip", clip.get("clip", clip))


def cmd_build_trt(args) -> int:
    """Build a TensorRT vision plan per --precision, exporting the ONNX graph first if needed."""
    from . import model  # noqa: F401
    from .model import trt_vision

    onnx = Path(args.bundle) / "trt" / "vision.onnx"
    if not onnx.exists() or args.reexport:
        t0 = time.perf_counter()
        trt_vision.export_onnx(args.bundle, onnx)
        print(f"exported {onnx} in {time.perf_counter() - t0:.0f} s")
    for prec in args.precision:
        src = onnx
        if prec == "int8":
            from .model.smolvla import ModelSpec

            spec = ModelSpec(**json.loads((Path(args.bundle) / "meta.json").read_text())["spec"])
            clip = _act_clip(args.bundle, args.act_clip)
            src = trt_vision.quantize_onnx(onnx, onnx.with_name("vision_int8.onnx"), spec, clip)
        out = trt_vision.plan_path(args.bundle, prec)
        t0 = time.perf_counter()
        trt_vision.build(src, out, prec, workspace_gb=args.workspace_gb)
        print(f"built {out} in {time.perf_counter() - t0:.0f} s")
    return 0


# ---------------------------------------------------------------------------------------------- env
def _arch_supported(cap: tuple[int, int], archs: list[str]) -> bool:
    """Whether a torch built for `archs` has kernels for compute capability `cap`.

    An sm_XY cubin runs on any sm_XZ with Z >= Y, and compute_XY PTX JIT-compiles for any later
    GPU, so the sm_80 kernels serve the Orin's sm_87.
    """
    for a in archs:
        kind, _, num = a.partition("_")
        if not num.isdigit():
            continue
        major, minor = int(num[:-1]), int(num[-1])
        if kind == "sm" and major == cap[0] and minor <= cap[1]:
            return True
        if kind == "compute" and (major, minor) <= cap:
            return True
    return False


def cmd_doctor(args) -> int:
    """Check torch and CUDA, Triton, serial ports, cameras and group membership."""
    import glob
    import grp
    import platform

    print(f"python {platform.python_version()}  {platform.machine()}  {platform.platform()}")
    rel = Path("/etc/nv_tegra_release")
    if rel.exists():
        print("L4T:", rel.read_text().splitlines()[0])
    try:
        # the CUDA library preload in .model has to run before torch is imported
        from .model import _preload  # noqa: F401, I001
        import torch

        cuda = torch.cuda.is_available()
        print(f"torch {torch.__version__}  CUDA {torch.version.cuda}  available={cuda}")
        if cuda:
            cap = torch.cuda.get_device_capability()
            arch = f"sm_{cap[0]}{cap[1]}"
            archs = torch.cuda.get_arch_list()
            print(f"  device {torch.cuda.get_device_name()} ({arch}); wheel built for {archs}")
            if not _arch_supported(cap, archs):
                print(f"  !! this torch has no kernels for {arch}")
            x = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
            (x @ x).sum().item()
            print("  bf16 GEMM ok;", "cuDNN", torch.backends.cudnn.version())
    except Exception as exc:  # noqa: BLE001
        print(f"torch: {type(exc).__name__}: {exc}")
    try:
        import triton

        print(f"triton {triton.__version__} (torch.compile available)")
    except Exception:  # noqa: BLE001
        print("triton: not installed (no torch.compile; the CUDA graph does not need it)")
    for mod in ("cv2", "serial", "safetensors", "yaml"):
        try:
            m = __import__(mod)
            print(f"{mod} {getattr(m, '__version__', 'ok')}")
        except Exception as exc:  # noqa: BLE001
            print(f"{mod}: missing ({exc})")
    ports = glob.glob("/dev/serial/by-path/*")
    ports = ports or glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*")
    print("serial ports:", *(ports or ["none"]), sep="\n  ")
    cameras = glob.glob("/dev/v4l/by-path/*index0") or glob.glob("/dev/video*")
    print("cameras:", *(cameras or ["none"]), sep="\n  ")
    if os.geteuid() == 0:  # root (e.g. in a container) opens the devices anyway
        return 0
    groups = set()
    for gid in os.getgroups():
        try:
            groups.add(grp.getgrgid(gid).gr_name)
        except KeyError:  # a gid with no name here, e.g. one a container runtime added
            groups.add(str(gid))
    for need in ("dialout", "video"):
        if need not in groups:
            print(
                f"!! user is not in group '{need}': sudo usermod -aG {need} $USER "
                "(then log in again)"
            )
    return 0


def cmd_scan(args) -> int:
    """Ping a range of ids on one bus and print the registers of each servo that answers.

    The first servo to answer also tells whether the adapter echoes what it sends. Exit 1 when
    nothing answers.
    """
    from .robot import protocol as P
    from .robot.bus import Bus

    lo, hi = (int(x) for x in args.ids.split("-"))
    regs = (
        ("model", P.MODEL_NUMBER),
        ("pos", P.PRESENT_POSITION),
        ("torque", P.TORQUE_ENABLE),
        ("min", P.MIN_POSITION_LIMIT),
        ("max", P.MAX_POSITION_LIMIT),
        ("homing", P.HOMING_OFFSET),
        ("volt", P.PRESENT_VOLTAGE),
        ("temp", P.PRESENT_TEMPERATURE),
    )
    with Bus(args.port, args.baud, echo=None) as bus:
        echo = None
        for sid in range(lo, hi + 1):
            if echo is None:
                echo = bus.detect_echo(sid)
                if echo is None:
                    continue
                print(f"adapter echo: {'on' if echo else 'off'}")
            if not bus.ping(sid):
                continue
            vals = {name: bus.read_reg(sid, reg) for name, reg in regs}
            if vals["homing"] is not None:
                vals["homing"] = P.decode_sign_magnitude(vals["homing"], P.HOMING_OFFSET_SIGN_BIT)
            if vals["volt"] is not None:
                vals["volt"] = vals["volt"] / 10
            print(f"id {sid:3d}: " + "  ".join(f"{k} {v}" for k, v in vals.items()))
        if echo is None:
            print(f"nothing answered on {args.port} for ids {lo}-{hi}")
            return 1
    return 0


def _arms(cfg, sim=None):
    """Follower arms from the config, on the simulated buses in `sim` when given."""
    from .robot.arm import Arm
    from .robot.bus import Bus

    arms = {}
    for name, a in cfg.arms.items():
        bus = None
        if sim is not None:
            bus = Bus(sim[name].path, cfg.baud, cfg.reply_margin_s, echo=False)
        arms[name] = Arm(
            name,
            a.port,
            a.calibration,
            cfg.baud,
            "off" if sim else a.echo,
            cfg.reply_margin_s,
            cfg.servo,
            bus=bus,
            convention=a.convention,
        )
    return arms


def _leader_arms(cfg, sim=None):
    """Leader arms, named like their followers. They are only read: never configured, torque off."""
    from .robot.arm import Arm
    from .robot.bus import Bus

    arms = {}
    for name, a in cfg.leaders.items():
        bus = None
        if sim is not None:
            bus = Bus(sim["leader:" + name].path, cfg.baud, cfg.reply_margin_s, echo=False)
        arms[name] = Arm(
            name,
            a.port,
            a.calibration,
            cfg.baud,
            "off" if sim else a.echo,
            cfg.reply_margin_s,
            cfg.servo,
            bus=bus,
            convention=a.convention,
        )
    return arms


def cmd_monitor(args) -> int:
    """Print every joint's position and telemetry until Ctrl-C, leaving torque as it is."""
    from .control.config import load_config

    cfg = load_config(args.config)
    arms = _arms(cfg)
    for a in arms.values():
        a.connect(verify_calibration=cfg.control.verify_calibration, configure=False)
    clear = "\033[2J\033[H"  # clear the terminal, cursor to the top left
    try:
        while True:
            lines = []
            for a in arms.values():
                q, tel, err = a.read()
                for c, qi, t in zip(a.cal, q, tel, strict=True):
                    if t is None:
                        lines.append(f"{c.name:22s} (no reply)")
                    else:
                        flag = f"  err 0x{err[c.id]:02x}" if c.id in err else ""
                        lines.append(
                            f"{c.name:22s} {qi:+7.3f} rad  load {t.load / 10:+6.1f}%  "
                            f"{t.current:+5.2f} A  {t.temperature:3d} C  {t.voltage:4.1f} V{flag}"
                        )
            print(clear + "\n".join(lines), flush=True)
            time.sleep(1 / args.rate)
    except KeyboardInterrupt:
        pass
    finally:
        for a in arms.values():
            a.close()
    return 0


def _same(a: str, b: str) -> bool:
    """Whether two control names match, ignoring case and anything not alphanumeric."""

    def key(s: str) -> str:
        return "".join(ch for ch in s.lower() if ch.isalnum())

    return key(a) == key(b)


def cmd_cameras(args) -> int:
    """Snapshot each camera beside the simulator frame the policy was trained on.

    The reference is the bundle's golden observation, and the side-by-side shows whether mounting
    and field of view match. --controls lists the V4L2 controls instead. Exit 1 when a camera
    fails or a control was not applied.
    """
    import cv2

    from .control.cameras import Camera
    from .control.config import load_config

    cfg = load_config(args.config)
    if args.controls:
        from .control.v4l2 import list_controls

        for k, c in cfg.cameras.items():
            print(f"{k}  {c.device}")
            try:
                for x in list_controls(c.device):
                    in_cfg = any(_same(x["name"], n) for n in c.controls)
                    mark = "  <- set in the config" if in_cfg else ""
                    inactive = "  (inactive)" if x["inactive"] else ""
                    print(
                        f"  {x['name']:32s} {x['type']:5s} {x['min']:>6}..{x['max']:<6} "
                        f"default {x['default']:<6} now {x['value']}{inactive}{mark}"
                    )
            except OSError as exc:
                print(f"  cannot read controls: {exc}")
        return 0
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    ref = {}
    golden = Path(cfg.bundle) / "golden.npz"
    if golden.exists():
        data = np.load(golden)
        ref = {k[4:]: data[k][0] for k in data.files if k.startswith("img.")}
    cams = {}
    rc = 0
    for k, c in cfg.cameras.items():
        try:
            cams[k] = Camera(k, **vars(c)).start()
        except RuntimeError as exc:  # one dead camera must not hide the others
            print(f"{k}: {exc}")
            rc = 1
    try:
        time.sleep(args.seconds)
        for k, cam in cams.items():
            frame, _ = cam.latest()
            if frame is None:
                print(f"{k}: no frame from {cam.device} ({cam.failures} failed reads)")
                rc = 1
                continue
            for msg in cam.control_problems:
                print(f"{k}: control not applied: {msg}")
                rc = 1
            name = k.split(".")[-1]
            cv2.imwrite(str(out / f"{name}.jpg"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            h, w = frame.shape[:2]
            kept = min(cam.crop, cam.native[1] * w / h / cam.native[0])  # of the driver's width
            clipped = float((frame.max(axis=2) >= 250).mean()) * 100  # % of near-white pixels
            line = (
                f"{k}: driver {cam.native[0]}x{cam.native[1]} -> {w}x{h} "
                f"(keeps {kept:.2f} of the width), {cam.frames / args.seconds:.1f} fps, "
                f"mean brightness {frame.mean():.0f}, clipped white {clipped:.0f} %"
            )
            r = ref.get(k)
            if r is not None and r.shape == frame.shape:
                blend = ((frame.astype(np.uint16) + r) // 2).astype(np.uint8)
                side_by_side = np.concatenate([frame, r, blend], axis=1)
                cv2.imwrite(
                    str(out / f"{name}_vs_reference.jpg"),
                    cv2.cvtColor(side_by_side, cv2.COLOR_RGB2BGR),
                )
                line += f" | simulator reference {r.mean():.0f}"
            print(line)
    finally:
        for cam in cams.values():
            cam.stop()
    print(
        f"saved to {out}: <camera>.jpg, and <camera>_vs_reference.jpg = "
        "live | simulator reference | 50% blend"
    )
    return rc


def cmd_calibration(args) -> int:
    """Report whether each calibrated joint reaches the simulator's joint limits.

    In the default `degrees` convention a simulator angle is the same physical angle, so the swept
    range has to cover the simulator's limits or commands beyond them clamp. `stretched x` is the
    factor the other convention would scale the joint by (real degrees per simulator degree).
    """
    import math

    from .robot.calibration import load_lerobot_calibration

    header = (
        f"  {'joint':14s} {'ticks min..max':>15s} {'reach deg':>16s} {'sim deg':>14s} "
        f"{'stretched x':>11s}"
    )
    short = 0
    for f in args.files:
        cal = load_lerobot_calibration(f, "", args.convention)
        print(f"{f}  (convention: {args.convention})\n{header}")
        for c in cal:
            ends = (math.degrees(c.to_rad(c.range_min)), math.degrees(c.to_rad(c.range_max)))
            reach_lo, reach_hi = sorted(ends)
            lo, hi = c.limits_deg
            scale = (c.range_max - c.range_min) * 360.0 / 4095.0 / (hi - lo)
            miss = []
            if reach_lo > lo + args.tol:
                miss.append(f"below {reach_lo:.0f}")
            if reach_hi < hi - args.tol:
                miss.append(f"above {reach_hi:.0f}")
            short += bool(miss)
            note = f"  <-- simulator angles {' and '.join(miss)} deg clamp" if miss else ""
            print(
                f"  {c.motor:14s} {c.range_min:>6d}..{c.range_max:<6d}  "
                f"{reach_lo:>7.1f}..{reach_hi:<7.1f} {lo:>6.0f}..{hi:<6.0f} {scale:>10.3f}{note}"
            )
    if not short:
        print("every joint reaches the simulator's limits")
    else:
        print(
            f"{short} joint(s) cannot reach the simulator's limits by more than {args.tol:g} deg: "
            "re-sweep them in lerobot-calibrate (docs/real_robot.md, 'Reach and direction')"
        )
    return 0


def cmd_torque(args) -> int:
    """Switch torque on or off on every configured arm.

    On holds the measured positions and stops at an arm whose positions are unreadable (exit 1).
    Off lets the arms fall, so it asks for confirmation unless --yes.
    """
    from .control.config import load_config

    cfg = load_config(args.config)
    arms = _arms(cfg)
    if args.state == "off" and not args.yes:
        if input("Torque off: the arms will drop. Support them, then type YES: ").strip() != "YES":
            print("cancelled")
            return 1
    for a in arms.values():
        a.connect(verify_calibration=cfg.control.verify_calibration, configure=False)
        if args.state == "on":
            q, _, _ = a.read()
            if not np.isfinite(q).all():
                print(f"{a.name}: positions unreadable; not enabling")
                return 1
            a.set_torque(True, q)
        else:
            a.set_torque(False)
        print(f"{a.name}: torque {args.state}")
        a.close()
    return 0


# ---------------------------------------------------------------------------------------------- run
class FakePolicy:
    """Stand-in policy for control-path dry runs: a small sinusoid around the current state."""

    def __init__(self, n_joints: int, chunk: int = 50, delay_s: float = 0.2):
        self.n = n_joints
        self.h = chunk  # actions per chunk
        self.delay = delay_s  # s, stands in for inference time
        self.k = 0  # chunks served

    def infer(self, images, state):
        """Return a (chunk, n_joints) float32 chunk around `state` after the delay."""
        time.sleep(self.delay)
        self.k += 1
        t = np.arange(self.h)[:, None] / 30.0  # s, at 30 actions per second
        phase = 0.3 * np.arange(self.n)[None, :]  # rad, one per joint
        wave = 0.15 * np.sin(2 * np.pi * 0.25 * t + phase)  # 0.15 rad at 0.25 Hz
        return (state[None, :] + wave).astype(np.float32)


def _sim_setup(cfg, bundle_meta, golden):
    """Simulated servos and replay cameras for `run --sim`.

    Each arm and leader gets a synthetic calibration file, which replaces its calibration and port
    in `cfg`, and a ServoSim with matching limit registers, posed at the golden state. The cameras
    replay the bundle's golden images.
    """
    import tempfile

    from .control.cameras import ReplayCamera
    from .robot import protocol as P
    from .robot.calibration import MOTORS, load_lerobot_calibration
    from .robot.servo_sim import ServoSim

    tmp = Path(tempfile.mkdtemp(prefix="so101-sim-"))
    span = {"drive_mode": 0, "homing_offset": 0, "range_min": 100, "range_max": 3995}  # +-171 deg
    sims = {}
    state0 = golden["state"][0] if golden is not None else None
    followers = list(cfg.arms.items())
    leaders = [("leader:" + name, a) for name, a in cfg.leaders.items()]
    for key, a in followers + leaders:
        name = key.split(":")[-1]  # a simulated leader starts where its follower is
        calf = tmp / f"{key.replace(':', '_')}.json"
        calf.write_text(json.dumps({m: {"id": i + 1, **span} for i, m in enumerate(MOTORS)}))
        a.calibration, a.port = str(calf), "sim"
        cal = load_lerobot_calibration(calf, name + "_", a.convention)
        sim = ServoSim([c.id for c in cal])
        for c in cal:
            sim.set(c.id, P.MIN_POSITION_LIMIT, c.range_min)
            sim.set(c.id, P.MAX_POSITION_LIMIT, c.range_max)
            q = 0.0
            if state0 is not None and c.name in bundle_meta["joints"]:
                q = float(state0[bundle_meta["joints"].index(c.name)])
            sim.set_position(c.id, c.to_ticks(q))
            sim.set(c.id, P.GOAL_POSITION, c.to_ticks(q))
        sims[key] = sim
    cams = {}
    if golden is not None:
        for key in bundle_meta["cameras"]:
            cams[key] = ReplayCamera(key, golden[f"img.{key}"])
    return sims, cams


def cmd_run(args) -> int:
    """Run the robot process: arms, leaders, cameras and the policy under one Runner.

    Commands come from the interactive console, or from --commands for a scripted run. Exit 1 when
    the policy needs a camera the config does not have.
    """
    from .control.cameras import Camera
    from .control.config import as_dict, load_config
    from .control.runner import Runner
    from .control.telemetry import RunLog

    cfg = load_config(args.config)
    if args.log_dir:
        cfg.log_dir = args.log_dir
    if args.bundle:
        cfg.bundle = str(Path(args.bundle).expanduser())
    meta = json.loads((Path(cfg.bundle) / "meta.json").read_text())
    golden_path = Path(cfg.bundle) / "golden.npz"
    golden = dict(np.load(golden_path)) if golden_path.exists() else None
    sims = None
    if args.sim:
        sims, cams = _sim_setup(cfg, meta, golden)
    else:
        missing = set(meta["cameras"]) - set(cfg.cameras)
        if missing and not args.no_policy:
            need = sorted(missing)
            print(f"the policy needs cameras {need}; add them under `cameras:` in {args.config}")
            return 1
        cams = {k: Camera(k, **vars(c)) for k, c in cfg.cameras.items()}
    if args.no_policy:
        policy = None
    elif args.fake_policy:
        policy = FakePolicy(len(meta["joints"]))
    else:
        ic = cfg.inference
        if ic.compile:
            cache = Path(
                os.environ.get("TORCHINDUCTOR_CACHE_DIR")
                or ic.kernel_cache
                or Path(cfg.bundle).resolve() / "inductor"
            )
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache)
            n = sum(1 for f in cache.rglob("*") if f.is_file()) if cache.is_dir() else 0
            # a warm cache holds well over 500 files; a cold one makes this start take minutes
            cold = "" if n > 500 else "; empty or new, so this start compiles for minutes"
            print(f"compiled kernels: {cache} ({n} files){cold}", flush=True)
        t0 = time.monotonic()
        from .model.engine import Engine

        policy = Engine(
            cfg.bundle,
            dtype=ic.dtype,
            residual=ic.residual,
            cuda_graph=ic.cuda_graph,
            compile=ic.compile,
            vision=ic.vision,
            trt_precision=ic.trt_precision,
        )
        print(f"policy loaded in {time.monotonic() - t0:.1f} s")
    arms = _arms(cfg, sims)
    for a in arms.values():
        a.connect(
            verify_calibration=cfg.control.verify_calibration,
            configure=cfg.control.configure_servos,
        )
    leaders = _leader_arms(cfg, sims)
    for a in leaders.values():
        a.connect(verify_calibration=cfg.control.verify_calibration, configure=False)
        a.set_torque(False)  # a leader is moved by hand, always
    runlog = RunLog(
        cfg.log_dir, {n: a.joints for n, a in arms.items()}, as_dict(cfg) | {"argv": sys.argv}
    )
    print(f"logging to {runlog.dir}")
    recorder = None
    if cfg.record.enabled or args.record:
        from .control.recorder import Recorder

        joints = [j for a in arms.values() for j in a.joints]
        info = {
            "bundle": cfg.bundle,
            "task": meta.get("task"),
            "action_dt": cfg.inference.action_dt,
            "conventions": {n: a.convention for n, a in cfg.arms.items()},
            "sim": bool(args.sim),
        }
        recorder = Recorder(cfg.record, joints, cams, info)
        print(f"recording episodes to {recorder.session}")
    viz = None
    if cfg.viz.rerun or args.rerun:
        try:
            from .control.viz import RerunViz

            viz = RerunViz(cfg.viz, [j for a in arms.values() for j in a.joints], cams)
        except ImportError:
            print("rerun is not installed (pip install rerun-sdk): running without the live view")
    runner = Runner(
        cfg, arms, cams, policy, runlog, meta["joints"], leaders=leaders, recorder=recorder, viz=viz
    )
    if viz is not None:
        try:
            urls = viz.start(runner._record_sample)
            print("live view (open one of these on another machine):")
            for url in urls:
                print(f"  {url}")
        except Exception as exc:  # noqa: BLE001
            # a taken port or the like costs the live view, never the run
            print(f"live view not started ({type(exc).__name__}: {exc}); continuing without it")
            runner.viz = viz = None
    runner.start()
    try:
        if cfg.control.enable_on_start:
            runner.enable()
        if args.commands:
            for cmd in args.commands.split(","):
                cmd = cmd.strip()
                if cmd.startswith("wait"):
                    time.sleep(float(cmd[4:] or 1))
                    print(runner.status_line(), flush=True)
                else:
                    try:
                        runner.command(cmd)
                    except Exception as exc:  # noqa: BLE001
                        why = f"{cmd}: {type(exc).__name__}: {exc}"
                        runner.event("warn", "command_failed", why)
        else:
            runner.console()
    finally:
        runner.stop()
        if viz is not None:
            viz.close()
        runlog.close()
        for s in (sims or {}).values():
            s.close()
    print(f"logs: {runlog.dir}")
    return 0


# --------------------------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    """Parse the command line and run the subcommand; return its exit code."""
    ap = argparse.ArgumentParser(
        prog="so101-smolvla-edge",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("-v", "--verbose", action="store_true", help="log at INFO level")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def engine_opts(p):
        p.add_argument("bundle")
        p.add_argument(
            "--device",
            default="auto",
            help="cuda or cpu; auto picks cuda when there is one (on a CPU use --dtype fp32)",
        )
        p.add_argument(
            "--dtype", default="bf16", choices=["bf16", "fp16", "fp32"], help="compute dtype"
        )
        p.add_argument(
            "--residual",
            default="fp32",
            choices=["fp32", "compute"],
            help="residual streams in float32, like the reference, or in the compute dtype",
        )
        p.add_argument(
            "--no-graph", action="store_true", help="run eagerly, without the CUDA graph"
        )
        p.add_argument(
            "--compile",
            action="store_true",
            help="torch.compile each stage (needs Triton)",
        )
        p.add_argument(
            "--autotune",
            action="store_true",
            help="--compile plus GEMM autotuning of the torch prefix stage; the first start is "
            "slow, later ones use the kernel cache",
        )
        p.add_argument(
            "--vision",
            default="torch",
            choices=["torch", "trt"],
            help="vision tower in torch or as a TensorRT plan (trt: run build-trt first)",
        )
        p.add_argument(
            "--trt-precision",
            default="bf16",
            choices=["fp16", "bf16", "fp32", "int8"],
            help="precision of the TensorRT vision plan. int8 is W8A8 (build-trt --precision "
            "int8, needs the activation clips)",
        )

    p = sub.add_parser("export", help="write an inference bundle from a training checkpoint")
    p.add_argument("checkpoint")
    p.add_argument("bundle")
    p.add_argument(
        "--task",
        default="fold the garment on the table",
        help="the language instruction, fixed in the bundle",
    )
    p.add_argument(
        "--golden",
        help="observations .npz to compute golden.npz from, the reference for verify and bench "
        "(recommended)",
    )
    p.set_defaults(fn=cmd_export)
    p = sub.add_parser("verify", help="compare the engine with the bundle's reference actions")
    engine_opts(p)
    p.add_argument("--tol", type=float, help="max error in rad (default by dtype)")
    p.set_defaults(fn=cmd_verify)
    p = sub.add_parser("bench", help="measure inference latency")
    engine_opts(p)
    p.add_argument("-n", type=int, default=50, help="timed runs")
    p.add_argument("--profile", action="store_true", help="also time each stage")
    p.add_argument(
        "--period",
        type=float,
        default=0.0,
        help="idle seconds between runs, e.g. 0.5 for the robot's replan period, to show what "
        "clock scaling costs",
    )
    p.set_defaults(fn=cmd_bench)
    p = sub.add_parser("build-trt", help="build the TensorRT vision engine for this device")
    p.add_argument("bundle")
    p.add_argument(
        "--precision",
        nargs="+",
        default=["bf16"],
        choices=["fp16", "bf16", "fp32", "int8"],
        help="one plan per precision; int8 needs the activation clips",
    )
    p.add_argument(
        "--act-clip",
        help="int8: activation clip json (wm_rank/scripts/w8a8_vision_proxy.py output); "
        "default <bundle>/trt/act_clip.json",
    )
    p.add_argument("--workspace-gb", type=float, default=2.0, help="builder workspace, GiB")
    p.add_argument(
        "--reexport", action="store_true", help="re-export the ONNX graph even if present"
    )
    p.set_defaults(fn=cmd_build_trt)
    p = sub.add_parser("doctor", help="check torch, CUDA, Triton, ports and cameras")
    p.set_defaults(fn=cmd_doctor)
    p = sub.add_parser("scan", help="list the servos on one bus")
    p.add_argument("--port", required=True, help="serial port, e.g. /dev/ttyACM0")
    p.add_argument("--baud", type=int, default=1_000_000)
    p.add_argument("--ids", default="1-12", help="ids to ping, FIRST-LAST")
    p.set_defaults(fn=cmd_scan)
    p = sub.add_parser("monitor", help="read-only telemetry, torque untouched")
    p.add_argument("config")
    p.add_argument("--rate", type=float, default=5, help="screen updates per second")
    p.set_defaults(fn=cmd_monitor)
    p = sub.add_parser("cameras", help="snapshot each camera beside the simulator reference frame")
    p.add_argument("config")
    p.add_argument("--out", default="~/so101-logs/cameras", help="directory for the snapshots")
    p.add_argument(
        "--seconds",
        type=float,
        default=2.0,
        help="capture time before the snapshot (also the fps window)",
    )
    p.add_argument(
        "--controls",
        action="store_true",
        help="list each camera's V4L2 controls instead (for `controls:` in the config)",
    )
    p.set_defaults(fn=cmd_cameras)
    p = sub.add_parser(
        "calibration", help="report whether a calibration reaches the simulator's joint limits"
    )
    p.add_argument("files", nargs="+")
    p.add_argument(
        "--convention",
        default="degrees",
        choices=["degrees", "stretched"],
        help="tick-to-angle mapping, as an arm's `convention:` in the config",
    )
    p.add_argument(
        "--tol",
        type=float,
        default=5.0,
        help="degrees of the simulator's range allowed to be unreachable",
    )
    p.set_defaults(fn=cmd_calibration)
    p = sub.add_parser("torque", help="torque on or off for all configured arms")
    p.add_argument("config")
    p.add_argument("state", choices=["on", "off"])
    p.add_argument("--yes", action="store_true", help="turn torque off without asking")
    p.set_defaults(fn=cmd_torque)
    p = sub.add_parser("run", help="run the policy on the arms")
    p.add_argument("config")
    p.add_argument(
        "--sim",
        action="store_true",
        help="simulated servos and replayed golden images, no hardware",
    )
    p.add_argument(
        "--no-policy", action="store_true", help="hold only (test the control loop and safety)"
    )
    p.add_argument(
        "--fake-policy", action="store_true", help="sinusoid around the state instead of the model"
    )
    p.add_argument(
        "--commands",
        help="comma-separated console commands instead of the console, e.g. 'enable,go,wait10' "
        "(waitN sleeps N s and prints the status)",
    )
    p.add_argument(
        "--record",
        action="store_true",
        help="record episodes (frames, joints, actions, mode) to record.dir",
    )
    p.add_argument(
        "--rerun",
        action="store_true",
        help="serve a live Rerun view: cameras, the frames the model was given, plans and joints",
    )
    p.add_argument("--log-dir", help="instead of the config's log_dir")
    p.add_argument("--bundle", help="instead of the config's bundle")
    p.set_defaults(fn=cmd_run)
    args = ap.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
