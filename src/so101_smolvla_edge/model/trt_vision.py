"""TensorRT engine for the vision stage (SigLIP and the pixel-shuffle connector).

The vision tower is the largest stage on the Orin, and TensorRT runs it with fused attention, fused
LayerNorm/GELU/bias and tuned tensor-core GEMMs. The engine runs on the current CUDA stream with
fixed input and output buffers, so it is captured into the same per-chunk CUDA graph as the rest of
the forward. Plans are specific to the GPU and the TensorRT version, so they are built on the robot:

    so101-smolvla-edge build-trt BUNDLE   # BUNDLE/trt/vision.<precision>.<gpu>.trt<ver>.plan

The ONNX graph is exported at opset 17, where SDPA lowers to MatMul-Softmax-MatMul, the pattern
TensorRT fuses into its attention kernel. int8 is W8A8 on the tower's and connector's linears, with
activation clips from a calibration file (wm_rank/scripts/w8a8_vision_proxy.py). bf16 is the
default because int8 needs that calibration file.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch import nn

log = logging.getLogger("so101_smolvla_edge.trt")

PRECISIONS = ("fp16", "bf16", "fp32", "int8")  # plan precisions build() accepts


def _trt():
    """Import tensorrt, or exit with a hint on where JetPack keeps it."""
    try:
        import tensorrt as trt
    except ImportError as exc:  # pragma: no cover, depends on the device
        raise SystemExit(
            "TensorRT python bindings not importable. On JetPack they live in the system Python's "
            "dist-packages; scripts/setup_jetson.sh links them into the venv."
        ) from exc
    return trt


def plan_path(bundle: str | Path, precision: str) -> Path:
    """Plan file for this bundle, precision, GPU and TensorRT version."""
    trt = _trt()
    name = torch.cuda.get_device_name().replace(" ", "_") if torch.cuda.is_available() else "cpu"
    return Path(bundle) / "trt" / f"vision.{precision}.{name}.trt{trt.__version__}.plan"


class _Vision(nn.Module):
    """`SmolVLA.encode_images` as a module's forward, for ONNX export."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, px):
        return self.m.encode_images(px)


def export_onnx(
    bundle: str | Path, out: str | Path, opset: int = 17, cams: int | None = None
) -> Path:
    """Export the bundle's vision stage, in float32, as an ONNX graph at `out`.

    `cams` is the batch the graph is traced with: every camera by default, one for the OpenVINO
    engine, which runs a camera per inference.
    """
    from .bundle import load_bundle

    model, _, _ = load_bundle(bundle, "cpu", torch.float32, "fp32")
    s = model.spec
    px = torch.rand(cams or s.n_cams, 3, s.image_size, s.image_size) * 2 - 1
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        model.encode_images(px)  # fill the LayerNorm cast cache before tracing
        torch.onnx.export(
            _Vision(model),
            (px,),
            str(out),
            input_names=["px"],
            output_names=["rows"],
            opset_version=opset,
            dynamo=False,
            do_constant_folding=True,
        )
    return out


def _linear_names(spec) -> list[tuple[str, int]]:
    """(name, output features) of every weighted MatMul in the exported graph, in graph order."""
    out = [("patch", spec.vis_dim)]
    for i in range(spec.vis_layers):
        out += [
            (f"vis.{i}.qkv", 3 * spec.vis_dim),
            (f"vis.{i}.out", spec.vis_dim),
            (f"vis.{i}.fc1", spec.vis_mlp),
            (f"vis.{i}.fc2", spec.vis_dim),
        ]
    return out + [("conn", spec.text_dim)]


def quantize_onnx(
    onnx_in: str | Path, onnx_out: str | Path, spec, act_clip: dict[str, float]
) -> Path:
    """Insert W8A8 Q/DQ nodes on every linear named in `act_clip` and save the graph to `onnx_out`.

    Weights become int8 per output channel, inputs int8 per tensor with the static clip from
    `act_clip`. The patch embedding, attention and norms stay bf16.
    """
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper

    m = onnx.load(str(onnx_in))
    inits = {i.name: i for i in m.graph.initializer}
    names = iter(_linear_names(spec))
    nodes, extra, done = [], [], []
    for node in m.graph.node:
        w = inits.get(node.input[1]) if node.op_type == "MatMul" and len(node.input) > 1 else None
        if w is None or len(w.dims) != 2:
            nodes.append(node)
            continue
        name, n_out = next(names)
        if w.dims[1] != n_out:
            raise RuntimeError(
                f"graph order changed: MatMul {node.name} has {w.dims[1]} outputs, expected {name}"
            )
        if name not in act_clip:
            nodes.append(node)
            continue
        W = numpy_helper.to_array(w).astype(np.float32)  # (in, out)
        sw = (np.abs(W).max(0) / 127.0).clip(1e-8).astype(np.float32)
        Wq = np.clip(np.round(W / sw), -127, 127).astype(np.int8)
        b = f"qdq.{name}"
        extra += [
            numpy_helper.from_array(Wq, b + ".wq"),
            numpy_helper.from_array(sw, b + ".ws"),
            numpy_helper.from_array(np.zeros_like(Wq[0]), b + ".wz"),
            numpy_helper.from_array(np.array(act_clip[name] / 127.0, np.float32), b + ".xs"),
            numpy_helper.from_array(np.array(0, np.int8), b + ".xz"),
        ]
        nodes += [
            helper.make_node("QuantizeLinear", [node.input[0], b + ".xs", b + ".xz"], [b + ".xq"]),
            helper.make_node("DequantizeLinear", [b + ".xq", b + ".xs", b + ".xz"], [b + ".xd"]),
            helper.make_node(
                "DequantizeLinear", [b + ".wq", b + ".ws", b + ".wz"], [b + ".wd"], axis=1
            ),
            helper.make_node("MatMul", [b + ".xd", b + ".wd"], list(node.output), name=node.name),
        ]
        done.append(name)
    missing = set(act_clip) - set(done)
    if missing:
        raise RuntimeError(f"no MatMul for {sorted(missing)[:4]}")
    del m.graph.node[:]
    m.graph.node.extend(nodes)
    m.graph.initializer.extend(extra)
    used = {i for n in m.graph.node for i in n.input}
    keep = [i for i in m.graph.initializer if i.name in used]
    del m.graph.initializer[:]
    m.graph.initializer.extend(keep)
    onnx.save(m, str(onnx_out))
    log.info("W8A8 Q/DQ on %d linears -> %s", len(done), onnx_out)
    return Path(onnx_out)


def build(
    onnx: str | Path,
    out: str | Path,
    precision: str = "bf16",
    workspace_gb: float = 2.0,
    opt_level: int = 3,
) -> Path:
    """Build a plan from the ONNX graph and write it to `out`.

    The network is weakly typed: fp16 and bf16 let TensorRT pick reduced-precision kernels per
    layer, falling back to fp32 where it has no faster one.
    """
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}")
    trt = _trt()
    if not hasattr(trt.BuilderFlag, "BF16"):
        raise SystemExit(
            f"TensorRT {trt.__version__} builds strongly typed networks only; this builder makes a "
            "weakly typed one from the float32 ONNX graph (TensorRT 10.x, as on JetPack 7)"
        )
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx)):
        errs = "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"ONNX parse failed: {errs}")
    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30)))
    cfg.builder_optimization_level = opt_level
    if precision == "fp16":
        cfg.set_flag(trt.BuilderFlag.FP16)
    elif precision == "bf16":
        cfg.set_flag(trt.BuilderFlag.BF16)
    elif precision == "int8":  # explicit Q/DQ graph, everything else in bf16
        cfg.set_flag(trt.BuilderFlag.INT8)
        cfg.set_flag(trt.BuilderFlag.BF16)
    plan = builder.build_serialized_network(network, cfg)
    if plan is None:
        raise RuntimeError("TensorRT build failed (see the log above)")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(bytes(plan))
    return out


class TrtVision:
    """The vision stage as a TensorRT engine, called like `SmolVLA.encode_images`.

    Takes px (n_cams, 3, S, S) float32 in [-1, 1] and returns rows (n_cams * 64, text_dim) float32.
    Input and output are fixed buffers, so the returned tensor is overwritten by the next call, and
    the call can be captured into a CUDA graph after one call outside capture, which TensorRT
    needs to finish its setup.
    """

    def __init__(
        self,
        plan: str | Path,
        n_cams: int,
        image_size: int,
        n_rows: int,
        text_dim: int,
        device: torch.device,
    ):
        trt = _trt()
        self._logger = trt.Logger(trt.Logger.WARNING)  # kept alive with the engine
        self._runtime = trt.Runtime(self._logger)
        self.engine = self._runtime.deserialize_cuda_engine(Path(plan).read_bytes())
        if self.engine is None:
            raise RuntimeError(
                f"{plan}: could not deserialise (built for another GPU or TensorRT?)"
            )
        self.ctx = self.engine.create_execution_context()
        self.px = torch.zeros(n_cams, 3, image_size, image_size, dtype=torch.float32, device=device)
        self.rows = torch.zeros(n_rows, text_dim, dtype=torch.float32, device=device)
        for name, buf in (("px", self.px), ("rows", self.rows)):
            shape = tuple(self.engine.get_tensor_shape(name))
            if shape != tuple(buf.shape):
                raise RuntimeError(f"{plan}: tensor {name} is {shape}, expected {tuple(buf.shape)}")
            self.ctx.set_tensor_address(name, buf.data_ptr())

    def __call__(self, px: torch.Tensor) -> torch.Tensor:
        """Enqueue the engine on the current CUDA stream."""
        self.px.copy_(px)
        if not self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise RuntimeError("TensorRT enqueue failed")
        return self.rows
