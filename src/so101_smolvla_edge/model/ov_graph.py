"""SmolVLA as three static OpenVINO graphs, for the Intel NPU (and OpenVINO's CPU plugin).

    vision   px (1, 3, S, S) in [-1, 1], one camera    -> rows (64, text_dim)
    prefix   rows of every camera, state (max_state,)  -> K, V (n_layers, n_kv, L, head_dim)
    step     x_t (1, chunk, max_action), time (1, 1, exp_dim), K, V -> v_t

`so101-smolvla-edge build-ov BUNDLE` writes them to BUNDLE/openvino. They compute what
`smolvla.SmolVLA` computes, restated for a compiler. Building needs torch, running them
(`ov_engine.py`) does not.
"""

from __future__ import annotations

import json
import logging
import math
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .ov_engine import FORMAT, GRAPH, STAGES, ir_dir, weights_id
from .smolvla import SmolVLA, apply_rope, param_shapes, pixel_shuffle, rope_tables, sinusoidal

log = logging.getLogger("so101_smolvla_edge.ov")

NEG = -1e4  # masked score: finite, not -inf, and exp() of it is exactly 0 in float16 too
_AUG = 100.0  # sqrt(-NEG), for the prefix mask written as a product of two columns


# --------------------------------------------------------------------------------------------- math
def _rms(x: Tensor, w: Tensor, eps: float) -> Tensor:
    """RMSNorm in its plain form, which the NPU compiler recognises.

    The NPU computes in float16 and the text stream's activations square far past its range, but
    the compiler evaluates a recognised RMSNorm or LayerNorm safely, so the norms must stay in
    their plain form.
    """
    return F.rms_norm(x, (x.shape[-1],), w, eps)


def _fold(q: Tensor, g: int) -> Tensor:
    """(1, L, n_heads, hd) -> (1, n_kv, g * L, hd): the g query heads of a kv head, end to end.

    Grouped-query attention then runs against the kv heads as they are, with no copies of the keys
    and values per query head and nothing above four dimensions.
    """
    _, L, nq, hd = q.shape
    return q.transpose(1, 2).reshape(1, nq // g, g * L, hd)


def _unfold(a: Tensor, g: int) -> Tensor:
    """(1, n_kv, g * L, hd) -> (1, L, n_heads * hd), the inverse of `_fold` plus the head merge."""
    _, nk, gl, hd = a.shape
    return a.reshape(1, nk * g, gl // g, hd).transpose(1, 2).reshape(1, gl // g, nk * g * hd)


def _attn(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    bias: Tensor | None = None,
    aug: tuple[Tensor, Tensor] | None = None,
) -> Tensor:
    """softmax(q k^T / sqrt(hd) + mask) v over (..., Lq, hd) and (..., Lk, hd).

    The mask is either `bias`, added to the scores, or `aug` = (a, b) with mask = a b^T, appended
    to q and k as one more column.
    """
    q = q * (q.shape[-1] ** -0.5)
    if aug is not None:
        q, k = torch.cat([q, aug[0]], dim=-1), torch.cat([k, aug[1]], dim=-1)
    s = torch.matmul(q, k.transpose(-1, -2))
    if bias is not None:
        s = s + bias
    return torch.matmul(torch.softmax(s, dim=-1), v)


def _bias(mask: Tensor, g: int) -> Tensor:
    """(Lq, Lk) bool mask, True = attend, as an additive (g * Lq, Lk) float for folded heads."""
    return torch.where(mask, 0.0, NEG).to(torch.float32).repeat(g, 1)


def _grid(n: int) -> tuple[int, int] | None:
    """(H, W) to lay n tokens out on for an MLP of 1x1 convolutions, or None to keep a sequence.

    The NPU's DPU is a convolution engine and runs a tall grid faster than a sequence, a square or
    a wide grid, so this picks the squarest grid at least twice as tall as it is wide.
    """
    best = None
    for w in range(2, int(n**0.5) + 1):
        if n % w == 0 and n // w >= 2 * w:
            best = (n // w, w)
    return best


def _to_grid(h: Tensor, grid: tuple[int, int] | None) -> Tensor:
    """(N, L, d) -> (N, d, H, W); the identity without a grid."""
    return h if grid is None else h.transpose(1, 2).reshape(h.shape[0], h.shape[2], *grid)


def _from_grid(y: Tensor, grid: tuple[int, int] | None) -> Tensor:
    """(N, d, H, W) -> (N, L, d), the inverse of `_to_grid`."""
    return y if grid is None else y.reshape(y.shape[0], y.shape[1], -1).transpose(1, 2)


def _fc(x: Tensor, w: Tensor, b: Tensor | None = None) -> Tensor:
    """A linear layer over the last axis of a sequence, or over the channels of a grid."""
    return F.conv2d(x, w[:, :, None, None], b) if x.dim() == 4 else F.linear(x, w, b)


# ------------------------------------------------------------------------------------------- graphs
class Vision(nn.Module):
    """`SmolVLA.encode_images`, for any number of images; the built graph takes one.

    One camera per inference because the NPU compiler runs a batch of cameras much slower than one
    inference each. The patch embedding is a strided convolution again: `SmolVLA` unfolds patches
    into a GEMM only to avoid cuDNN's TF32. The MLPs run on a grid (`_grid`).
    """

    def __init__(self, m: SmolVLA):
        super().__init__()
        self.m = m

    def forward(self, px: Tensor) -> Tensor:
        m, s = self.m, self.m.spec
        n, d, P = px.shape[0], s.vis_dim, s.patch
        x = F.conv2d(px, m.p("vis.patch.w").view(d, 3, P, P), m.p("vis.patch.b"), stride=P)
        x = x.flatten(2).transpose(1, 2) + m.p("vis.pos")  # (N, 1024, 768)
        L, hd = x.shape[1], d // s.vis_heads
        grid = _grid(L)

        def ln(t, name):
            return F.layer_norm(t, (d,), m.p(name + ".w"), m.p(name + ".b"), s.vis_eps)

        def heads(t):
            return t.reshape(n, L, s.vis_heads, hd).transpose(1, 2)  # (N, heads, L, hd)

        for i in range(s.vis_layers):
            qkv = F.linear(ln(x, f"vis.{i}.ln1"), m.p(f"vis.{i}.qkv.w"), m.p(f"vis.{i}.qkv.b"))
            q, k, v = (heads(t) for t in qkv.chunk(3, dim=-1))
            a = _attn(q, k, v).transpose(1, 2).reshape(n, L, d)
            x = x + F.linear(a, m.p(f"vis.{i}.out.w"), m.p(f"vis.{i}.out.b"))
            h = _to_grid(ln(x, f"vis.{i}.ln2"), grid)
            h = _fc(h, m.p(f"vis.{i}.fc1.w"), m.p(f"vis.{i}.fc1.b"))
            h = _fc(F.gelu(h, approximate="tanh"), m.p(f"vis.{i}.fc2.w"), m.p(f"vis.{i}.fc2.b"))
            x = x + _from_grid(h, grid)
        x = F.linear(pixel_shuffle(ln(x, "vis.post_ln"), s.shuffle), m.p("conn.w"))  # (N, 64, 960)
        return (x * math.sqrt(s.text_dim)).reshape(n * x.shape[1], s.text_dim)


class Prefix(nn.Module):
    """`SmolVLA.prefix_kv` + `expert_cache`: the VLM layers over [image rows, task, state], and from
    each the keys and values the matching expert layer attends to.

    Functional where `SmolVLA` writes its caches in place: the keys and values are the outputs.
    They are what a step needs and no more, at the kv heads: the VLM layer's own for the expert's
    self-attention layers, and the expert's projections of them, once per chunk, for its
    cross-attention layers. The MLPs run on a grid (`_grid`).

    `mask` is the prefix mask, (L, L) bool with True = attend. If it has rank one, as the
    reference's does (only the state token attends to the state token), it goes into the score
    matmul as a column (`_attn`'s `aug`), which the NPU runs faster than adding a constant to the
    scores. Any other mask is added.
    """

    def __init__(self, m: SmolVLA, lang: Tensor, mask: Tensor):
        super().__init__()
        self.m = m
        s = m.spec
        g = self.g = s.n_heads // s.n_kv
        L = s.prefix_len
        cos, sin = rope_tables(torch.arange(L), s.head_dim, s.rope_wavelength)
        self.register_buffer("lang", lang.float())
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)
        self.grid = _grid(L)
        rows, cols = (~mask).any(dim=1), (~mask).any(dim=0)  # who is denied, what is denied
        self.rank1 = bool(torch.equal(~mask, rows[:, None] & cols[None, :]))
        if self.rank1:

            def col(t, n):
                return t.float()[None, None, :, None].expand(1, s.n_kv, n, 1).clone()

            self.register_buffer("aug_q", col(_AUG * rows.repeat(g), g * L))
            self.register_buffer("aug_k", col(NEG / _AUG * cols, L))
        else:
            self.register_buffer("bias", _bias(mask, g))

    def forward(self, rows: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
        m, s, g = self.m, self.m.spec, self.g
        nq, nk, hd = s.n_heads, s.n_kv, s.head_dim
        grid, ch = self.grid, -1 if self.grid is None else 1
        mask = {"aug": (self.aug_q, self.aug_k)} if self.rank1 else {"bias": self.bias}
        st_row = F.linear(state, m.p("state.w"), m.p("state.b"))[None]
        x = torch.cat([rows, self.lang, st_row], dim=0)[None]  # (1, L, 960)
        L = x.shape[1]
        Ks, Vs = [], []
        for i in range(s.n_layers):
            qkv = F.linear(_rms(x, m.p(f"txt.{i}.ln_in"), s.rms_eps), m.p(f"txt.{i}.qkv"))
            q, k, v = qkv.split([nq * hd, nk * hd, nk * hd], dim=-1)
            k = apply_rope(k.reshape(1, L, nk, hd), self.cos, self.sin)
            v = v.reshape(1, L, nk, hd)
            kt, vt = k.transpose(1, 2), v.transpose(1, 2)  # (1, nk, L, hd)
            if s.is_self_attn(i):
                Ks.append(kt)
                Vs.append(vt)
            else:
                kx = F.linear(k.reshape(1, L, nk * hd), m.p(f"exp.{i}.k"))
                Ks.append(kx.reshape(1, L, nk, hd).transpose(1, 2))
                vx = F.linear(v.reshape(1, L, nk * hd), m.p(f"exp.{i}.v"))
                Vs.append(vx.reshape(1, L, nk, hd).transpose(1, 2))
            if i == s.n_layers - 1:
                break  # the last layer's output is never read
            q = apply_rope(q.reshape(1, L, nq, hd), self.cos, self.sin)
            x = x + F.linear(_unfold(_attn(_fold(q, g), kt, vt, **mask), g), m.p(f"txt.{i}.o"))
            h = _to_grid(_rms(x, m.p(f"txt.{i}.ln_post"), s.rms_eps), grid)
            gt, u = _fc(h, m.p(f"txt.{i}.gate_up")).chunk(2, dim=ch)
            x = x + _from_grid(_fc(F.silu(gt) * u, m.p(f"txt.{i}.down")), grid)
        return torch.cat(Ks, dim=0), torch.cat(Vs, dim=0)  # (n_layers, nk, L, hd)


class Step(nn.Module):
    """`SmolVLA.velocity` for one denoising step; `time` is that step's row of the time table.

    One step per graph, with the Euler update on the host in float32, so the state x_t is not
    rounded to float16 at every step. The expert's MLPs stay matmuls: at a chunk's few tokens the
    sequence is faster than a grid.
    """

    def __init__(self, m: SmolVLA, mask: Tensor):
        super().__init__()
        self.m = m
        s = m.spec
        self.g = s.n_heads // s.n_kv
        L, H = s.prefix_len, s.chunk
        self.register_buffer("bias", _bias(mask, self.g))
        for name, pos in (("s", torch.arange(L, L + H)), ("c", torch.arange(H))):
            cos, sin = rope_tables(pos, s.head_dim, s.rope_wavelength)
            self.register_buffer("cos_" + name, cos)
            self.register_buffer("sin_" + name, sin)

    def forward(self, x_t: Tensor, time: Tensor, K: Tensor, V: Tensor) -> Tensor:
        m, s, g = self.m, self.m.spec, self.g
        nq, nk, hd = s.n_heads, s.n_kv, s.head_dim
        h = F.linear(x_t, m.p("act.in.w"), m.p("act.in.b"))
        h = F.linear(h, m.p("act.time_in.w_action")) + time
        e = F.linear(F.silu(h), m.p("act.time_out.w"), m.p("act.time_out.b"))  # (1, H, 720)
        H = e.shape[1]
        for i in range(s.n_layers):
            h = _rms(e, m.p(f"exp.{i}.ln_in"), s.rms_eps)
            kp, vp = K[i : i + 1], V[i : i + 1]  # (1, nk, L, hd)
            if s.is_self_attn(i):
                qkv = F.linear(h, m.p(f"exp.{i}.qkv"))
                q, k, v = qkv.split([nq * hd, nk * hd, nk * hd], dim=-1)
                q = apply_rope(q.reshape(1, H, nq, hd), self.cos_s, self.sin_s)
                k = apply_rope(k.reshape(1, H, nk, hd), self.cos_s, self.sin_s).transpose(1, 2)
                v = v.reshape(1, H, nk, hd).transpose(1, 2)
                a = _attn(
                    _fold(q, g), torch.cat([kp, k], dim=2), torch.cat([vp, v], dim=2), self.bias
                )
            else:
                q = F.linear(h, m.p(f"exp.{i}.q")).reshape(1, H, nq, hd)
                q = apply_rope(q, self.cos_c, self.sin_c)
                a = _attn(_fold(q, g), kp, vp)
            e = e + F.linear(_unfold(a, g), m.p(f"exp.{i}.o"))
            h = _rms(e, m.p(f"exp.{i}.ln_post"), s.rms_eps)
            gt, u = F.linear(h, m.p(f"exp.{i}.gate_up")).chunk(2, dim=-1)
            e = e + F.linear(F.silu(gt) * u, m.p(f"exp.{i}.down"))
        return F.linear(_rms(e, m.p("exp.norm"), s.rms_eps), m.p("act.out.w"), m.p("act.out.b"))


# -------------------------------------------------------------------------------------------- build
def time_table(m: SmolVLA) -> Tensor:
    """(n_steps, 1, 1, exp_dim): the time half of action_time_mlp_in for every step.

    Precomputed as `Engine` does. The sinusoid's argument runs to thousands of radians, so it is
    never evaluated in float16.
    """
    s = m.spec
    times = torch.tensor([1.0 - k / s.n_steps for k in range(s.n_steps)], dtype=torch.float32)
    temb = sinusoidal(times, s.exp_dim, s.min_period, s.max_period).float()
    return (temb @ m.p("act.time_in.w_time").T + m.p("act.time_in.b"))[:, None, None, :]


def graphs(
    m: SmolVLA, lang: Tensor
) -> dict[str, tuple[nn.Module, dict[str, tuple[int, ...]], tuple[str, ...]]]:
    """The graphs by name, each as (module, {input name: shape}, output names).

    `m` must hold float32 weights on the CPU.
    """
    from .engine import _prefix_mask, _suffix_mask

    s = m.spec
    L, H = s.prefix_len, s.chunk
    kv = (s.n_layers, s.n_kv, L, s.head_dim)
    prefix_in = {"rows": (s.n_cams * s.img_tokens, s.text_dim), "state": (s.max_state,)}
    step_in = {"x_t": (1, H, s.max_action), "time": (1, 1, s.exp_dim), "K": kv, "V": kv}
    return {
        "vision": (Vision(m), {"px": (1, 3, s.image_size, s.image_size)}, ("rows",)),
        "prefix": (Prefix(m, lang, _prefix_mask(s, "cpu")), prefix_in, ("K", "V")),
        "step": (Step(m, _suffix_mask(s, "cpu")), step_in, ("v_t",)),
    }


def int8_weights(ovm, weights: list[np.ndarray]) -> int:
    """Store the weights of an OpenVINO model's matmuls and 1x1 convolutions as int8, one scale per
    output channel, and return the number of layers changed.

    The denoising steps stream the expert's weights from memory at every step, so half the bytes
    makes them faster. It rounds the weights, so it is opt-in per stage.

    Each weight becomes Constant(int8) -> Convert -> Multiply(scale) in front of its layer, the form
    the NPU compiler keeps compressed (NNCF's INT8_SYM writes the same); activations stay float16.
    The quantizer is symmetric, 127 levels each way, with the scale rounded to float16 (as it is
    stored) before the weights are divided by it.

    Only layers whose weight equals one of `weights` are touched. The caller passes the projections
    `param_shapes` stores in the compute dtype, so the small layers that stay float32 everywhere
    else (state and action in, time, action out) stay float here too, as does the patch embedding.
    """
    import openvino as ov
    import openvino.opset13 as ops

    by_shape: dict[tuple, list[np.ndarray]] = {}
    for w in weights:
        by_shape.setdefault(w.shape, []).append(w)
    n = 0
    for node in ovm.get_ordered_ops():
        kind = node.get_type_name()
        if kind not in ("MatMul", "Convolution"):
            continue
        w = node.input_value(1).get_node()
        if w.get_type_name() != "Constant" or w.get_output_element_type(0) != ov.Type.f32:
            continue  # a product of two activations (attention)
        W = w.get_data().astype(np.float32)  # (out, in) or (out, in, 1, 1)
        if kind == "MatMul" and not node.get_attributes()["transpose_b"]:
            continue
        flat = W.reshape(W.shape[:2]) if W.ndim == 4 and W.shape[2:] == (1, 1) else W
        if not any(np.array_equal(flat, c) for c in by_shape.get(flat.shape, ())):
            continue
        axes = tuple(range(1, W.ndim))
        scale = (np.abs(W).max(axis=axes, keepdims=True).clip(1e-8) / 127.0).astype(np.float16)
        q = np.clip(np.round(W / scale.astype(np.float32)), -127, 127).astype(np.int8)
        deq = ops.multiply(ops.convert(ops.constant(q), ov.Type.f16), ops.constant(scale))
        node.input(1).replace_source_output(ops.convert(deq, ov.Type.f32).output(0))
        n += 1
    return n


def build_ir(
    bundle: str | Path,
    out: str | Path | None = None,
    fp16: bool = True,
    int8: tuple[str, ...] = (),
    act_clip: dict[str, float] | None = None,
) -> Path:
    """Convert a bundle to OpenVINO IR, one static graph per stage, and return the directory.

    `out` defaults to <bundle>/openvino. `fp16` stores the weights as float16: half the size,
    exact for the checkpoint's bf16 weights unless tiny, and the CPU plugin still computes in
    float32. `int8` names the stages (of `ov_engine.STAGES`) whose large weights are stored as
    int8 instead, see `int8_weights`. `act_clip` makes the vision graph W8A8, see `w8a8_vision`.
    """
    from .bundle import load_bundle
    from .ov_engine import core

    core()  # sets up a user-space NPU driver before the import
    import openvino as ov

    unknown = set(int8) - set(STAGES)
    if unknown:
        raise ValueError(f"int8: unknown stage(s) {sorted(unknown)}; the stages are {STAGES}")
    if act_clip and "vision" in int8:
        raise ValueError("vision is either W8A8 (act_clip) or int8 weights, not both")
    out = Path(out) if out else ir_dir(bundle)
    out.mkdir(parents=True, exist_ok=True)
    model, meta, ex = load_bundle(bundle, "cpu", torch.float32, "fp32")
    s = model.spec
    stage_of = {GRAPH.get(st, st): st for st in STAGES}
    big = [
        model.p(n).numpy()
        for n, (_, dt) in param_shapes(s, torch.float16).items()
        if dt == torch.float16
    ]
    io = {}
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the tracer noting that shapes become constants
        for name, (module, inputs, outputs) in graphs(model, ex["lang.emb"]).items():
            if name == "vision" and act_clip:
                ovm = w8a8_vision(bundle, s, act_clip, out)
            else:
                example = tuple(torch.zeros(shape) for shape in inputs.values())
                ovm = ov.convert_model(module.eval(), example_input=example)
            for port, in_name in zip(ovm.inputs, inputs, strict=True):
                port.get_tensor().set_names({in_name})
            ovm.reshape({in_name: list(shape) for in_name, shape in inputs.items()})
            for port, out_name in zip(ovm.outputs, outputs, strict=True):
                port.get_tensor().set_names({out_name})
            if stage_of[name] in int8:
                log.info("%s: int8 weights on %d layers", name, int8_weights(ovm, big))
            ov.save_model(ovm, str(out / f"{name}.xml"), compress_to_fp16=fp16)
            io[name] = {
                "inputs": {k: list(v) for k, v in inputs.items()},
                "outputs": {o.get_any_name(): list(o.get_shape()) for o in ovm.outputs},
            }
            log.info("%s: %d ops -> %s", name, len(ovm.get_ops()), out / f"{name}.xml")
        np.save(out / "time.npy", time_table(model).numpy())
    ir_meta = {
        "format": FORMAT,
        "weights": weights_id(bundle),
        "weights_fp16": fp16,
        "int8": sorted(int8),
        "w8a8": ["vision"] if act_clip else [],
        "openvino": ov.__version__,
        "torch": torch.__version__,
        "n_steps": s.n_steps,
        "dt": -1.0 / s.n_steps,
        "stages": io,
    }
    (out / "meta.json").write_text(json.dumps(ir_meta, indent=1))
    return out


def w8a8_vision(bundle: str | Path, spec, act_clip: dict[str, float], work: Path):
    """The vision graph for one camera with int8 weights and int8 activations.

    This is the graph `build-trt --precision int8` gives TensorRT: the tower exported to ONNX and
    every linear wrapped in Q/DQ nodes, weights per output channel and inputs per tensor clipped
    at `act_clip` (`trt_vision.quantize_onnx`). The NPU computes what that graph defines, so the
    Orin and the NPU run the same int8 vision. Attention, norms and the patch embedding stay float.
    """
    import openvino as ov

    from .trt_vision import export_onnx, quantize_onnx

    plain = export_onnx(bundle, work / "vision1.onnx", cams=1)
    quant = quantize_onnx(plain, work / "vision1_int8.onnx", spec, act_clip)
    ovm = ov.convert_model(str(quant))
    plain.unlink()
    quant.unlink()
    return ovm
