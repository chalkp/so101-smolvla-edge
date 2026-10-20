"""The OpenVINO engine end to end: on the CPU plugin against the torch engine, with a small random
model wherever openvino is installed, and on the NPU against a bundle's reference actions
(SO101_BUNDLE=<bundle dir>, on a machine that has one)."""

import json
import os
from pathlib import Path

import numpy as np
import pytest

import so101_smolvla_edge.model  # noqa: F401

ov_engine = pytest.importorskip("so101_smolvla_edge.model.ov_engine")
try:
    CORE = ov_engine.core()
except SystemExit:
    pytest.skip("openvino is not installed", allow_module_level=True)
torch = pytest.importorskip("torch")
BUNDLE = os.environ.get("SO101_BUNDLE")


def built(bundle):
    from so101_smolvla_edge.model.ov_graph import build_ir

    build_ir(bundle, fp16=False)  # float32 weights: the CPU plugin is then exact
    return bundle


@pytest.mark.parametrize("action_repr", ["absolute", "delta", "velocity"])
def test_cpu_matches_the_torch_engine(tiny_bundle, observation, action_repr):
    from so101_smolvla_edge.model.engine import Engine

    bundle = built(tiny_bundle(action_repr))
    ref = Engine(bundle, device="cpu", dtype="fp32", cuda_graph=False, warmup=0)
    eng = ov_engine.OVEngine(bundle, device="CPU", warmup=1)
    for seed in (0, 1):
        images, state, noise = observation(bundle, seed)
        want = ref.infer(images, state, noise=noise)
        got = eng.infer(images, state, noise=noise)
        assert got.shape == want.shape == (8, 6)
        assert np.abs(got - want).max() < 1e-4


def test_int8_weights_are_quant_int8_per_channel(tiny_bundle, observation, tmp_path):
    """`build-ov --int8` graphs on the CPU plugin compute what the torch engine computes from
    weights quantized the same way: symmetric int8 per output channel, 127 levels each way, with
    the scale held in float16, on the large projections only.

    The reference must round its scale to float16 too: with a float32 scale enough weights round
    the other way to move this small model's output as far as quantizing does.
    """
    import openvino as ov
    from safetensors.torch import load_file, save_file

    from so101_smolvla_edge.model.engine import Engine
    from so101_smolvla_edge.model.ov_graph import build_ir
    from so101_smolvla_edge.model.smolvla import ModelSpec, param_shapes

    def int8_per_channel(w):  # the arithmetic of ov_graph.int8_weights
        scale = (w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 127.0).half()
        return (torch.clamp(torch.round(w / scale.float()), -127, 127).half() * scale).float()

    bundle = tiny_bundle()
    spec = ModelSpec(**json.loads((bundle / "meta.json").read_text())["spec"])
    shapes = param_shapes(spec, torch.float16)
    big = {n for n, (_, dt) in shapes.items() if dt == torch.float16} - {"vis.patch.w"}
    sd = load_file(str(bundle / "model.safetensors"))
    fake = tmp_path / "fake_quantized"
    fake.mkdir()
    quantized = {k: int8_per_channel(v) if k in big else v for k, v in sd.items()}
    save_file(quantized, str(fake / "model.safetensors"))
    (fake / "meta.json").write_text((bundle / "meta.json").read_text())
    want_engine = Engine(fake, device="cpu", dtype="fp32", cuda_graph=False, warmup=0)
    float_engine = Engine(bundle, device="cpu", dtype="fp32", cuda_graph=False, warmup=0)

    build_ir(bundle, fp16=False, int8=("vision", "prefix", "denoise"))
    counts = {}
    for graph in ("vision", "prefix", "step"):
        m = CORE.read_model(str(ov_engine.ir_dir(bundle) / f"{graph}.xml"))
        counts[graph] = sum(
            o.get_type_name() == "Constant" and o.get_output_element_type(0) == ov.Type.i8
            for o in m.get_ops()
        )
    # vision: qkv, out, fc1, fc2 per layer and the connector. prefix: qkv, o, gate_up, down per
    # layer but the last, which needs only its qkv, and the expert's k and v per cross-attention
    # layer. step: o, gate_up, down per layer, and qkv or q.
    v, n = spec.vis_layers, spec.n_layers
    assert counts == {"vision": 4 * v + 1, "prefix": 4 * (n - 1) + 1 + n, "step": 3 * n + n}
    eng = ov_engine.OVEngine(bundle, device="CPU", warmup=0)
    assert eng.int8 == ["denoise", "prefix", "vision"]
    for seed in (0, 1):
        images, state, noise = observation(bundle, seed)
        got = eng.infer(images, state, noise=noise)
        want = want_engine.infer(images, state, noise=noise)
        unquantized = float_engine.infer(images, state, noise=noise)
        err, quantization = np.abs(got - want).max(), np.abs(want - unquantized).max()
        assert err < 3e-3 and err < 0.2 * quantization


def test_int8_stage_names_are_checked(tiny_bundle):
    from so101_smolvla_edge.model.ov_graph import build_ir

    with pytest.raises(ValueError, match="unknown stage"):
        build_ir(tiny_bundle(), int8=("step",))


def test_preprocessing_matches_torch(tiny_bundle, observation):
    """cv2's resize and padding against `Engine._vision`'s interpolate and pad."""
    import torch.nn.functional as F

    bundle = built(tiny_bundle())
    eng = ov_engine.OVEngine(bundle, device="CPU", warmup=0)
    images, state, noise = observation(bundle)
    eng.infer(images, state, noise=noise)
    frames = torch.from_numpy(np.stack([images[c] for c in eng.cams]))
    x = frames.permute(0, 3, 1, 2).float() / 255.0
    resized = F.interpolate(x, size=(96, 128), mode="bilinear", align_corners=False)
    px = F.pad(resized, (0, 0, 32, 0)) * 2 - 1
    assert np.abs(eng.px - px.numpy()).max() < 1e-5


def test_results_do_not_alias(tiny_bundle, observation):
    bundle = built(tiny_bundle())
    eng = ov_engine.OVEngine(bundle, device="CPU", warmup=0)
    a = eng.infer(*observation(bundle, 0))
    kept = a.copy()
    b = eng.infer(*observation(bundle, 1))
    assert np.array_equal(a, kept) and not np.array_equal(a, b)


def test_noise_is_drawn_when_not_given(tiny_bundle, observation):
    bundle = built(tiny_bundle())
    eng = ov_engine.OVEngine(bundle, device="CPU", warmup=0)
    images, state, _ = observation(bundle)
    assert not np.array_equal(eng.infer(images, state), eng.infer(images, state))


def test_graphs_of_other_weights_are_refused(tiny_bundle):
    bundle = built(tiny_bundle())
    w = bundle / "model.safetensors"
    os.utime(w, ns=(w.stat().st_atime_ns, w.stat().st_mtime_ns + 1_000_000_000))  # "re-exported"
    with pytest.raises(SystemExit, match="rebuild"):
        ov_engine.OVEngine(bundle, device="CPU", warmup=0)


def test_missing_graphs_say_how_to_build_them(tiny_bundle):
    with pytest.raises(SystemExit, match="build-ov"):
        ov_engine.OVEngine(tiny_bundle(), device="CPU", warmup=0)


def test_wrong_image_size_is_an_error(tiny_bundle, observation):
    bundle = built(tiny_bundle())
    eng = ov_engine.OVEngine(bundle, device="CPU", warmup=0)
    images, state, _ = observation(bundle)
    images[eng.cams[0]] = images[eng.cams[0]][:60]
    with pytest.raises(ValueError, match="expects"):
        eng.infer(images, state)


def test_stage_devices():
    sd = ov_engine.stage_devices
    assert sd("npu") == {"vision": "NPU", "prefix": "NPU", "denoise": "NPU"}
    assert sd("NPU, cpu,CPU") == {"vision": "NPU", "prefix": "CPU", "denoise": "CPU"}
    for bad in ("AUTO", "NPU,CPU", "HETERO:NPU,CPU", "NPU,AUTO,CPU", ""):
        with pytest.raises(ValueError):
            sd(bad)


def test_an_unavailable_device_is_an_error_not_a_fallback(tiny_bundle):
    bundle = built(tiny_bundle())
    with pytest.raises(SystemExit, match="not available"):
        ov_engine.OVEngine(bundle, device="GPU.7", warmup=0)


@pytest.mark.skipif(
    not (BUNDLE and "NPU" in CORE.available_devices), reason="needs an NPU and SO101_BUNDLE"
)
def test_npu_matches_reference_and_runs_on_the_npu():
    g = dict(np.load(Path(BUNDLE) / "golden.npz"))
    cams = json.loads((Path(BUNDLE) / "meta.json").read_text())["cameras"]
    eng = ov_engine.OVEngine(BUNDLE, device="NPU", warmup=1)
    busy = ov_engine.npu_busy_us()
    out = np.stack(
        [
            eng.infer({c: g[f"img.{c}"][i] for c in cams}, g["state"][i], noise=g["noise"][i])
            for i in range(len(g["state"]))
        ]
    )
    assert np.abs(out - g["actions_fp32"]).max() < 3e-2
    # the kernel driver saw the NPU work, not just OpenVINO
    assert ov_engine.npu_busy_us() - busy > 50_000 * len(out)


def test_w8a8_vision_is_the_tensorrt_int8_graph(tiny_bundle, tmp_path):
    """`build-ov --w8a8-vision` computes the vision tower with int8 weights per output channel and
    int8 inputs per tensor clipped at act_clip, as `build-trt --precision int8` does.

    Checked one linear at a time: with all of them quantized, an input on a rounding boundary in
    an early layer rounds the other way and the step spreads through attention to every row.
    """
    import openvino as ov

    from so101_smolvla_edge.model.bundle import load_bundle
    from so101_smolvla_edge.model.ov_graph import build_ir
    from so101_smolvla_edge.model.smolvla import SmolVLA
    from so101_smolvla_edge.model.trt_vision import export_onnx, quantize_onnx

    bundle = tiny_bundle()
    model, _, _ = load_bundle(bundle, "cpu", torch.float32, "fp32")
    s = model.spec
    px = torch.rand(1, 3, s.image_size, s.image_size, generator=torch.Generator().manual_seed(0))
    px = px * 2 - 1
    vision = {"vis.patch.w": "patch", "conn.w": "conn"}
    vision |= {
        f"vis.{i}.{n}.w": f"vis.{i}.{n}"
        for i in range(s.vis_layers)
        for n in ("qkv", "out", "fc1", "fc2")
    }
    clip = {}

    def calibrate(self, x, w, b=None):
        if w in vision:
            clip[vision[w]] = max(clip.get(vision[w], 0.0), float(x.abs().max()))
        return SmolVLA.lin(self, x, w, b)

    def fake_quant(key):
        def lin(self, x, w, b=None):
            if vision.get(w) != key:
                return SmolVLA.lin(self, x, w, b)
            xs = clip[key] / 127.0
            xq = torch.clamp(torch.round(x / xs), -128, 127) * xs
            W = self.p(w)
            ws = (W.abs().amax(1, keepdim=True) / 127.0).clamp(min=1e-8)
            Wq = torch.clamp(torch.round(W / ws), -127, 127) * ws
            return torch.nn.functional.linear(xq, Wq, None if b is None else self.p(b))

        return lin

    with torch.no_grad():
        model.lin = calibrate.__get__(model)
        plain = model.encode_images(px).numpy()
    cfg = {"INFERENCE_PRECISION_HINT": "f32", "LP_TRANSFORMS_MODE": False}  # as OVEngine on CPU
    onnx = export_onnx(bundle, tmp_path / "vision.onnx", cams=1)
    for key in vision.values():
        with torch.no_grad():
            model.lin = fake_quant(key).__get__(model)
            want = model.encode_images(px).numpy()
        graph = quantize_onnx(onnx, tmp_path / "q.onnx", s, {key: clip[key]})
        got = CORE.compile_model(ov.convert_model(str(graph)), "CPU", cfg)({0: px.numpy()})[0]
        # float32 noise on one CPU, a few inputs rounding the other way on another; a wrong
        # scale axis or clip is off by several percent
        assert np.abs(got - want).max() < 1e-2 * np.abs(want).max(), key
        assert np.abs(want - plain).max() > 1e-3 * np.abs(want).max(), key  # quantized at all

    out = build_ir(bundle, fp16=False, act_clip=clip)
    assert json.loads((out / "meta.json").read_text())["w8a8"] == ["vision"]
    got = CORE.compile_model(str(out / "vision.xml"), "CPU", cfg)({"px": px.numpy()})["rows"]
    assert got.shape == plain.shape
    assert 1e-3 < np.abs(got - plain).max() / np.abs(plain).max() < 5e-2


def test_vision_is_w8a8_or_int8_weights_not_both(tiny_bundle):
    from so101_smolvla_edge.model.ov_graph import build_ir

    with pytest.raises(ValueError):
        build_ir(tiny_bundle(), int8=("vision",), act_clip={"conn": 1.0})
