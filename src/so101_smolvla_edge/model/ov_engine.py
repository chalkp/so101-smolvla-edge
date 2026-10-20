"""Inference on OpenVINO: the three graphs of `ov_graph.py` on an Intel NPU or the CPU.

    eng = OVEngine("bundle/", device="NPU")     # compiles, cached in <bundle>/openvino/cache
    chunk = eng.infer({"observation.images.top_rgb": rgb_uint8_hwc, ...}, state_12)  # (50, 12) rad

Same contract as `engine.Engine`, with numpy, OpenCV and openvino instead of torch. Build the graphs
first, in an environment with the `npu` extra: `so101-smolvla-edge build-ov BUNDLE`.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from safetensors import safe_open

from .._npu_libs import preload
from .timing import Timing

log = logging.getLogger("so101_smolvla_edge.ov")

FORMAT = 2  # layout of <bundle>/openvino; graphs of another format are refused
STAGES = ("vision", "prefix", "denoise")  # what a device is chosen for
GRAPH = {"denoise": "step"}  # ... and its graph, where the name differs


def ir_dir(bundle: str | Path) -> Path:
    """Directory of a bundle's OpenVINO graphs."""
    return Path(bundle) / "openvino"


def weights_id(bundle: str | Path) -> str:
    """Identify the weights an IR was built from: size and mtime, not a hash of 0.7 GB."""
    st = (Path(bundle) / "model.safetensors").stat()
    return f"{st.st_size}:{st.st_mtime_ns}"


_CORE = None  # the process' one Core, see core()


def core():
    """Return the process' OpenVINO Core, setting up a user-space NPU driver first if there is one.

    One Core per process: every new Core loads the device plugins again, and with Level Zero's
    loader installed but no NPU driver (a desktop with `libze1`) the NPU plugin's second
    initialisation segfaults inside `zeInitDrivers`.

    It also hides the `openvino_telemetry` package, so the model converter that `build-ov` runs
    falls back to its no-op stub instead of reporting each conversion to Intel. Like the driver
    setup, that has to happen before openvino is imported.
    """
    global _CORE
    if _CORE is not None:
        return _CORE
    preload()
    sys.modules.setdefault("openvino_telemetry", None)  # None makes the import raise
    try:
        import openvino as ov
    except ImportError as exc:
        raise SystemExit(
            "openvino is not installed: pixi install -e npu (or uv sync --extra npu)"
        ) from exc
    _CORE = ov.Core()
    return _CORE


def npu_busy_us() -> int | None:
    """Microseconds the NPU has spent executing since boot, by the kernel driver's count.

    This is the ground truth for whether something ran on the NPU; a plugin's own report is not.
    """
    try:
        return int(Path("/sys/class/accel/accel0/device/npu_busy_time_us").read_text())
    except (OSError, ValueError):
        return None


def cache_key(c, device: str) -> str:
    """Directory name for a device's compiled blobs.

    OpenVINO's cache does not notice a new NPU driver and would load a blob from the old driver's
    compiler as it is, so the driver and compiler versions go in the name.
    """
    if not device.startswith("NPU"):
        return device.lower()
    versions = (
        str(c.get_property("NPU", k)) for k in ("NPU_DRIVER_VERSION", "NPU_COMPILER_VERSION")
    )
    return "npu-" + "-".join(versions)


def device_exists(c, device: str) -> bool:
    """Whether OpenVINO can open exactly this device ("GPU.1", not just some GPU).

    A prefix match is not enough: a desktop whose NVIDIA card OpenVINO lists as "GPU" (through
    OpenCL) has no "GPU.7", and asking for it must fail here rather than inside compile_model.
    """
    if device.split(".")[0] not in {x.split(".")[0] for x in c.available_devices}:
        return False
    try:
        c.get_property(device, "FULL_DEVICE_NAME")
    except RuntimeError:
        return False
    return True


def stage_devices(device: str) -> dict[str, str]:
    """Map each stage to a device, from one device for all or "vision,prefix,denoise".

    AUTO, HETERO and MULTI are refused: a stage the NPU cannot compile should fail at load, not
    move to the CPU unannounced.
    """
    d = [x.strip().upper() for x in device.split(",")]
    if len(d) == 1:
        d = d * 3
    if len(d) != 3 or any(x.split(":")[0] in ("AUTO", "HETERO", "MULTI", "") for x in d):
        raise ValueError(
            f"device must be one OpenVINO device or 'vision,prefix,denoise', not {device!r}"
        )
    return dict(zip(STAGES, d, strict=True))


class OVEngine:
    """SmolVLA on OpenVINO, one action chunk per `infer`.

    A chunk is the image preprocessing on the host, one vision inference per camera, one prefix
    inference, then `n_steps` step inferences with the Euler update on the host in float32. The
    tensors between stages are bound once, at load, with the prefix's K and V handed to the step
    graph as they are, so a chunk moves no data through Python. Each camera is preprocessed while
    the device runs the one before it.

    `device` is one OpenVINO device for every stage or three, "vision,prefix,denoise" (e.g.
    "NPU,CPU,CPU"). `turbo` sets NPU_TURBO, the NPU's `jetson_clocks`: at the robot's replan
    period an idle NPU clocks down and suspends between chunks, and every chunk would start on a
    cold device. It costs energy, and the driver then waits for each inference spinning on a host
    core.
    """

    def __init__(
        self,
        bundle: str | Path,
        device: str = "NPU",
        warmup: int = 2,
        cache: bool = True,
        turbo: bool = True,
    ):
        c = core()  # before the first `import openvino`: see _npu_libs
        import openvino as ov

        bundle = Path(bundle)
        self.meta = json.loads((bundle / "meta.json").read_text())
        s = self.spec = self.meta["spec"]
        self.cams = self.meta["cameras"]
        self.repr = self.meta["action_repr"]
        self.state_dim, self.action_dim = self.meta["state_dim"], self.meta["action_dim"]
        d = ir_dir(bundle)
        if not (d / "meta.json").exists():
            raise SystemExit(
                f"no OpenVINO graphs at {d}; build them: so101-smolvla-edge build-ov {bundle}"
            )
        ir = json.loads((d / "meta.json").read_text())
        if ir.get("format") != FORMAT:
            raise SystemExit(
                f"{d}: IR format {ir.get('format')}, this code reads {FORMAT}; "
                "rebuild with build-ov"
            )
        if ir["weights"] != weights_id(bundle):
            raise SystemExit(
                f"{d} was built from other weights than {bundle}/model.safetensors; "
                "rebuild with build-ov"
            )
        self.n_steps, self.dt = ir["n_steps"], np.float32(ir["dt"])
        self.int8 = ir.get("int8", [])  # stages built with int8 weights (build-ov --int8)
        self.w8a8 = ir.get("w8a8", [])  # stages with int8 activations too (build-ov --w8a8-vision)
        self.time = np.load(d / "time.npy").astype(np.float32)  # (n_steps, 1, 1, exp_dim)
        with safe_open(str(bundle / "model.safetensors"), "np") as f:  # only six small tensors

            def get(k):
                return f.get_tensor(k).astype(np.float32)

            self.s_mean, self.s_std = get("norm.state.mean"), get("norm.state.std")
            self.a_mean, self.a_std = get("norm.action.mean"), get("norm.action.std")
            self.r_mean, self.r_std = get("repr.mean"), get("repr.std")

        # resize keeping the aspect ratio, then pad left and top, as the training code does
        S = s["image_size"]
        h, w = self.meta["image_hw"]
        ratio = max(w / S, h / S)
        self._rh, self._rw = int(h / ratio), int(w / ratio)
        self._hw = (h, w)

        # the infer requests share these buffers: write into them, never rebind them
        self.px = np.full((s["n_cams"], 3, S, S), -1.0, dtype=np.float32)  # padding: pixel 0 is -1
        self.st = np.zeros(s["max_state"], dtype=np.float32)
        self.x = np.zeros((1, s["chunk"], s["max_action"]), dtype=np.float32)
        self._t = np.zeros((1, 1, self.time.shape[-1]), dtype=np.float32)

        self.devices = stage_devices(device)
        have = c.available_devices
        compiled = {}
        t0 = time.perf_counter()
        for name, dev in self.devices.items():
            if not device_exists(c, dev):
                hint = (
                    " (NPU driver not found: see scripts/setup_intel_npu.sh)"
                    if dev == "NPU"
                    else ""
                )
                raise SystemExit(
                    f"OpenVINO device {dev} is not available here; devices: {have}{hint}"
                )
            cfg = {"PERFORMANCE_HINT": "LATENCY"}
            if dev == "CPU":
                # compute exactly what the graph says: float32, no int8 activations on top of
                # int8 weights (the CPU plugin's default for them), and Q/DQ pairs as written
                # rather than rewritten into its own int8 kernels, which round differently
                cfg["INFERENCE_PRECISION_HINT"] = "f32"
                cfg["DYNAMIC_QUANTIZATION_GROUP_SIZE"] = 0
                cfg["LP_TRANSFORMS_MODE"] = False
            if dev.startswith("NPU") and turbo:
                cfg["NPU_TURBO"] = True
            if cache:
                cfg["CACHE_DIR"] = str(d / "cache" / cache_key(c, dev))
            t1 = time.perf_counter()
            compiled[name] = c.compile_model(str(d / f"{GRAPH.get(name, name)}.xml"), dev, cfg)
            log.info("%s on %s: ready in %.1f s", name, dev, time.perf_counter() - t1)
        self.load_s = time.perf_counter() - t0

        def shared(a):
            return ov.Tensor(a, shared_memory=True)

        n_img = ir["stages"]["vision"]["outputs"]["rows"][0]  # rows per camera
        self.rows = np.zeros((s["n_cams"] * n_img, s["text_dim"]), dtype=np.float32)
        self._vision = []  # one request per camera, each bound to its slices
        for i in range(s["n_cams"]):
            v = compiled["vision"].create_infer_request()
            v.set_tensor("px", shared(self.px[i : i + 1]))
            v.set_tensor("rows", shared(self.rows[i * n_img : (i + 1) * n_img]))
            self._vision.append(v)
        p = self._prefix = compiled["prefix"].create_infer_request()
        st = self._step = compiled["denoise"].create_infer_request()
        p.set_tensor("rows", shared(self.rows))
        p.set_tensor("state", shared(self.st))
        st.set_tensor("x_t", shared(self.x))
        st.set_tensor("time", shared(self._t))
        st.set_tensor("K", p.get_tensor("K"))
        st.set_tensor("V", p.get_tensor("V"))
        self._v_t = st.get_tensor("v_t")
        self.timing = Timing()
        self.stage_ms = {"preprocess": 0.0, "vision": 0.0, "prefix": 0.0, "denoise": 0.0}
        zeros = {c_: np.zeros((h, w, 3), np.uint8) for c_ in self.cams}
        for _ in range(warmup):
            self.infer(zeros, np.zeros(self.state_dim, np.float32))
        self.timing.history.clear()

    # ------------------------------------------------------------------ host side
    def _preprocess(self, images: dict[str, np.ndarray], i: int) -> None:
        """Write camera i into `px` as `Engine._vision` does: resize, pad left and top, [-1, 1].

        cv2's INTER_LINEAR on floats is torch's bilinear with align_corners=False. Scaling after
        the resize rather than before it can move a pixel by a float32 ulp.
        """
        S = self.px.shape[-1]
        im = images[self.cams[i]]
        if im.shape != (*self._hw, 3):
            raise ValueError(
                f"{self.cams[i]}: image {im.shape}, the bundle expects {(*self._hw, 3)}"
            )
        x = np.ascontiguousarray(im, dtype=np.float32)
        if (self._rh, self._rw) != self._hw:
            x = cv2.resize(x, (self._rw, self._rh), interpolation=cv2.INTER_LINEAR)
        dst = self.px[i, :, S - self._rh :, S - self._rw :]
        np.multiply(x.transpose(2, 0, 1), np.float32(2.0 / 255.0), out=dst)
        dst -= 1.0

    def _normalised_state(self, state: np.ndarray) -> np.ndarray:
        """Normalised state, zero-padded to max_state."""
        st = np.zeros(self.st.shape[0], dtype=np.float32)
        x = np.asarray(state, dtype=np.float32)
        st[: self.state_dim] = (x - self.s_mean) / (self.s_std + 1e-8)
        return st

    # ------------------------------------------------------------------ public
    def infer(
        self, images: dict[str, np.ndarray], state: np.ndarray, noise: np.ndarray | None = None
    ) -> np.ndarray:
        """Compute one action chunk, (chunk, action_dim) in radians.

        `images` maps each camera key to an HxWx3 uint8 RGB frame, `state` is (state_dim,) in
        radians, and `noise` is the initial x_t, drawn when not given.
        """
        t0 = time.perf_counter()
        self._preprocess(images, 0)
        st = self._normalised_state(state)
        self.st[:] = st
        if noise is None:
            self.x[:] = np.random.standard_normal(self.x.shape)
        else:
            self.x[:] = np.asarray(noise, dtype=np.float32).reshape(self.x.shape)
        t1 = time.perf_counter()
        for i, v in enumerate(self._vision):
            v.start_async()
            try:
                if i + 1 < len(self._vision):
                    self._preprocess(images, i + 1)  # while the device runs camera i
            finally:
                v.wait()  # also on a bad image: never leave a request running
        t2 = time.perf_counter()
        self._prefix.infer()
        t3 = time.perf_counter()
        step = self._step
        for k in range(self.n_steps):
            self._t[:] = self.time[k]
            step.infer()
            self.x += self.dt * self._v_t.data
        a = self.x[0].copy()
        if self.repr != "absolute":
            a = a * self.r_std + self.r_mean
            a = a + st[None] if self.repr == "delta" else st[None] + np.cumsum(a, axis=0)
        out = a[:, : self.action_dim] * self.a_std + self.a_mean
        t4 = time.perf_counter()
        self.stage_ms = {
            "preprocess": (t1 - t0) * 1e3,
            "vision": (t2 - t1) * 1e3,
            "prefix": (t3 - t2) * 1e3,
            "denoise": (t4 - t3) * 1e3,
        }
        self.timing.add((t4 - t0) * 1e3)
        return out

    def profile(self, images: dict[str, np.ndarray], state: np.ndarray, reps: int = 5) -> dict:
        """Mean wall time of each stage over `reps` chunks, in ms.

        The stages run synchronously, so this is `infer`'s own timing.
        """
        acc = dict.fromkeys(self.stage_ms, 0.0)
        for _ in range(reps):
            self.infer(images, state)
            for k, v in self.stage_ms.items():
                acc[k] += v
        return {k: v / reps for k, v in acc.items()}
