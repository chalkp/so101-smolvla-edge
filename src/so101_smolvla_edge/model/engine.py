"""The CUDA inference engine: static buffers, one CUDA graph per chunk.

    eng = Engine("bundle/", device="cuda")  # loads, warms up, captures the graph
    chunk = eng.infer({"observation.images.top_rgb": rgb, ...}, state)  # (50, 12) rad

A chunk is image preprocessing, the vision tower, the prefix pass, ten denoising steps and the
unnormalisation. On a Jetson the steps are bound by kernel launches from the slow ARM cores, so
the whole chunk is captured once and replayed with one launch. Everything the graph touches lives
in a fixed buffer that `infer` only writes into.

Options, each falling back with a log line where it cannot run:

    vision="trt"         the vision tower as a TensorRT engine (`trt_vision.py`), inside the graph
    kernels="triton"     the prefix pass and denoising loop on the Triton kernels (`smolvla_triton`)
    compile=True         torch.compile per stage; "autotune" also autotunes the prefix GEMMs

`kernels="auto"` takes Triton wherever it can run. Compile only applies to stages that neither
TensorRT nor Triton already cover; set TORCHINDUCTOR_CACHE_DIR, or every start compiles again.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .bundle import load_bundle
from .smolvla import rope_tables, sinusoidal
from .timing import Timing

log = logging.getLogger("so101_smolvla_edge.engine")


def auto_kernels(device: torch.device, cdt: torch.dtype, spec=None) -> str:
    """Pick the kernels for `kernels="auto"`: Triton wherever it can run (a CUDA GPU, Triton
    installed, bf16 or fp16, a model the kernels can index), else torch."""
    from .. import kernels as K

    if device.type != "cuda" or cdt == torch.float32 or not K.available():
        return "torch"
    from .smolvla_triton import unsupported

    return "torch" if spec is not None and unsupported(spec, cdt) else "triton"


def _resolve_kernels(kernels: str, device: torch.device, cdt: torch.dtype, spec) -> str:
    if kernels not in ("auto", "torch", "triton"):
        raise ValueError(f"kernels must be 'auto', 'torch' or 'triton', not {kernels!r}")
    if kernels == "auto":
        return auto_kernels(device, cdt, spec)
    if kernels == "torch":
        return kernels
    from .. import kernels as K

    if device.type != "cuda" or not K.available():
        why = "needs CUDA and Triton"
    else:
        from .smolvla_triton import unsupported  # imports triton, so only now

        why = unsupported(spec, cdt)
    if why:
        log.warning("kernels=triton: %s; using the torch kernels", why)
        return "torch"
    return kernels


def _prefix_mask(spec, device) -> torch.Tensor:
    """Prefix attention mask: images and language see each other, the state token sees all of
    them, and nothing sees the state token."""
    att = torch.tensor([0] * (spec.prefix_len - 1) + [1], device=device)
    c = torch.cumsum(att, 0)
    return c[None, :] <= c[:, None]


def _suffix_mask(spec, device) -> torch.Tensor:
    """Action tokens see the whole prefix and are causal among themselves."""
    H, L = spec.chunk, spec.prefix_len
    causal = torch.tril(torch.ones(H, H, dtype=torch.bool, device=device))
    return torch.cat([torch.ones(H, L, dtype=torch.bool, device=device), causal], dim=1)


class Engine:
    """SmolVLA on a CUDA device (or the CPU), one call per action chunk."""

    def __init__(
        self,
        bundle: str | Path,
        device: str = "cuda",
        dtype: str = "bf16",
        residual: str = "fp32",
        cuda_graph: bool = True,
        compile: bool | str = False,
        warmup: int = 3,
        vision: str = "torch",
        trt_precision: str = "bf16",
        kernels: str = "auto",
    ):
        self.device = dev = torch.device(device)
        cdt = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]
        self.model, self.meta, ex = load_bundle(bundle, device, cdt, residual)
        self.spec = s = self.model.spec
        self.kernels = kernels = _resolve_kernels(kernels, dev, cdt, s)
        self.cams = self.meta["cameras"]
        self.repr = self.meta["action_repr"]
        self.state_dim = self.meta["state_dim"]
        self.action_dim = self.meta["action_dim"]
        h, w = self.meta["image_hw"]
        L, H = s.prefix_len, s.chunk

        # graph inputs and output
        self.img = torch.zeros(len(self.cams), h, w, 3, dtype=torch.uint8, device=dev)
        self.state = torch.zeros(self.state_dim, dtype=torch.float32, device=dev)
        self.noise = torch.zeros(1, H, s.max_action, dtype=torch.float32, device=dev)
        self.out = torch.zeros(H, self.action_dim, dtype=torch.float32, device=dev)
        self._host_img = None  # pinned staging buffer for the camera frames
        if dev.type == "cuda":
            self._host_img = torch.zeros(self.img.shape, dtype=torch.uint8).pin_memory()

        self.lang = ex["lang.emb"]
        self.s_mean, self.s_std = ex["norm.state.mean"], ex["norm.state.std"]
        self.a_mean, self.a_std = ex["norm.action.mean"], ex["norm.action.std"]
        self.r_mean, self.r_std = ex["repr.mean"], ex["repr.std"]
        self.mask_p = _prefix_mask(s, dev)
        self.mask_s = _suffix_mask(s, dev)

        def rope(pos):
            return rope_tables(pos, s.head_dim, s.rope_wavelength)

        self.cos_p, self.sin_p = rope(torch.arange(L, device=dev))
        self.cos_s, self.sin_s = rope(torch.arange(L, L + H, device=dev))
        self.cos_c, self.sin_c = rope(torch.arange(H, device=dev))
        self.dt = -1.0 / s.n_steps
        times = torch.tensor([1.0 + k * self.dt for k in range(s.n_steps)], device=dev)
        temb = sinusoidal(times, s.exp_dim, s.min_period, s.max_period).float()
        time_in = temb @ self.model.p("act.time_in.w_time").T + self.model.p("act.time_in.b")
        self.time_proj = time_in[:, None, None, :]  # (n_steps, 1, 1, exp_dim)

        # caches; the Triton path keeps its own at the kv heads
        self.tp = None
        if kernels == "triton":
            from .smolvla_triton import TritonPath

            self.tp = TritonPath(self.model, dev)
        kv = (1, L, s.n_kv, s.head_dim) if kernels == "torch" else (0,)
        self.k_pre = [torch.zeros(kv, dtype=cdt, device=dev) for _ in range(s.n_layers)]
        self.v_pre = [torch.zeros(kv, dtype=cdt, device=dev) for _ in range(s.n_layers)]
        if kernels == "torch":
            self_attn = [s.is_self_attn(i) for i in range(s.n_layers)]

            def heads(rows, on):
                shape = (1, s.n_heads, rows, s.head_dim)
                return [torch.zeros(shape, dtype=cdt, device=dev) if x else None for x in on]

            cross = [not x for x in self_attn]
            self.ks, self.vs = heads(L + H, self_attn), heads(L + H, self_attn)
            self.kx, self.vx = heads(L, cross), heads(L, cross)

        self.timing = Timing()
        self.trt = None
        if vision == "trt":
            from .trt_vision import TrtVision, plan_path

            plan = plan_path(bundle, trt_precision)
            if not plan.exists():
                raise SystemExit(
                    f"no TensorRT plan at {plan}; build it on this device: "
                    f"so101-smolvla-edge build-trt {bundle}"
                )
            rows = s.n_cams * s.img_tokens
            self.trt = TrtVision(plan, s.n_cams, s.image_size, rows, s.text_dim, dev)
            log.info("vision: TensorRT %s (%s)", trt_precision, plan.name)
        elif vision != "torch":
            raise ValueError(f"vision must be 'torch' or 'trt', not {vision!r}")
        self._encode = self.trt if self.trt is not None else self.model.encode_images

        self._stages = [self._vision, self._prefix, self._denoise]
        if self.tp is not None:
            self._stages = [self._vision, self._prefix_triton, self._denoise_triton]
        if compile:
            self._compile_stages(bundle, compile)
        self._fwd = self._forward
        self.graph = None
        if cuda_graph and dev.type == "cuda":
            self._capture(warmup)
        else:
            with torch.no_grad():
                for _ in range(warmup):
                    self._fwd()

    def _compile_stages(self, bundle, how: bool | str) -> None:
        """Compile the stages TensorRT and Triton leave to torch; stay eager if that fails.

        The cache goes next to the bundle unless TORCHINDUCTOR_CACHE_DIR is set, so a restart
        reuses it.
        """
        os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(Path(bundle).resolve() / "inductor"))
        covered = [self.trt is not None, self.tp is not None, self.tp is not None]
        try:
            compiled = [
                f if done else self._compile(f, how)
                for f, done in zip(self._stages, covered, strict=True)
            ]
            with torch.no_grad():
                rows = compiled[0]()
                compiled[1](rows)
                compiled[2]()
        except Exception as exc:  # noqa: BLE001 (any backend failure falls back the same way)
            name = type(exc).__name__
            log.warning("torch.compile unavailable (%s: %s); using eager kernels", name, exc)
            return
        self._stages = compiled
        log.info("torch.compile active")

    def _compile(self, stage, how: bool | str):
        """Compile one stage. "autotune" also autotunes the prefix stage's GEMMs.

        Inductor only autotunes GEMMs on GPUs with 68 or more SMs, so that gate is lifted for the
        prefix. The denoising stage keeps the default mode: its 50-row GEMMs get slower when
        autotuned. The first autotuned start is long; the cache makes later ones quick.
        """
        if how == "autotune" and stage == self._prefix:  # bound methods compare by function
            import torch._inductor.config as icfg
            import torch._inductor.utils as iutils

            iutils.is_big_gpu = lambda *a, **k: True
            icfg.max_autotune_gemm_backends = "ATEN,TRITON"
            return torch.compile(stage, mode="max-autotune-no-cudagraphs")
        return torch.compile(stage)

    # -------------------------------------------------------------------- the captured forward
    def _vision(self) -> torch.Tensor:
        """Turn the uint8 frames into image rows: resize with left/top padding, scale to [-1, 1]."""
        s = self.spec
        x = self.img.permute(0, 3, 1, 2).float() / 255.0
        _, _, h, w = x.shape
        ratio = max(w / s.image_size, h / s.image_size)
        rh, rw = int(h / ratio), int(w / ratio)
        x = F.interpolate(x, size=(rh, rw), mode="bilinear", align_corners=False)
        x = F.pad(x, (s.image_size - rw, 0, s.image_size - rh, 0), value=0.0)
        return self._encode(x * 2.0 - 1.0)

    def _prefix_rows(self, img_rows: torch.Tensor) -> torch.Tensor:
        s, m = self.spec, self.model
        st = (self.state - self.s_mean) / (self.s_std + 1e-8)
        st = F.pad(st, (0, s.max_state - st.shape[0]))
        st_row = F.linear(st, m.p("state.w"), m.p("state.b"))[None]
        return torch.cat([img_rows, self.lang, st_row], dim=0)

    def _prefix(self, img_rows: torch.Tensor) -> None:
        m = self.model
        rows = self._prefix_rows(img_rows)
        m.prefix_kv(rows, self.mask_p, self.cos_p, self.sin_p, self.k_pre, self.v_pre)
        m.expert_cache(self.k_pre, self.v_pre, self.ks, self.vs, self.kx, self.vx)

    def _prefix_triton(self, img_rows: torch.Tensor) -> None:
        self.tp.prefix(self._prefix_rows(img_rows))

    def step(self, k: int, x_t: torch.Tensor) -> torch.Tensor:
        """One Euler step of the denoising loop: x_t + dt * v_t."""
        if self.tp is not None:
            v = self.tp.velocity(x_t, self.time_proj[k])
        else:
            caches = (self.ks, self.vs, self.kx, self.vx)
            rope = (self.cos_s, self.sin_s, self.cos_c, self.sin_c)
            v = self.model.velocity(x_t, self.time_proj[k], *caches, self.mask_s, *rope)
        return x_t + self.dt * v

    def _denoise(self) -> None:
        x_t = self.noise
        for k in range(self.spec.n_steps):
            x_t = self.step(k, x_t)
        self._finish(x_t)

    _denoise_triton = _denoise  # the same loop under its own name, so compile can tell them apart

    def _finish(self, x_t: torch.Tensor) -> None:
        """Undo the action representation and the normalisation into the output buffer."""
        s = self.spec
        a = x_t[0]
        if self.repr != "absolute":
            st = (self.state - self.s_mean) / (self.s_std + 1e-8)
            st = F.pad(st, (0, s.max_state - self.state_dim))
            a = a * self.r_std + self.r_mean
            a = a + st[None] if self.repr == "delta" else st[None] + torch.cumsum(a, dim=0)
        a = a[:, : self.action_dim] * self.a_std + self.a_mean
        self.out.copy_(a)

    @torch.no_grad()
    def _forward(self) -> None:
        vision, prefix, denoise = self._stages
        prefix(vision())
        denoise()

    def _capture(self, warmup: int) -> None:
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(max(1, warmup)):
                self._fwd()
        torch.cuda.current_stream().wait_stream(side)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._fwd()
        torch.cuda.synchronize()

    # -------------------------------------------------------------------- public
    def infer(
        self, images: dict[str, np.ndarray], state: np.ndarray, noise: np.ndarray | None = None
    ) -> np.ndarray:
        """Return the action chunk (chunk, action_dim) for one observation.

        `images` maps each camera key to an HxWx3 uint8 RGB frame, `state` is (state_dim,) in
        radians. `noise` fixes the initial noise, for comparing against a reference.
        """
        t0 = time.perf_counter()
        for i, c in enumerate(self.cams):
            src = torch.from_numpy(np.ascontiguousarray(images[c]))
            if self._host_img is not None:
                self._host_img[i].copy_(src)
            else:
                self.img[i].copy_(src)
        if self._host_img is not None:
            self.img.copy_(self._host_img, non_blocking=True)
        st = torch.as_tensor(np.asarray(state, dtype=np.float32))
        self.state.copy_(st, non_blocking=True)
        if noise is None:
            self.noise.normal_()
        else:
            noise = torch.as_tensor(np.asarray(noise, dtype=np.float32))
            self.noise.copy_(noise.view_as(self.noise))
        if self.graph is not None:
            self.graph.replay()
        else:
            self._fwd()
        out = self.out.cpu().numpy()  # synchronises
        if self.device.type == "cpu":  # .cpu() is then a view of the static buffer
            out = out.copy()
        self.timing.add((time.perf_counter() - t0) * 1e3)
        return out

    @torch.no_grad()
    def profile(self, images: dict[str, np.ndarray], state: np.ndarray, reps: int = 20) -> dict:
        """Median GPU time in ms of each stage.

        Each stage is captured as its own CUDA graph, so the numbers add up to one replay of the
        whole chunk. Without a graph the stages run eagerly and launch gaps count.
        """
        self.infer(images, state)
        vision, prefix, denoise = self._stages
        rows = [vision()]

        def run_vision():
            rows[0] = vision()

        stages = {"vision": run_vision, "prefix": lambda: prefix(rows[0]), "denoise": denoise}
        if self.graph is not None:
            for name, fn in stages.items():
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    fn()
                stages[name] = g.replay
        res = {}
        for name, fn in stages.items():
            ts = []
            for _ in range(reps):
                e0 = torch.cuda.Event(enable_timing=True)
                e1 = torch.cuda.Event(enable_timing=True)
                e0.record()
                fn()
                e1.record()
                torch.cuda.synchronize()
                ts.append(e0.elapsed_time(e1))
            res[name] = float(np.median(ts))
        return res
