"""SmolVLA in plain PyTorch, the reference the optimized inference paths are tested against.

SigLIP tower, pixel-shuffle connector, a 16-layer SmolLM2 prefix and the interleaved action
expert, with nothing from transformers or lerobot. It computes what
`lehome_vla.policy.VLAFlowMatching.sample_actions` computes for the "absolute", "delta" and
"velocity" action representations (tests/test_model_parity.py checks this). Shapes are fixed and
the KV caches are plain tensors written in place, so the whole forward can be captured into a CUDA
graph. Work that is the same for every denoising step runs once per chunk, and projections that
read the same input are fused into one GEMM.

Norm statistics are float32, matmul inputs use the compute dtype (bfloat16 by default), and the
residual streams are float32 like the reference unless `residual="compute"`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass
class ModelSpec:
    """Architecture constants. Defaults are SmolVLA on SmolVLM2-500M with a 0.75x expert."""

    n_cams: int = 3
    image_size: int = 512  # square input, pixels
    patch: int = 16  # SigLIP patch side, pixels
    vis_dim: int = 768
    vis_layers: int = 12
    vis_heads: int = 12
    vis_mlp: int = 3072
    vis_eps: float = 1e-6
    shuffle: int = 4  # pixel-shuffle factor: 1024 patches become 64 tokens per image
    text_dim: int = 960
    n_layers: int = 16  # VLM layers kept, and expert layers
    n_heads: int = 15  # query heads, text and expert
    n_kv: int = 5  # key/value heads
    head_dim: int = 64
    text_mlp: int = 2560
    rms_eps: float = 1e-5
    exp_dim: int = 720
    exp_mlp: int = 2048
    self_attn_every: int = 2  # expert layers i % this == 0 self-attend, the rest cross-attend
    rope_wavelength: float = 10_000.0
    chunk: int = 50  # actions per chunk
    max_state: int = 32  # state padded to this width
    max_action: int = 32  # actions padded to this width
    n_steps: int = 10  # denoising steps per chunk
    min_period: float = 4e-3  # sinusoidal time embedding periods
    max_period: float = 4.0
    lang_len: int = 0  # task tokens, set from the bundle

    @property
    def img_tokens(self) -> int:
        """Prefix tokens per camera image."""
        return (self.image_size // self.patch) ** 2 // (self.shuffle**2)

    @property
    def prefix_len(self) -> int:
        """Image tokens, task tokens and the state token."""
        return self.n_cams * self.img_tokens + self.lang_len + 1

    def is_self_attn(self, layer: int) -> bool:
        """True if expert layer `layer` attends to its own rows as well as to the prefix."""
        return self.self_attn_every > 0 and layer % self.self_attn_every == 0


def rms_norm(x: Tensor, w: Tensor, eps: float) -> Tensor:
    """LlamaRMSNorm, computed in float32."""
    return F.rms_norm(x.float(), (x.shape[-1],), w, eps)


def rope_tables(positions: Tensor, head_dim: int, wavelength: float) -> tuple[Tensor, Tensor]:
    """RoPE cos/sin, each (L, 1, head_dim / 2), for positions (L,).

    Same tables as `lehome_vla.policy.smolvlm_with_expert.apply_rope`.
    """
    half = head_dim // 2
    exps = (2.0 / head_dim) * torch.arange(half, dtype=torch.float32, device=positions.device)
    radians = positions.float()[:, None] / (wavelength**exps)[None, :]
    return torch.cos(radians)[:, None, :], torch.sin(radians)[:, None, :]


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply RoPE to x (B, L, H, D) in float32 and return x's dtype, like the reference."""
    half = x.shape[-1] // 2
    dt = x.dtype
    x = x.float()
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).to(dt)


def sinusoidal(t: Tensor, dim: int, min_period: float, max_period: float) -> Tensor:
    """Sinusoidal time embedding (the reference's create_sinusoidal_pos_embedding), in float64."""
    frac = torch.linspace(0.0, 1.0, dim // 2, dtype=torch.float64, device=t.device)
    period = min_period * (max_period / min_period) ** frac
    x = t.double()[..., None] * (2 * math.pi / period)
    return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)


def pixel_shuffle(x: Tensor, s: int) -> Tensor:
    """SmolVLMConnector.pixel_shuffle, verbatim: (B, N, D) -> (B, N / s^2, D * s^2)."""
    b, n, d = x.shape
    h = w = int(n**0.5)
    x = x.view(b, h, w, d)
    x = x.view(b, h, w // s, d * s)
    x = x.permute(0, 2, 1, 3)
    x = x.reshape(b, w // s, h // s, d * s * s)
    x = x.permute(0, 2, 1, 3)
    return x.reshape(b, n // (s * s), d * s * s)


class SmolVLA(nn.Module):
    """SmolVLA's forward in stages, over parameters named as in `param_shapes`.

    Parameters are registered with "." spelled "__", since module attribute names cannot hold dots.
    `residual="fp32"` keeps the residual streams in float32 like the reference, "compute" keeps them
    in the compute dtype.
    """

    def __init__(
        self, spec: ModelSpec, compute_dtype: torch.dtype = torch.bfloat16, residual: str = "fp32"
    ):
        super().__init__()
        if residual not in ("fp32", "compute"):
            raise ValueError(f"residual must be 'fp32' or 'compute', not {residual!r}")
        self.spec = spec
        self.cdt = compute_dtype  # matmul inputs
        self.rdt = torch.float32 if residual == "fp32" else compute_dtype  # residual streams
        self._cast: dict[str, Tensor] = {}  # pr()'s residual-dtype copies, by name
        for name, (shape, dt) in param_shapes(spec, compute_dtype).items():
            self.register_parameter(
                name.replace(".", "__"),
                nn.Parameter(torch.empty(shape, dtype=dt), requires_grad=False),
            )

    def p(self, name: str) -> Tensor:
        """Parameter by its bundle name."""
        return getattr(self, name.replace(".", "__"))

    def pr(self, name: str) -> Tensor:
        """Parameter in the residual dtype, cast once and cached (LayerNorm affine terms)."""
        t = self._cast.get(name)
        if t is None:
            t = self._cast[name] = self.p(name).to(self.rdt)
        return t

    def lin(self, x: Tensor, w: str, b: str | None = None) -> Tensor:
        """Linear layer with input and bias cast to the weight's dtype."""
        W = self.p(w)
        out = F.linear(x.to(W.dtype), W, None if b is None else self.p(b).to(W.dtype))
        return out

    # ------------------------------------------------------------------ vision
    def encode_images(self, px: Tensor) -> Tensor:
        """Encode px (n_cams, 3, 512, 512) in [-1, 1] into (n_cams * 64, text_dim) prefix rows,
        already scaled by sqrt(text_dim).

        The patch embedding is a GEMM over unfolded patches rather than a strided conv, because
        cuDNN runs fp32 convs in TF32 by default.
        """
        s = self.spec
        n, g, P = px.shape[0], s.image_size // s.patch, s.patch
        patches = px.view(n, 3, g, P, g, P).permute(0, 2, 4, 1, 3, 5).reshape(n, g * g, 3 * P * P)
        # x: (N, 1024, 768)
        x = (self.lin(patches, "vis.patch.w", "vis.patch.b") + self.p("vis.pos")).to(self.rdt)
        L, d = x.shape[1], x.shape[2]
        hd = d // s.vis_heads

        def ln(t: Tensor, name: str) -> Tensor:
            return F.layer_norm(t, (d,), self.pr(name + ".w"), self.pr(name + ".b"), s.vis_eps)

        for i in range(s.vis_layers):
            qkv = self.lin(ln(x, f"vis.{i}.ln1"), f"vis.{i}.qkv.w", f"vis.{i}.qkv.b").view(
                n, L, 3, s.vis_heads, hd
            )
            q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)  # (N, heads, L, hd) each
            a = F.scaled_dot_product_attention(q, k, v)
            x = x + self.lin(a.transpose(1, 2).reshape(n, L, d), f"vis.{i}.out.w", f"vis.{i}.out.b")
            h = F.gelu(
                self.lin(ln(x, f"vis.{i}.ln2"), f"vis.{i}.fc1.w", f"vis.{i}.fc1.b"),
                approximate="tanh",
            )
            x = x + self.lin(h, f"vis.{i}.fc2.w", f"vis.{i}.fc2.b")
        x = self.lin(pixel_shuffle(ln(x, "vis.post_ln"), s.shuffle), "conn.w")  # (N, 64, 960)
        return (x.float() * math.sqrt(s.text_dim)).reshape(n * x.shape[1], s.text_dim)

    # ------------------------------------------------------------------ prefix
    def prefix_kv(
        self,
        prefix: Tensor,
        mask: Tensor,
        cos: Tensor,
        sin: Tensor,
        k_out: list[Tensor],
        v_out: list[Tensor],
    ) -> None:
        """Run the VLM layers over prefix (L, text_dim) and write each layer's K (after RoPE) and V,
        both (1, L, n_kv, hd), into k_out/v_out in place. `mask` is (L, L) bool, True = attend."""
        s = self.spec
        x = prefix[None].to(self.rdt)  # (1, L, 960)
        L = x.shape[1]
        nq, nk, hd = s.n_heads, s.n_kv, s.head_dim
        for i in range(s.n_layers):
            h = rms_norm(x, self.p(f"txt.{i}.ln_in"), s.rms_eps)
            qkv = self.lin(h, f"txt.{i}.qkv")
            q, k, v = qkv.split([nq * hd, nk * hd, nk * hd], dim=-1)
            k = apply_rope(k.view(1, L, nk, hd), cos, sin)
            k_out[i].copy_(k)
            v_out[i].copy_(v.view(1, L, nk, hd))
            if i == s.n_layers - 1:
                break  # only the last layer's K/V are read, not its output
            q = apply_rope(q.view(1, L, nq, hd), cos, sin)
            a = _attend(q, k, v.view(1, L, nk, hd), mask, nq // nk, self.cdt)
            x = x + self.lin(a, f"txt.{i}.o")
            h = rms_norm(x, self.p(f"txt.{i}.ln_post"), s.rms_eps)
            g, u = self.lin(h, f"txt.{i}.gate_up").chunk(2, dim=-1)
            x = x + self.lin(F.silu(g) * u, f"txt.{i}.down")

    def expert_cache(
        self,
        k_pre: list[Tensor],
        v_pre: list[Tensor],
        ks: list[Tensor | None],
        vs: list[Tensor | None],
        kx: list[Tensor | None],
        vx: list[Tensor | None],
    ) -> None:
        """Fill the prefix side of every expert layer's K/V buffers, once per chunk.

        Buffers are in SDPA's (1, n_heads, len, hd) layout with the kv heads already repeated for
        the query heads, so a denoising step only expands its own rows:

            ks/vs[i]  self-attention layers, (1, nq, L + H, hd): rows [:L] get the VLM layer's
                      cached keys/values, each denoising step writes its own rows [L:]
            kx/vx[i]  cross-attention layers, (1, nq, L, hd): the expert's k/v projections of them
        """
        s = self.spec
        g = s.n_heads // s.n_kv
        for i in range(s.n_layers):
            L = k_pre[i].shape[1]
            if s.is_self_attn(i):
                k, v = k_pre[i], v_pre[i]
            else:
                k = self.lin(k_pre[i].reshape(1, L, -1), f"exp.{i}.k").view(
                    1, L, s.n_kv, s.head_dim
                )
                v = self.lin(v_pre[i].reshape(1, L, -1), f"exp.{i}.v").view(
                    1, L, s.n_kv, s.head_dim
                )
            dk, dv = (ks[i], vs[i]) if s.is_self_attn(i) else (kx[i], vx[i])
            dk[:, :, :L].copy_(k.transpose(1, 2).repeat_interleave(g, dim=1))
            dv[:, :, :L].copy_(v.transpose(1, 2).repeat_interleave(g, dim=1))

    # ------------------------------------------------------------------ denoising
    def velocity(
        self,
        x_t: Tensor,
        time_proj: Tensor,
        ks: list[Tensor | None],
        vs: list[Tensor | None],
        kx: list[Tensor | None],
        vx: list[Tensor | None],
        self_mask: Tensor,
        cos_s: Tensor,
        sin_s: Tensor,
        cos_c: Tensor,
        sin_c: Tensor,
    ) -> Tensor:
        """Flow velocity v_t (1, chunk, 32) for the noisy actions x_t (1, chunk, 32), in float32.

        `time_proj` is this step's time-embedding half of action_time_mlp_in, bias included.
        ks/vs/kx/vx come from `expert_cache`, and self-attention layers write this step's rows
        into ks/vs.
        """
        s = self.spec
        nq, nk, hd, g = s.n_heads, s.n_kv, s.head_dim, s.n_heads // s.n_kv
        a_emb = F.linear(x_t, self.p("act.in.w"), self.p("act.in.b"))  # fp32
        h = F.linear(a_emb, self.p("act.time_in.w_action")) + time_proj
        # e: (1, H, 720)
        e = F.linear(F.silu(h), self.p("act.time_out.w"), self.p("act.time_out.b")).to(self.rdt)
        H = e.shape[1]
        for i in range(s.n_layers):
            h = rms_norm(e, self.p(f"exp.{i}.ln_in"), s.rms_eps)
            if s.is_self_attn(i):
                q, k, v = self.lin(h, f"exp.{i}.qkv").split([nq * hd, nk * hd, nk * hd], dim=-1)
                q = apply_rope(q.view(1, H, nq, hd), cos_s, sin_s).transpose(1, 2)
                k = apply_rope(k.view(1, H, nk, hd), cos_s, sin_s)
                Lk = ks[i].shape[2]
                ks[i][:, :, Lk - H :].copy_(k.transpose(1, 2).repeat_interleave(g, dim=1))
                vs[i][:, :, Lk - H :].copy_(
                    v.view(1, H, nk, hd).transpose(1, 2).repeat_interleave(g, dim=1)
                )
                a = F.scaled_dot_product_attention(q, ks[i], vs[i], attn_mask=self_mask[None, None])
            else:
                q = apply_rope(
                    self.lin(h, f"exp.{i}.q").view(1, H, nq, hd), cos_c, sin_c
                ).transpose(1, 2)
                a = F.scaled_dot_product_attention(q, kx[i], vx[i])
            e = e + self.lin(a.transpose(1, 2).reshape(1, H, nq * hd), f"exp.{i}.o")
            h = rms_norm(e, self.p(f"exp.{i}.ln_post"), s.rms_eps)
            gt, u = self.lin(h, f"exp.{i}.gate_up").chunk(2, dim=-1)
            e = e + self.lin(F.silu(gt) * u, f"exp.{i}.down")
        e = rms_norm(e, self.p("exp.norm"), s.rms_eps)
        return F.linear(e, self.p("act.out.w"), self.p("act.out.b"))  # fp32 (1, H, 32)


def _attend(
    q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None, groups: int, cdt: torch.dtype
) -> Tensor:
    """Grouped-query attention: q (1, Lq, nq, hd), k/v (1, Lk, nk, hd) -> (1, Lq, nq * hd).

    Each kv head is repeated `groups` times consecutively, the reference's expand-and-reshape order.
    """
    q = q.transpose(1, 2).to(cdt)
    k = k.transpose(1, 2).to(cdt).repeat_interleave(groups, dim=1)
    v = v.transpose(1, 2).to(cdt).repeat_interleave(groups, dim=1)
    a = F.scaled_dot_product_attention(
        q, k, v, attn_mask=None if mask is None else mask[None, None]
    )
    return a.transpose(1, 2).reshape(1, q.shape[2], -1)


def param_shapes(
    s: ModelSpec, weight_dtype: torch.dtype = torch.bfloat16
) -> dict[str, tuple[tuple, torch.dtype]]:
    """(shape, dtype) of every model parameter, by bundle name.

    Norms, biases, the position table and the small state, action and time projections stay
    float32; the big matrices use `weight_dtype`.
    """
    f32, w = torch.float32, weight_dtype
    out: dict[str, tuple[tuple, torch.dtype]] = {
        "vis.patch.w": ((s.vis_dim, 3 * s.patch * s.patch), w),
        "vis.patch.b": ((s.vis_dim,), f32),
        "vis.pos": (((s.image_size // s.patch) ** 2, s.vis_dim), f32),
        "vis.post_ln.w": ((s.vis_dim,), f32),
        "vis.post_ln.b": ((s.vis_dim,), f32),
        "conn.w": ((s.text_dim, s.vis_dim * s.shuffle**2), w),
        "state.w": ((s.text_dim, s.max_state), f32),
        "state.b": ((s.text_dim,), f32),
        "act.in.w": ((s.exp_dim, s.max_action), f32),
        "act.in.b": ((s.exp_dim,), f32),
        "act.time_in.w_action": ((s.exp_dim, s.exp_dim), f32),
        "act.time_in.w_time": ((s.exp_dim, s.exp_dim), f32),
        "act.time_in.b": ((s.exp_dim,), f32),
        "act.time_out.w": ((s.exp_dim, s.exp_dim), f32),
        "act.time_out.b": ((s.exp_dim,), f32),
        "act.out.w": ((s.max_action, s.exp_dim), f32),
        "act.out.b": ((s.max_action,), f32),
        "exp.norm": ((s.exp_dim,), f32),
    }
    for i in range(s.vis_layers):
        out.update(
            {
                f"vis.{i}.ln1.w": ((s.vis_dim,), f32),
                f"vis.{i}.ln1.b": ((s.vis_dim,), f32),
                f"vis.{i}.ln2.w": ((s.vis_dim,), f32),
                f"vis.{i}.ln2.b": ((s.vis_dim,), f32),
                f"vis.{i}.qkv.w": ((3 * s.vis_dim, s.vis_dim), w),
                f"vis.{i}.qkv.b": ((3 * s.vis_dim,), f32),
                f"vis.{i}.out.w": ((s.vis_dim, s.vis_dim), w),
                f"vis.{i}.out.b": ((s.vis_dim,), f32),
                f"vis.{i}.fc1.w": ((s.vis_mlp, s.vis_dim), w),
                f"vis.{i}.fc1.b": ((s.vis_mlp,), f32),
                f"vis.{i}.fc2.w": ((s.vis_dim, s.vis_mlp), w),
                f"vis.{i}.fc2.b": ((s.vis_dim,), f32),
            }
        )
    kv = s.n_kv * s.head_dim
    for i in range(s.n_layers):
        out.update(
            {
                f"txt.{i}.ln_in": ((s.text_dim,), f32),
                f"txt.{i}.ln_post": ((s.text_dim,), f32),
                f"txt.{i}.qkv": ((s.n_heads * s.head_dim + 2 * kv, s.text_dim), w),
                f"txt.{i}.o": ((s.text_dim, s.n_heads * s.head_dim), w),
                f"txt.{i}.gate_up": ((2 * s.text_mlp, s.text_dim), w),
                f"txt.{i}.down": ((s.text_dim, s.text_mlp), w),
                f"exp.{i}.ln_in": ((s.exp_dim,), f32),
                f"exp.{i}.ln_post": ((s.exp_dim,), f32),
                f"exp.{i}.o": ((s.exp_dim, s.n_heads * s.head_dim), w),
                f"exp.{i}.gate_up": ((2 * s.exp_mlp, s.exp_dim), w),
                f"exp.{i}.down": ((s.exp_dim, s.exp_mlp), w),
            }
        )
        if s.is_self_attn(i):
            out[f"exp.{i}.qkv"] = ((s.n_heads * s.head_dim + 2 * kv, s.exp_dim), w)
        else:
            out[f"exp.{i}.q"] = ((s.n_heads * s.head_dim, s.exp_dim), w)
            out[f"exp.{i}.k"] = ((kv, kv), w)
            out[f"exp.{i}.v"] = ((kv, kv), w)
    return out
