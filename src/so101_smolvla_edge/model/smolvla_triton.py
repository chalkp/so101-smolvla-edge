"""SmolVLA's prefix pass and denoising velocity on the Triton kernels (`Engine(kernels="triton")`).

Same weights and arithmetic as `smolvla.py`, which stays the reference this is tested against, in
seven kernels per transformer layer instead of about thirty:

    rms_norm, linear(qkv), rope_kv, attention, linear(o + residual),
    rms_norm, linear(gate_up, SiLU gate), linear(down + residual)

The expert's caches stay at the 5 kv heads in the token-major layout the projections write,
(len, n_kv * hd), which the attention kernel reads directly:

    self-attention layers    ck/cv[i] (L + H, kv): rows [:L] the VLM layer's keys and values,
                             written by the prefix pass; rows [L:] the step's own
    cross-attention layers   kx/vx[i] (L, kv): the expert's projections of the VLM layer's keys
                             and values, computed in the prefix pass

so `SmolVLA.expert_cache` folds into the prefix pass and no step copies keys to 15 heads.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..kernels.attention import attention
from ..kernels.gemm import _counters, linear
from ..kernels.norm import rms_norm
from ..kernels.rope import rope_kv
from .smolvla import SmolVLA, rope_tables


def unsupported(spec, cdt: torch.dtype) -> str | None:
    """Say why the Triton path cannot run this model, or return None.

    The attention and RoPE kernels index a head and its two halves with tl.arange, so the head size
    must be a power of two, and tl.dot needs each half at least 16 wide.
    """
    if cdt not in (torch.bfloat16, torch.float16):
        return "it runs bf16/fp16 matmuls (tl.dot on float32 would be TF32)"
    hd = spec.head_dim
    if hd < 32 or hd & (hd - 1):
        return f"head_dim {hd} is not a power of two >= 32"
    return None


class TritonPath:
    """The prefix pass and the velocity of one denoising step, with their own caches."""

    def __init__(self, m: SmolVLA, device: torch.device):
        s = self.spec = m.spec
        why = unsupported(s, m.cdt)
        if why:
            raise ValueError(f"the Triton path cannot run this model: {why}")
        _counters(device)  # split-K tickets must exist before any graph capture
        self.m = m
        L, H, kv = s.prefix_len, s.chunk, s.n_kv * s.head_dim

        def zeros(rows):
            return torch.zeros(rows, kv, dtype=m.cdt, device=device)

        def rope(pos):
            return rope_tables(pos, s.head_dim, s.rope_wavelength)

        layers = range(s.n_layers)
        self.ck = [zeros(L + H) if s.is_self_attn(i) else None for i in layers]
        self.cv = [zeros(L + H) if s.is_self_attn(i) else None for i in layers]
        self.kx = [None if s.is_self_attn(i) else zeros(L) for i in layers]
        self.vx = [None if s.is_self_attn(i) else zeros(L) for i in layers]
        self.k_scratch = zeros(L)  # a cross layer's rotated VLM keys, input to the expert's k
        self.cos_p, self.sin_p = rope(torch.arange(L, device=device))
        self.cos_s, self.sin_s = rope(torch.arange(L, L + H, device=device))
        self.cos_c, self.sin_c = rope(torch.arange(H, device=device))
        self.w = {name: m.p(name) for name in _weights(s)}

    def lin(self, x: torch.Tensor, name: str, **kw) -> torch.Tensor:
        return linear(x, self.w[name], **kw)

    def prefix(self, prefix: torch.Tensor) -> None:
        """Run the VLM layers over the prefix rows (L, text_dim), filling ck/cv[:L] and kx/vx.

        This is `SmolVLA.prefix_kv` followed by `SmolVLA.expert_cache`.
        """
        s, w, m = self.spec, self.w, self.m
        nq, nk, hd, L = s.n_heads, s.n_kv, s.head_dim, s.prefix_len
        x = prefix.to(m.rdt)  # (L, text_dim) residual stream
        for i in range(s.n_layers):
            h = rms_norm(x, w[f"txt.{i}.ln_in"], s.rms_eps, m.cdt)
            qkv = self.lin(h, f"txt.{i}.qkv")  # (L, (nq + 2 nk) hd)
            if s.is_self_attn(i):
                k, v = self.ck[i][:L], self.cv[i][:L]
                rope_kv(qkv, nq, nk, self.cos_p, self.sin_p, k, v)
            else:
                k, v = self.k_scratch, qkv[:, (nq + nk) * hd :]
                rope_kv(qkv, nq, nk, self.cos_p, self.sin_p, k)
                self.lin(k, f"exp.{i}.k", out=self.kx[i])
                self.lin(v, f"exp.{i}.v", out=self.vx[i])
            if i == s.n_layers - 1:
                break  # the last layer's output is never read
            a = attention(qkv, k, v, nq, nk, cos=self.cos_p, sin=self.sin_p, causal_from=L - 1)
            x = self.lin(a, f"txt.{i}.o", residual=x)
            h = rms_norm(x, w[f"txt.{i}.ln_post"], s.rms_eps, m.cdt)
            g = self.lin(h, f"txt.{i}.gate_up", silu_gate=True)
            x = self.lin(g, f"txt.{i}.down", residual=x)

    def velocity(self, x_t: torch.Tensor, time_proj: torch.Tensor) -> torch.Tensor:
        """Return v_t (1, chunk, max_action) float32 for x_t, as `SmolVLA.velocity` does."""
        s, w, m = self.spec, self.w, self.m
        nq, nk, L = s.n_heads, s.n_kv, s.prefix_len
        a_emb = F.linear(x_t, w["act.in.w"], w["act.in.b"])  # float32, as the reference
        h = F.linear(a_emb, w["act.time_in.w_action"]) + time_proj
        e = F.linear(F.silu(h), w["act.time_out.w"], w["act.time_out.b"])
        e = e.to(m.rdt)[0]  # (H, exp_dim)
        for i in range(s.n_layers):
            h = rms_norm(e, w[f"exp.{i}.ln_in"], s.rms_eps, m.cdt)
            if s.is_self_attn(i):
                qkv = self.lin(h, f"exp.{i}.qkv")
                rope_kv(qkv, nq, nk, self.cos_s, self.sin_s, self.ck[i][L:], self.cv[i][L:])
                a = attention(
                    qkv, self.ck[i], self.cv[i], nq, nk,
                    cos=self.cos_s, sin=self.sin_s, causal_from=L, q_offset=L,
                )  # fmt: skip
            else:
                q = self.lin(h, f"exp.{i}.q")
                a = attention(q, self.kx[i], self.vx[i], nq, nk, cos=self.cos_c, sin=self.sin_c)
            e = self.lin(a, f"exp.{i}.o", residual=e)
            h = rms_norm(e, w[f"exp.{i}.ln_post"], s.rms_eps, m.cdt)
            g = self.lin(h, f"exp.{i}.gate_up", silu_gate=True)
            e = self.lin(g, f"exp.{i}.down", residual=e)
        e = rms_norm(e, w["exp.norm"], s.rms_eps, torch.float32)
        return F.linear(e, w["act.out.w"], w["act.out.b"])[None]  # float32


def _weights(s) -> list[str]:
    """Names of every weight the two passes read."""
    names = ["act.in.w", "act.in.b", "act.time_in.w_action", "act.time_out.w", "act.time_out.b"]
    names += ["act.out.w", "act.out.b", "exp.norm"]
    for i in range(s.n_layers):
        names += [f"txt.{i}.{n}" for n in ("ln_in", "ln_post", "qkv", "o", "gate_up", "down")]
        names += [f"exp.{i}.{n}" for n in ("ln_in", "ln_post", "o", "gate_up", "down")]
        names += [f"exp.{i}.qkv"] if s.is_self_attn(i) else [f"exp.{i}.{n}" for n in "qkv"]
    return names
