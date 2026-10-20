"""Each Triton kernel against the torch code it replaces, on the model's shapes and on odd ones.

Needs CUDA and Triton, skipped elsewhere.
"""

import pytest

torch = pytest.importorskip("torch")
from so101_smolvla_edge import kernels  # noqa: E402

pytestmark = pytest.mark.skipif(not kernels.available(), reason="needs CUDA and Triton")
F = torch.nn.functional


def rnd(*shape, dtype=torch.float32, scale=1.0, seed=0):
    """Seeded normal noise on the GPU."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(*shape, generator=g, device="cuda") * scale).to(dtype)


@pytest.mark.parametrize("rows,n", [(50, 720), (200, 960), (7, 33), (1, 2048)])
@pytest.mark.parametrize("in_dtype", [torch.float32, torch.bfloat16])
def test_rms_norm(rows, n, in_dtype):
    """rms_norm is F.rms_norm in float32, rounded once to bfloat16."""
    from so101_smolvla_edge.kernels.norm import rms_norm

    x = rnd(1, rows, n, scale=30.0).to(in_dtype)  # residual-stream magnitudes
    w = 1 + 0.1 * rnd(n, seed=1)
    want = F.rms_norm(x.float(), (n,), w, 1e-5).to(torch.bfloat16)
    got = rms_norm(x, w, 1e-5)
    assert got.shape == want.shape and got.dtype == torch.bfloat16
    # one bfloat16 rounding of the same float32 value; reduction order may flip the last bit
    assert (got.float() - want.float()).abs().max() <= 2**-7 * want.float().abs().max()
    assert (got.float() - want.float()).abs().mean() <= 1e-3 * want.float().abs().mean()


# (M, K, N): the expert's four projections, the prefix's qkv, then odd sizes
GEMM_SHAPES = [
    (50, 720, 1600),
    (50, 960, 720),
    (50, 720, 4096),
    (50, 2048, 720),
    (200, 960, 1600),
    (7, 100, 33),
    (64, 64, 64),
    (1, 4096, 8),
]


def gemm_ref(x, w, bias=None, residual=None, silu_gate=False):
    """The fused GEMM and its epilogue in float32 torch."""
    y = x.float() @ w.float().T
    if bias is not None:
        y = y + bias.float()
    if silu_gate:
        g, u = y.chunk(2, dim=-1)
        y = F.silu(g) * u
    if residual is not None:
        y = y + residual.float()
    return y


def close(got, want, out_dtype):
    """One rounding to the output dtype plus float32 accumulation-order noise."""
    eps = 2**-8 if out_dtype == torch.bfloat16 else 1e-5
    err = (got.float() - want).abs()
    return bool(err.max() <= 2 * eps * want.abs().max() + 1e-4)


@pytest.mark.parametrize("m,k,n", GEMM_SHAPES)
@pytest.mark.parametrize("epi", ["none", "bias", "residual", "residual_bf16", "silu_gate"])
def test_linear(m, k, n, epi):
    """linear with each epilogue matches the float32 reference within one output rounding."""
    from so101_smolvla_edge.kernels.gemm import linear

    if epi == "silu_gate" and n % 2:
        n += 1
    x = rnd(m, k, dtype=torch.bfloat16)
    w = rnd(n, k, dtype=torch.bfloat16, scale=k**-0.5, seed=1)
    kw = {}
    if epi == "bias":
        kw["bias"] = rnd(n, seed=2)
    if epi.startswith("residual"):
        res_dtype = torch.bfloat16 if epi == "residual_bf16" else torch.float32
        kw["residual"] = rnd(m, n, seed=3, scale=10.0).to(res_dtype)
    if epi == "silu_gate":
        kw["silu_gate"] = True
    got = linear(x, w, **kw)
    want = gemm_ref(x, w, **kw)
    assert got.shape == want.shape
    assert close(got, want, got.dtype), (got.float() - want).abs().max()


@pytest.mark.parametrize("split", [1, 2, 3, 7])
def test_linear_split_k_is_deterministic_and_agrees(split):
    """Split-K gives the same bits on every run and agrees with the reference."""
    from so101_smolvla_edge.kernels.gemm import Config, linear

    x = rnd(50, 2048, dtype=torch.bfloat16)
    w = rnd(720, 2048, dtype=torch.bfloat16, scale=2048**-0.5, seed=1)
    r = rnd(50, 720, seed=2)
    cfg = Config(64, 32, 128, split)
    a = linear(x, w, residual=r, config=cfg)
    b = linear(x, w, residual=r, config=cfg)
    assert torch.equal(a, b)
    assert close(a, gemm_ref(x, w, residual=r), torch.float32)


def test_linear_takes_float32_input_like_lin():
    """A float32 input is cast to the weight dtype on load, as SmolVLA.lin casts it."""
    from so101_smolvla_edge.kernels.gemm import linear

    x = rnd(50, 720)
    w = rnd(960, 720, dtype=torch.bfloat16, scale=720**-0.5, seed=1)
    assert torch.equal(linear(x, w), linear(x.to(torch.bfloat16), w))


def attention_ref(qkv_q, k, v, n_heads, n_kv, mask, cos=None, sin=None):
    """Attention as SmolVLA computes it: q rotated in float32, k/v expanded to the query heads,
    SDPA with a bool mask."""
    from so101_smolvla_edge.model.smolvla import apply_rope

    lq, lk, hd = qkv_q.shape[0], k.shape[0], k.shape[1] // n_kv
    q = qkv_q[:, : n_heads * hd].reshape(1, lq, n_heads, hd)
    if cos is not None:
        q = apply_rope(q, cos, sin)
    g = n_heads // n_kv
    kk = k.reshape(1, lk, n_kv, hd).transpose(1, 2).repeat_interleave(g, dim=1)
    vv = v.reshape(1, lk, n_kv, hd).transpose(1, 2).repeat_interleave(g, dim=1)
    attn_mask = None if mask is None else mask[None, None]
    a = F.scaled_dot_product_attention(q.transpose(1, 2), kk, vv, attn_mask=attn_mask)
    return a.transpose(1, 2).reshape(lq, n_heads * hd)


def masks(L, H):
    """The engine's prefix and suffix masks for prefix length L and chunk H."""
    from so101_smolvla_edge.model.engine import _prefix_mask, _suffix_mask
    from so101_smolvla_edge.model.smolvla import ModelSpec

    spec = ModelSpec(n_cams=1, image_size=64, patch=16, shuffle=2, chunk=H, lang_len=L - 5)
    assert spec.prefix_len == L
    return _prefix_mask(spec, "cuda"), _suffix_mask(spec, "cuda")


@pytest.mark.parametrize("case", ["expert_self", "expert_cross", "prefix", "small"])
@pytest.mark.parametrize("sms", [None, 8])
def test_attention(case, sms, monkeypatch):
    """attention matches the SDPA reference and repeats bit for bit; sms=8 uses Orin block sizes."""
    from so101_smolvla_edge.kernels import attention as A
    from so101_smolvla_edge.kernels.attention import attention
    from so101_smolvla_edge.model.smolvla import rope_tables

    if sms:
        monkeypatch.setattr(A, "sm_count", lambda *_: sms)

    nq, nk, hd = (15, 5, 64) if case != "small" else (4, 2, 32)
    L, H = (200, 50) if case != "small" else (13, 7)
    p_mask, s_mask = masks(L, H)
    if case == "expert_self":
        lq, lk, pos, mask = H, L + H, torch.arange(L, L + H), s_mask
        kw = {"causal_from": L, "q_offset": L}
    elif case == "expert_cross":
        lq, lk, pos, mask, kw = H, L, torch.arange(H), None, {}
    else:
        lq, lk, pos, mask, kw = L, L, torch.arange(L), p_mask, {"causal_from": L - 1}
    qkv = rnd(lq, (nq + 2 * nk) * hd, dtype=torch.bfloat16, scale=2.0)
    k = rnd(lk, nk * hd, dtype=torch.bfloat16, scale=2.0, seed=1)
    v = rnd(lk, nk * hd, dtype=torch.bfloat16, seed=2)
    cos, sin = rope_tables(pos.cuda(), hd, 10_000.0)
    got = attention(qkv, k, v, nq, nk, cos=cos, sin=sin, **kw)
    want = attention_ref(qkv, k, v, nq, nk, mask, cos, sin)
    err = (got.float() - want.float()).abs()
    assert (
        err.max() <= 0.02 * want.float().abs().max()
        and err.mean() <= 2e-3 * want.float().abs().mean() + 1e-4
    ), err.max()
    assert torch.equal(got, attention(qkv, k, v, nq, nk, cos=cos, sin=sin, **kw))


def test_attention_strided_values():
    """attention takes k and v as views into the fused qkv output, rows wider than the heads."""
    from so101_smolvla_edge.kernels.attention import attention

    nq, nk, hd, L = 15, 5, 64, 200
    qkv = rnd(L, (nq + 2 * nk) * hd, dtype=torch.bfloat16)
    k, v = qkv[:, nq * hd : (nq + nk) * hd], qkv[:, (nq + nk) * hd :]
    got = attention(qkv, k, v, nq, nk, causal_from=L - 1)
    want = attention_ref(qkv, k.contiguous(), v.contiguous(), nq, nk, masks(L, 50)[0])
    assert (got.float() - want.float()).abs().max() <= 0.02 * want.float().abs().max()


@pytest.mark.parametrize("rows,offset", [(50, 200), (200, 0), (7, 3)])
def test_rope_kv(rows, offset):
    """rope_kv writes rotated k and plain v into their cache rows and nothing else."""
    from so101_smolvla_edge.kernels.rope import rope_kv
    from so101_smolvla_edge.model.smolvla import apply_rope, rope_tables

    nq, nk, hd = 15, 5, 64
    qkv = rnd(rows, (nq + 2 * nk) * hd, dtype=torch.bfloat16, scale=3.0)
    cos, sin = rope_tables(torch.arange(offset, offset + rows, device="cuda"), hd, 10_000.0)
    cache_k = torch.zeros(offset + rows + 5, nk * hd, dtype=torch.bfloat16, device="cuda")
    cache_v = torch.zeros_like(cache_k)
    rope_kv(qkv, nq, nk, cos, sin, cache_k[offset : offset + rows], cache_v[offset : offset + rows])
    k = qkv[:, nq * hd : (nq + nk) * hd].reshape(1, rows, nk, hd)
    want_k = apply_rope(k, cos, sin).reshape(rows, nk * hd).float()
    got_k = cache_k[offset : offset + rows].float()
    # the same float32 rotation, contracted to FMAs: at most the last bf16 bit differs, rarely
    assert ((got_k - want_k).abs() <= 2**-7 * want_k.abs()).all()
    assert (got_k != want_k).float().mean() < 0.01
    assert torch.equal(cache_v[offset : offset + rows], qkv[:, (nq + nk) * hd :])
    assert not cache_k[:offset].any() and not cache_k[offset + rows :].any()
