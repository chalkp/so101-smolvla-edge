"""The OpenVINO graphs against the model they restate, in torch float32 on a small random model.
No bundle, GPU or OpenVINO needed. Whatever these graphs compute is what the NPU is asked for."""

import numpy as np
import pytest

# preloads the Jetson libraries, and a broken torch must fail here rather than skip
import so101_smolvla_edge.model  # noqa: F401

torch = pytest.importorskip("torch")


def stages(bundle, observation):
    """(graphs, feeds, references, (engine, time table)), the references from `Engine`'s stages."""
    import torch.nn.functional as F

    from so101_smolvla_edge.model import ov_graph as G
    from so101_smolvla_edge.model.engine import Engine

    eng = Engine(bundle, device="cpu", dtype="fp32", cuda_graph=False, warmup=0)
    m, s = eng.model, eng.spec
    L, g = s.prefix_len, s.n_heads // s.n_kv
    images, state, noise = observation(bundle)
    with torch.no_grad():
        for i, c in enumerate(eng.cams):
            eng.img[i].copy_(torch.from_numpy(images[c]))
        eng.state.copy_(torch.from_numpy(state))
        eng.noise.copy_(torch.from_numpy(noise)[None])
        x = eng.img.permute(0, 3, 1, 2).float() / 255.0
        resized = F.interpolate(x, size=(96, 128), mode="bilinear", align_corners=False)
        px = F.pad(resized, (0, 0, 32, 0)) * 2 - 1
        rows = m.encode_images(px)
        st = F.pad((eng.state - eng.s_mean) / (eng.s_std + 1e-8), (0, s.max_state - eng.state_dim))
        eng._prefix(rows)

        def pick(own, cross):
            # Engine keeps keys/values expanded to the query heads: every g-th head is one kv head
            return torch.cat(
                [
                    own[i][:, ::g, :L] if s.is_self_attn(i) else cross[i][:, ::g]
                    for i in range(s.n_layers)
                ]
            )

        K, V = pick(eng.ks, eng.kx), pick(eng.vs, eng.vx)
        v0 = m.velocity(
            eng.noise,
            eng.time_proj[1],
            eng.ks,
            eng.vs,
            eng.kx,
            eng.vx,
            eng.mask_s,
            eng.cos_s,
            eng.sin_s,
            eng.cos_c,
            eng.sin_c,
        )
        T = G.time_table(m)
    feeds = {"vision": (px,), "prefix": (rows, st), "step": (eng.noise.clone(), T[1], K, V)}
    refs = {"vision": (rows,), "prefix": (K, V), "step": (v0,)}
    return G.graphs(m, eng.lang), feeds, refs, (eng, T)


def test_graphs_compute_what_smolvla_computes(tiny_bundle, observation):
    graphs, feeds, refs, _ = stages(tiny_bundle(), observation)
    with torch.no_grad():
        for name, (module, _, _) in graphs.items():
            out = module(*feeds[name])
            for o, r in zip(out if isinstance(out, tuple) else (out,), refs[name], strict=True):
                assert o.shape == r.shape, name
                assert (o - r).abs().max() <= 1e-4 * max(1.0, r.abs().max().item()), name


def test_graph_shapes_are_the_declared_ones(tiny_bundle, observation):
    graphs, feeds, _, _ = stages(tiny_bundle(), observation)
    for name, (_, inputs, _) in graphs.items():
        got = [tuple(t.shape) for t in feeds[name]]
        if name == "vision":  # the graph is built for one camera, the module takes any number
            got = [(1, *got[0][1:])]
        assert got == [tuple(v) for v in inputs.values()], name


def test_time_table_is_the_engines(tiny_bundle, observation):
    _, _, _, (eng, T) = stages(tiny_bundle(), observation)
    assert torch.equal(T, eng.time_proj)


def test_folded_grouped_query_attention_is_sdpa_with_repeated_heads():
    """The folded attention against the reference: kv heads repeated for their query heads."""
    import torch.nn.functional as F

    from so101_smolvla_edge.model.ov_graph import _attn, _bias, _fold, _unfold

    g_, nk, hd, Lq, Lk = 3, 2, 8, 5, 7
    gen = torch.Generator().manual_seed(0)
    q = torch.randn(1, Lq, nk * g_, hd, generator=gen)
    k, v = torch.randn(1, nk, Lk, hd, generator=gen), torch.randn(1, nk, Lk, hd, generator=gen)
    mask = torch.rand(Lq, Lk, generator=gen) > 0.3
    mask[:, 0] = True
    ref = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.repeat_interleave(g_, 1),
        v.repeat_interleave(g_, 1),
        attn_mask=mask[None, None],
    )
    ref = ref.transpose(1, 2).reshape(1, Lq, -1)
    out = _unfold(_attn(_fold(q, g_), k, v, _bias(mask, g_)), g_)
    assert torch.allclose(out, ref, atol=1e-5)


def test_cpu_engine_returns_its_own_array(tiny_bundle, observation):
    """A returned chunk must not change when the next one is computed.

    On the CPU `tensor.cpu().numpy()` is a view of the engine's output buffer, and `verify` would
    then compare the last chunk with itself.
    """
    from so101_smolvla_edge.model.engine import Engine

    bundle = tiny_bundle()
    eng = Engine(bundle, device="cpu", dtype="fp32", cuda_graph=False, warmup=0)
    a = eng.infer(*observation(bundle, 0))
    kept = a.copy()
    b = eng.infer(*observation(bundle, 1))
    assert np.array_equal(a, kept) and not np.array_equal(a, b)


def test_grid_is_taller_than_wide_by_at_least_two():
    from so101_smolvla_edge.model.ov_graph import _grid

    assert _grid(1024) == (64, 16) and _grid(200) == (20, 10) and _grid(12) == (6, 2)
    assert _grid(199) is None and _grid(3) is None  # nothing to factor: the tokens stay a sequence


def test_mask_as_a_column_is_the_added_mask():
    """`_attn`'s two ways of masking, on the prefix's rank-one mask."""
    from so101_smolvla_edge.model.ov_graph import _AUG, NEG, _attn, _bias

    g_, nk, hd, L = 3, 2, 8, 7
    gen = torch.Generator().manual_seed(0)
    q, k, v = (torch.randn(1, nk, n, hd, generator=gen) for n in (g_ * L, L, L))
    mask = torch.ones(L, L, dtype=torch.bool)
    mask[:-1, -1] = False  # only the last token attends to the last token
    rows, cols = (~mask).any(1).float(), (~mask).any(0).float()

    def col(t):
        return t[None, None, :, None].expand(1, nk, -1, 1)

    a = _attn(q, k, v, aug=(col(_AUG * rows.repeat(g_)), col(NEG / _AUG * cols)))
    assert torch.allclose(a, _attn(q, k, v, bias=_bias(mask, g_)), atol=1e-6)


def test_prefix_adds_a_mask_that_is_not_rank_one(tiny_bundle, observation):
    """The reference's prefix mask goes into the matmul, and any other mask is still honoured."""
    from so101_smolvla_edge.model import ov_graph as G

    graphs, feeds, _, (eng, _) = stages(tiny_bundle(), observation)
    assert graphs["prefix"][0].rank1
    L = eng.spec.prefix_len
    causal = torch.tril(torch.ones(L, L, dtype=torch.bool))
    p = G.Prefix(eng.model, eng.lang, causal)
    assert not p.rank1
    k_out = [torch.zeros_like(t) for t in eng.k_pre]
    v_out = [torch.zeros_like(t) for t in eng.v_pre]
    with torch.no_grad():
        K, _ = p(*feeds["prefix"])
        rows, state = feeds["prefix"]
        st_row = torch.nn.functional.linear(state, eng.model.p("state.w"), eng.model.p("state.b"))
        prefix = torch.cat([rows, eng.lang, st_row[None]])
        eng.model.prefix_kv(prefix, causal, eng.cos_p, eng.sin_p, k_out, v_out)
    for i in range(eng.spec.n_layers):
        if eng.spec.is_self_attn(i):  # those layers hand the VLM's keys over as they are
            assert torch.allclose(K[i], k_out[i][0].transpose(0, 1), atol=1e-4)
