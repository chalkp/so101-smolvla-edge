"""The engine with kernels="triton" against the torch kernels, end to end, on a small random model
whose head size the Triton kernels take. Needs CUDA and Triton, no bundle."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from so101_smolvla_edge import kernels  # noqa: E402

pytestmark = pytest.mark.skipif(not kernels.available(), reason="needs CUDA and Triton")

# widths are multiples of 16 and head_dim is 32: tl.dot needs at least 16 per rotary half
SMALL = dict(
    text_dim=192, n_heads=6, n_kv=2, head_dim=32, text_mlp=256, exp_dim=96, exp_mlp=160, chunk=10
)


def engines(bundle, residual="fp32"):
    """The bundle as an fp32 reference and as bf16 engines on the torch and the Triton kernels."""
    from so101_smolvla_edge.model.engine import Engine

    def mk(**kw):
        return Engine(bundle, device="cuda", warmup=1, residual=residual, **kw)

    return {
        "fp32": mk(dtype="fp32", cuda_graph=False),
        "torch": mk(dtype="bf16", kernels="torch"),
        "triton": mk(dtype="bf16", kernels="triton"),
        "triton_eager": mk(dtype="bf16", kernels="triton", cuda_graph=False),
    }


@pytest.mark.parametrize("action_repr", ["absolute", "delta", "velocity"])
@pytest.mark.parametrize("residual", ["fp32", "compute"])
def test_triton_engine_is_the_torch_engine(tiny_bundle, observation, action_repr, residual):
    """Triton graph replay equals Triton eager bit for bit and is as close to fp32 as torch is."""
    bundle = tiny_bundle(action_repr, **SMALL)
    eng = engines(bundle, residual)
    assert eng["triton"].kernels == "triton" and eng["triton"].tp is not None
    for seed in (0, 1):
        obs = observation(bundle, seed)
        out = {k: e.infer(*obs[:2], noise=obs[2]) for k, e in eng.items()}
        assert np.array_equal(out["triton"], out["triton_eager"])
        err_triton = np.abs(out["triton"] - out["fp32"]).max()
        err_torch = np.abs(out["torch"] - out["fp32"]).max()
        scale = np.abs(out["fp32"]).max()
        assert err_triton <= 1.5 * err_torch + 1e-3 * scale, (err_triton, err_torch)


def test_triton_needs_bf16_and_falls_back_otherwise(tiny_bundle):
    """kernels="triton" in fp32 falls back to the torch kernels."""
    from so101_smolvla_edge.model.engine import Engine

    e = Engine(
        tiny_bundle(**SMALL),
        device="cuda",
        dtype="fp32",
        kernels="triton",
        cuda_graph=False,
        warmup=0,
    )
    assert e.kernels == "torch" and e.tp is None


def test_triton_path_keeps_no_expanded_caches(tiny_bundle):
    """The Triton path allocates none of the torch path's per-query-head caches, only its own at
    the kv heads."""
    from so101_smolvla_edge.model.engine import Engine

    e = Engine(tiny_bundle(**SMALL), device="cuda", kernels="triton", warmup=0, cuda_graph=False)
    assert not hasattr(e, "ks") and all(
        c.shape[1] == e.spec.n_kv * e.spec.head_dim for c in e.tp.ck if c is not None
    )


def test_auto_picks_triton_where_it_can_run(tiny_bundle):
    """auto picks Triton for bf16 on a CUDA GPU and torch for fp32 or on the CPU."""
    from so101_smolvla_edge.model import engine as E

    bundle = tiny_bundle(**SMALL)
    assert E.Engine(bundle, device="cuda", warmup=0, cuda_graph=False).kernels == "triton"
    assert (
        E.Engine(bundle, device="cuda", dtype="fp32", warmup=0, cuda_graph=False).kernels == "torch"
    )
    assert (
        E.Engine(bundle, device="cpu", dtype="fp32", warmup=0, cuda_graph=False).kernels == "torch"
    )


def test_head_sizes_the_kernels_cannot_index_fall_back(tiny_bundle):
    """With head_dim 8 (the default tiny model) auto and explicit triton keep the torch kernels."""
    from so101_smolvla_edge.model.engine import Engine

    bundle = tiny_bundle()
    assert Engine(bundle, device="cuda", warmup=0, cuda_graph=False).kernels == "torch"
    assert (
        Engine(bundle, device="cuda", kernels="triton", warmup=0, cuda_graph=False).kernels
        == "torch"
    )
