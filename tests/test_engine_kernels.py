"""The engine's choice of kernels where the Triton ones cannot run (no CUDA or no Triton).

Runs on any torch install, including the CPU-only `npu` environment, which has no Triton at all.
"""

import pytest

pytest.importorskip("torch")


def test_explicit_triton_without_cuda_or_triton_keeps_the_torch_kernels(tiny_bundle, caplog):
    """kernels="triton" on the CPU keeps the torch kernels and logs why."""
    from so101_smolvla_edge.model.engine import Engine

    e = Engine(
        tiny_bundle(), device="cpu", dtype="fp32", kernels="triton", cuda_graph=False, warmup=0
    )
    assert e.kernels == "torch" and e.tp is None
    assert "using the torch kernels" in caplog.text


def test_auto_on_a_cpu_is_torch(tiny_bundle):
    """auto on the CPU picks the torch kernels."""
    from so101_smolvla_edge.model.engine import Engine

    assert (
        Engine(tiny_bundle(), device="cpu", dtype="fp32", cuda_graph=False, warmup=0).kernels
        == "torch"
    )
