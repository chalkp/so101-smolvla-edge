"""Engine actions against the reference actions stored in a bundle.

Needs CUDA and SO101_BUNDLE=<bundle dir>.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

# preloads the Jetson libraries; a broken torch should fail this import, not skip the tests
import so101_smolvla_edge.model  # noqa: F401

torch = pytest.importorskip("torch")
BUNDLE = os.environ.get("SO101_BUNDLE")
pytestmark = pytest.mark.skipif(
    not (BUNDLE and torch.cuda.is_available()), reason="needs CUDA and SO101_BUNDLE"
)


def run(dtype, graph, residual="fp32", kernels="torch"):
    """Actions for every golden observation of the bundle, and the golden arrays."""
    from so101_smolvla_edge.model.engine import Engine

    g = dict(np.load(Path(BUNDLE) / "golden.npz"))
    cams = json.loads((Path(BUNDLE) / "meta.json").read_text())["cameras"]
    eng = Engine(
        BUNDLE, dtype=dtype, residual=residual, cuda_graph=graph, warmup=1, kernels=kernels
    )
    out = np.stack(
        [
            eng.infer({c: g[f"img.{c}"][i] for c in cams}, g["state"][i], noise=g["noise"][i])
            for i in range(len(g["state"]))
        ]
    )
    return out, g


def test_fp32_matches_reference():
    """fp32 eager reproduces the bundle's fp32 reference actions."""
    out, g = run("fp32", graph=False)
    assert np.abs(out - g["actions_fp32"]).max() < 1e-4


@pytest.mark.parametrize("residual", ["fp32", "compute"])
def test_bf16_graph_is_eager_and_close(residual):
    """bf16 graph replay equals eager exactly and stays close to the fp32 reference."""
    a, g = run("bf16", graph=True, residual=residual)
    b, _ = run("bf16", graph=False, residual=residual)
    assert np.array_equal(a, b)
    assert np.abs(a - g["actions_fp32"]).max() < 3e-2


@pytest.mark.parametrize("residual", ["fp32", "compute"])
def test_triton_graph_is_eager_and_close(residual):
    """The same with the Triton kernels: graph equals eager, close to the fp32 reference."""
    from so101_smolvla_edge import kernels

    if not kernels.available():
        pytest.skip("needs Triton")
    a, g = run("bf16", graph=True, residual=residual, kernels="triton")
    b, _ = run("bf16", graph=False, residual=residual, kernels="triton")
    assert np.array_equal(a, b)
    assert np.abs(a - g["actions_fp32"]).max() < 3e-2
