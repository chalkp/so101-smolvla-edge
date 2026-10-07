"""Shared fixtures: lerobot calibration files, tiny SmolVLA bundles and matching observations."""

import json

import pytest

from so101_smolvla_edge.robot.calibration import MOTORS


@pytest.fixture
def calib_file(tmp_path):
    """make(name, range_min, range_max, drive_mode) -> path of a lerobot calibration JSON.

    All six motors get the same range and drive mode."""

    def make(name="left", range_min=600, range_max=3500, drive_mode=0):
        p = tmp_path / f"{name}.json"
        p.write_text(
            json.dumps(
                {
                    m: {
                        "id": i + 1,
                        "drive_mode": drive_mode,
                        "homing_offset": 0,
                        "range_min": range_min,
                        "range_max": range_max,
                    }
                    for i, m in enumerate(MOTORS)
                }
            )
        )
        return p

    return make


@pytest.fixture
def tiny_bundle(tmp_path):
    """make(action_repr, seed, **ModelSpec overrides) -> directory of a small random bundle.

    A SmolVLA with two cameras, 12 prefix tokens and 3 denoising steps: every code path of the
    real model, in milliseconds on a CPU."""

    def make(action_repr="absolute", seed=0, **spec_kw):
        import torch
        from safetensors.torch import save_file

        from so101_smolvla_edge.model.smolvla import ModelSpec, param_shapes

        small = dict(
            n_cams=2,
            image_size=128,
            patch=16,
            vis_dim=32,
            vis_layers=2,
            vis_heads=4,
            vis_mlp=64,
            shuffle=4,
            text_dim=48,
            n_layers=4,
            n_heads=6,
            n_kv=2,
            head_dim=8,
            text_mlp=96,
            exp_dim=40,
            exp_mlp=64,
            chunk=8,
            max_state=8,
            max_action=8,
            n_steps=3,
            lang_len=3,
        )
        spec = ModelSpec(**(small | spec_kw))
        g = torch.Generator().manual_seed(seed)

        def rnd(*shape):
            return torch.randn(*shape, generator=g)

        norm_leaves = ("ln_in", "ln_post", "norm")
        norm_suffixes = ("ln1.w", "ln2.w", "post_ln.w")
        w = {}
        for name, (shape, _) in param_shapes(spec, torch.float32).items():
            if name.split(".")[-1] in norm_leaves or name.endswith(norm_suffixes):
                w[name] = 1 + 0.1 * rnd(*shape)
            elif len(shape) == 2:
                w[name] = rnd(*shape) / shape[1] ** 0.5
            else:
                w[name] = 0.1 * rnd(*shape)
        n = 6  # state and action dim
        w["lang.emb"] = 2 * rnd(spec.lang_len, spec.text_dim)
        w["repr.mean"], w["repr.std"] = (
            0.1 * rnd(spec.max_action),
            0.5 + torch.rand(spec.max_action, generator=g),
        )
        for k in ("state", "action"):
            w[f"norm.{k}.mean"], w[f"norm.{k}.std"] = 0.2 * rnd(n), 0.5 + torch.rand(n, generator=g)
        path = tmp_path / f"bundle_{action_repr}_{seed}_{len(spec_kw)}"
        path.mkdir()
        save_file(w, str(path / "model.safetensors"))
        cams = ["observation.images.a", "observation.images.b"]
        (path / "meta.json").write_text(
            json.dumps(
                {
                    "format": 1,
                    "action_repr": action_repr,
                    "task": "t",
                    "cameras": cams,
                    "image_hw": [120, 160],
                    "state_dim": n,
                    "action_dim": n,
                    "joints": [f"j{i}" for i in range(n)],
                    "spec": {k: getattr(spec, k) for k in ModelSpec.__dataclass_fields__},
                }
            )
        )
        return path

    return make


@pytest.fixture
def observation():
    """make(bundle, seed) -> (images, state, noise), random and shaped for that bundle."""

    def make(bundle, seed=0):
        import numpy as np

        meta = json.loads((bundle / "meta.json").read_text())
        spec = meta["spec"]
        rng = np.random.default_rng(seed)
        h, w = meta["image_hw"]
        images = {c: rng.integers(0, 256, (h, w, 3), dtype=np.uint8) for c in meta["cameras"]}
        state = rng.uniform(-1, 1, meta["state_dim"]).astype(np.float32)
        noise = rng.standard_normal((spec["chunk"], spec["max_action"])).astype(np.float32)
        return images, state, noise

    return make
