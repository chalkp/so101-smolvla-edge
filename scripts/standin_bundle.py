"""Build a stand-in bundle from the public `lerobot/smolvla_base` checkpoint.

It is for machines without a trained checkpoint. It has the architecture and pretrained weights
the fine-tunes start from, so export, compilation, latency and reduced-precision drift on it are
representative.

    url=https://huggingface.co/lerobot/smolvla_base/resolve/main
    for f in config.json model.safetensors; do curl -L --create-dirs -o ckpt/$f $url/$f; done
    python scripts/standin_bundle.py ckpt bundles/standin

It is not a policy for the arms: the normalisation statistics, cameras and joints are made up, and
its golden.npz holds what this package's own float32 engine computes on synthetic images, not what
the training code computes. `so101-smolvla-edge verify` on it compares a backend to that engine.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from safetensors.torch import load_file, save_file

from so101_smolvla_edge.model.bundle import FORMAT, JOINTS, _map_weights
from so101_smolvla_edge.model.smolvla import ModelSpec

TASK = "fold the garment on the table"
# TASK + "\n" under the HuggingFaceTB/SmolVLM2-500M-Video-Instruct tokenizer. With 3 x 64 image
# tokens and the state token that makes a 200-token prefix, like the trained bundles'.
TASK_IDS = [7177, 260, 25196, 335, 260, 3252, 198]
CAMERAS = [
    "observation.images.top_rgb",
    "observation.images.left_rgb",
    "observation.images.right_rgb",
]
GOLDEN_SOURCE = (
    "so101_smolvla_edge float32 engine on synthetic images (stand-in; not the training code)"
)


def scene(rng: np.random.Generator, h: int = 480, w: int = 640) -> np.ndarray:
    """Synthetic table-top RGB frame (h, w, 3) uint8: shaded background, blurred shapes, noise.

    SigLIP's activations on white noise are unlike those on a camera image, hence the shapes.
    """
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = rng.uniform(60, 190, 3)
    img = (
        base[None, None]
        + rng.uniform(-40, 40, 3)[None, None] * (yy / h)[..., None]
        + rng.uniform(-25, 25, 3)[None, None] * (xx / w)[..., None]
    )
    img = np.ascontiguousarray(np.clip(img, 0, 255).astype(np.uint8))
    for _ in range(int(rng.integers(5, 10))):
        colour = tuple(int(c) for c in rng.integers(0, 255, 3))
        c = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        if rng.random() < 0.5:
            cv2.ellipse(
                img,
                c,
                (int(rng.integers(20, 140)), int(rng.integers(20, 140))),
                float(rng.uniform(0, 180)),
                0,
                360,
                colour,
                -1,
            )
        else:
            pts = (np.array(c) + rng.integers(-150, 150, (int(rng.integers(3, 7)), 2))).astype(
                np.int32
            )
            cv2.fillPoly(img, [pts], colour)
    img = cv2.GaussianBlur(img, (0, 0), float(rng.uniform(0.8, 2.5)))
    return np.clip(img.astype(np.float32) + rng.normal(0, 4, img.shape), 0, 255).astype(np.uint8)


def main() -> None:
    """Write the stand-in bundle, then its golden.npz from the float32 engine on the CPU."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "checkpoint", help="directory with lerobot/smolvla_base's config.json and model.safetensors"
    )
    ap.add_argument("bundle", help="output directory")
    ap.add_argument("-n", type=int, default=2, help="number of golden observations")
    args = ap.parse_args()
    ckpt, out = Path(args.checkpoint), Path(args.bundle)
    cfg = json.loads((ckpt / "config.json").read_text())
    if cfg.get("attention_mode") != "cross_attn" or cfg.get("self_attn_every_n_layers") != 2:
        raise SystemExit(
            "only attention_mode=cross_attn with self_attn_every_n_layers=2 is implemented"
        )
    spec = ModelSpec(
        n_cams=len(CAMERAS),
        chunk=cfg["chunk_size"],
        n_steps=cfg["num_steps"],
        max_state=cfg["max_state_dim"],
        max_action=cfg["max_action_dim"],
        n_layers=cfg["num_vlm_layers"],
        min_period=cfg["min_period"],
        max_period=cfg["max_period"],
        image_size=cfg["resize_imgs_with_padding"][0],
        lang_len=len(TASK_IDS),
    )
    sd = load_file(str(ckpt / "model.safetensors"))
    weights = _map_weights(sd, spec)
    emb = sd["model.vlm_with_expert.vlm.model.text_model.embed_tokens.weight"]
    weights["lang.emb"] = (
        emb[torch.tensor(TASK_IDS)].float() * math.sqrt(spec.text_dim)
    ).contiguous()
    n = len(JOINTS)
    weights["repr.mean"], weights["repr.std"] = (
        torch.zeros(spec.max_action),
        torch.ones(spec.max_action),
    )
    for name in ("state", "action"):  # made up: a joint that moves +-0.6 rad about zero
        weights[f"norm.{name}.mean"] = torch.zeros(n)
        weights[f"norm.{name}.std"] = torch.full((n,), 0.6)
    out.mkdir(parents=True, exist_ok=True)
    save_file(weights, str(out / "model.safetensors"))
    meta = {
        "format": FORMAT,
        "source": str(ckpt.resolve()),
        "action_repr": "absolute",
        "task": TASK,
        "task_token_ids": TASK_IDS,
        "cameras": CAMERAS,
        "image_hw": [480, 640],
        "state_dim": n,
        "action_dim": n,
        "joints": JOINTS,
        "spec": {k: getattr(spec, k) for k in ModelSpec.__dataclass_fields__},
        "fps_label": 30,
        "standin": True,
        "golden_source": GOLDEN_SOURCE,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1))

    from so101_smolvla_edge.model.engine import Engine

    rng = np.random.default_rng(0)
    g = torch.Generator().manual_seed(1234)
    rec = {f"img.{c}": [] for c in CAMERAS} | {"state": [], "noise": [], "actions_fp32": []}
    eng = Engine(out, device="cpu", dtype="fp32", cuda_graph=False, warmup=0)
    for _ in range(args.n):
        imgs = {c: scene(rng) for c in CAMERAS}
        state = rng.uniform(-0.8, 0.8, n).astype(np.float32)
        noise = torch.randn((spec.chunk, spec.max_action), generator=g).numpy()
        t0 = time.perf_counter()
        rec["actions_fp32"].append(eng.infer(imgs, state, noise=noise))
        print(
            f"reference chunk in {time.perf_counter() - t0:.2f} s "
            f"(torch {torch.__version__}, CPU, float32)"
        )
        for c in CAMERAS:
            rec[f"img.{c}"].append(imgs[c])
        rec["state"].append(state)
        rec["noise"].append(noise)
    np.savez_compressed(out / "golden.npz", **{k: np.stack(v) for k, v in rec.items()})
    a = np.stack(rec["actions_fp32"])
    print(
        f"bundle written to {out}: actions in [{a.min():+.3f}, {a.max():+.3f}] rad, "
        f"std across the chunk {a.std(1).mean():.3f}"
    )


if __name__ == "__main__":
    main()
