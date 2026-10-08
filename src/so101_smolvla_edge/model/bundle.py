"""Inference bundles: a SmolVLA training checkpoint reduced to what the robot needs.

    so101-smolvla-edge export <pretrained_model dir> <bundle dir> [--golden obs.npz]

A bundle is a directory holding
    model.safetensors  fused weights (q/k/v, gate/up) in their original precision, the embedded
                       task tokens (`lang.emb`) and the normalisation statistics. There is no
                       lm_head or token-embedding table, since the task string is fixed.
    meta.json          model spec, task and its token ids, cameras, joint names, action repr
    golden.npz         optional: observations, their noise and the actions the reference
                       implementation produced for them, so `so101-smolvla-edge verify` on the
                       robot computer can check the engine without the training code

Exporting needs `transformers` for the tokenizer and, for golden.npz, `lehome_vla.policy`.
Loading needs only torch and safetensors.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from .smolvla import ModelSpec, SmolVLA, param_shapes

FORMAT = 1  # meta.json "format" this code reads and writes
# default joint names for a 12-dim action: two arms, left first
JOINTS = [
    "left_shoulder_pan",
    "left_shoulder_lift",
    "left_elbow_flex",
    "left_wrist_flex",
    "left_wrist_roll",
    "left_gripper",
    "right_shoulder_pan",
    "right_shoulder_lift",
    "right_elbow_flex",
    "right_wrist_flex",
    "right_wrist_roll",
    "right_gripper",
]
SUPPORTED_REPR = ("absolute", "delta", "velocity")  # action representations export accepts


def _vision_position_ids(spec: ModelSpec) -> torch.Tensor:
    """Position ids SmolVLM's vision embeddings actually use for a full, unpadded image.

    They are not arange(1024): the reference buckets fractional coordinates `i / 32 * (1 - 1e-6)`
    against boundaries `arange(1/32, 1, 1/32)`, and in float32 the second coordinate lands just
    under the first boundary, so each axis reads [0, 0, 1, ..., 30]. The model was trained that way.
    The bundle stores the gathered table, so the engine adds it without indexing.
    """
    n = spec.image_size // spec.patch
    boundaries = torch.arange(1 / n, 1.0, 1 / n)
    frac = torch.arange(n, dtype=torch.float32) / n * (1 - 1e-6)
    b = torch.bucketize(frac, boundaries, right=True)
    return (b[:, None] * n + b[None, :]).flatten()


def _map_weights(sd: dict[str, torch.Tensor], spec: ModelSpec) -> dict[str, torch.Tensor]:
    """Rename and fuse a SmolVLA state dict into the bundle's parameter names.

    Raises KeyError if a parameter is missing, ValueError if a shape differs from `param_shapes`.
    """
    V = "model.vlm_with_expert.vlm.model.vision_model."
    T = "model.vlm_with_expert.vlm.model.text_model.layers."
    E = "model.vlm_with_expert.lm_expert.layers."
    out = {
        "vis.patch.w": sd[V + "embeddings.patch_embedding.weight"].flatten(1),  # (768, 3*16*16)
        "vis.patch.b": sd[V + "embeddings.patch_embedding.bias"],
        "vis.pos": sd[V + "embeddings.position_embedding.weight"][_vision_position_ids(spec)],
        "vis.post_ln.w": sd[V + "post_layernorm.weight"],
        "vis.post_ln.b": sd[V + "post_layernorm.bias"],
        "conn.w": sd["model.vlm_with_expert.vlm.model.connector.modality_projection.proj.weight"],
        "state.w": sd["model.state_proj.weight"],
        "state.b": sd["model.state_proj.bias"],
        "act.in.w": sd["model.action_in_proj.weight"],
        "act.in.b": sd["model.action_in_proj.bias"],
        "act.time_in.w_action": sd["model.action_time_mlp_in.weight"][:, : spec.exp_dim],
        "act.time_in.w_time": sd["model.action_time_mlp_in.weight"][:, spec.exp_dim :],
        "act.time_in.b": sd["model.action_time_mlp_in.bias"],
        "act.time_out.w": sd["model.action_time_mlp_out.weight"],
        "act.time_out.b": sd["model.action_time_mlp_out.bias"],
        "act.out.w": sd["model.action_out_proj.weight"],
        "act.out.b": sd["model.action_out_proj.bias"],
        "exp.norm": sd["model.vlm_with_expert.lm_expert.norm.weight"],
    }
    for i in range(spec.vis_layers):
        p = f"{V}encoder.layers.{i}."
        a = p + "self_attn."
        out.update(
            {
                f"vis.{i}.ln1.w": sd[p + "layer_norm1.weight"],
                f"vis.{i}.ln1.b": sd[p + "layer_norm1.bias"],
                f"vis.{i}.ln2.w": sd[p + "layer_norm2.weight"],
                f"vis.{i}.ln2.b": sd[p + "layer_norm2.bias"],
                f"vis.{i}.qkv.w": torch.cat([sd[a + f"{n}_proj.weight"] for n in "qkv"]),
                f"vis.{i}.qkv.b": torch.cat([sd[a + f"{n}_proj.bias"] for n in "qkv"]),
                f"vis.{i}.out.w": sd[a + "out_proj.weight"],
                f"vis.{i}.out.b": sd[a + "out_proj.bias"],
                f"vis.{i}.fc1.w": sd[p + "mlp.fc1.weight"],
                f"vis.{i}.fc1.b": sd[p + "mlp.fc1.bias"],
                f"vis.{i}.fc2.w": sd[p + "mlp.fc2.weight"],
                f"vis.{i}.fc2.b": sd[p + "mlp.fc2.bias"],
            }
        )
    for i in range(spec.n_layers):
        t, e = f"{T}{i}.", f"{E}{i}."
        out.update(
            {
                f"txt.{i}.ln_in": sd[t + "input_layernorm.weight"],
                f"txt.{i}.ln_post": sd[t + "post_attention_layernorm.weight"],
                f"txt.{i}.qkv": torch.cat([sd[t + f"self_attn.{n}_proj.weight"] for n in "qkv"]),
                f"txt.{i}.o": sd[t + "self_attn.o_proj.weight"],
                f"txt.{i}.gate_up": torch.cat(
                    [sd[t + "mlp.gate_proj.weight"], sd[t + "mlp.up_proj.weight"]]
                ),
                f"txt.{i}.down": sd[t + "mlp.down_proj.weight"],
                f"exp.{i}.ln_in": sd[e + "input_layernorm.weight"],
                f"exp.{i}.ln_post": sd[e + "post_attention_layernorm.weight"],
                f"exp.{i}.o": sd[e + "self_attn.o_proj.weight"],
                f"exp.{i}.gate_up": torch.cat(
                    [sd[e + "mlp.gate_proj.weight"], sd[e + "mlp.up_proj.weight"]]
                ),
                f"exp.{i}.down": sd[e + "mlp.down_proj.weight"],
            }
        )
        if spec.is_self_attn(i):
            out[f"exp.{i}.qkv"] = torch.cat([sd[e + f"self_attn.{n}_proj.weight"] for n in "qkv"])
        else:
            out[f"exp.{i}.q"] = sd[e + "self_attn.q_proj.weight"]
            out[f"exp.{i}.k"] = sd[e + "self_attn.k_proj.weight"]
            out[f"exp.{i}.v"] = sd[e + "self_attn.v_proj.weight"]
    want = param_shapes(spec)
    missing = sorted(set(want) - set(out))
    if missing:
        raise KeyError(f"bundle mapping is missing {missing[:5]}")
    for k, (shape, _) in want.items():
        if tuple(out[k].shape) != tuple(shape):
            raise ValueError(f"{k}: checkpoint shape {tuple(out[k].shape)}, expected {shape}")
    return {k: v.contiguous() for k, v in out.items()}


def _tokenize(model_id: str, task: str, max_len: int) -> list[int]:
    """Token ids of `task`, tokenised the way the reference does it.

    Newline-terminated, truncated at `max_len`, padded to the longest in a batch of one (so not
    padded at all).
    """
    from transformers import AutoProcessor

    tok = AutoProcessor.from_pretrained(model_id).tokenizer
    text = task if task.endswith("\n") else task + "\n"
    ids = tok(
        [text],
        max_length=max_len,
        truncation=True,
        padding="longest",
        padding_side="right",
        return_tensors="pt",
    )["input_ids"][0].tolist()
    return ids


def export_bundle(
    ckpt: str | Path,
    out: str | Path,
    task: str = "fold the garment on the table",
    golden_obs: str | Path | None = None,
    joints: list[str] | None = None,
) -> Path:
    """Export the checkpoint directory `ckpt` as a bundle in `out` and return `out`.

    `golden_obs` is an .npz of observations to write golden.npz from. Exits with a message if the
    checkpoint uses a configuration the engine does not implement.
    """
    ckpt, out = Path(ckpt), Path(out)
    cfg = json.loads((ckpt / "config.json").read_text())
    repr_ = cfg.get("action_repr", "absolute")
    if repr_ not in SUPPORTED_REPR:
        raise SystemExit(
            f"action_repr={repr_!r} is not supported for export (supported: {SUPPORTED_REPR})"
        )
    if cfg.get("attention_mode") != "cross_attn" or cfg.get("self_attn_every_n_layers") != 2:
        raise SystemExit(
            "only attention_mode=cross_attn with self_attn_every_n_layers=2 is implemented"
        )
    if cfg.get("add_image_special_tokens") or cfg.get("prefix_length", -1) > 0:
        raise SystemExit("image special tokens / fixed prefix padding are not implemented")
    cams = [k for k, v in cfg["input_features"].items() if v["type"] == "VISUAL"]
    state_dim = cfg["input_features"]["observation.state"]["shape"][0]
    action_dim = cfg["output_features"]["action"]["shape"][0]
    spec = ModelSpec(
        n_cams=len(cams),
        chunk=cfg["chunk_size"],
        n_steps=cfg["num_steps"],
        max_state=cfg["max_state_dim"],
        max_action=cfg["max_action_dim"],
        n_layers=cfg["num_vlm_layers"],
        min_period=cfg["min_period"],
        max_period=cfg["max_period"],
        image_size=cfg["resize_imgs_with_padding"][0],
    )

    sd = load_file(str(ckpt / "model.safetensors"))
    weights = _map_weights(sd, spec)
    ids = _tokenize(cfg["vlm_model_name"], task, cfg["tokenizer_max_length"])
    emb = sd["model.vlm_with_expert.vlm.model.text_model.embed_tokens.weight"]
    weights["lang.emb"] = (emb[torch.tensor(ids)].float() * math.sqrt(spec.text_dim)).contiguous()
    weights["repr.mean"] = sd.get("model.repr_mean", torch.zeros(spec.max_action)).float()
    weights["repr.std"] = sd.get("model.repr_std", torch.ones(spec.max_action)).float()

    pre = sorted(ckpt.glob("policy_preprocessor_step_*_normalizer_processor.safetensors"))
    post = sorted(ckpt.glob("policy_postprocessor_step_*_unnormalizer_processor.safetensors"))
    if not pre or not post:
        raise SystemExit(f"{ckpt}: normaliser statistics not found")
    ps, qs = load_file(str(pre[0])), load_file(str(post[0]))
    for name, src, key in (("state", ps, "observation.state"), ("action", qs, "action")):
        weights[f"norm.{name}.mean"] = src[f"{key}.mean"].float()
        weights[f"norm.{name}.std"] = src[f"{key}.std"].float()

    out.mkdir(parents=True, exist_ok=True)
    save_file(weights, str(out / "model.safetensors"))
    h, w = cfg["input_features"][cams[0]]["shape"][1:]
    meta = {
        "format": FORMAT,
        "source": str(ckpt.resolve()),
        "action_repr": repr_,
        "task": task,
        "task_token_ids": ids,
        "cameras": cams,
        "image_hw": [h, w],
        "state_dim": state_dim,
        "action_dim": action_dim,
        "joints": joints or (JOINTS if action_dim == 12 else [f"j{i}" for i in range(action_dim)]),
        "spec": {k: getattr(spec, k) for k in ModelSpec.__dataclass_fields__}
        | {"lang_len": len(ids)},
        "fps_label": 30,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    if golden_obs:
        make_golden(ckpt, out, golden_obs, task)
    return out


def make_golden(ckpt: Path, out: Path, obs_path: str | Path, task: str, n: int = 2) -> None:
    """Write golden.npz: reference actions for `n` observations with fixed noise, in two passes.

        actions       lehome_vla.policy as the evaluation server ran it, text and expert weights
                      in bfloat16, so `verify` measures drift from the policy that was scored
        actions_fp32  the same policy upcast to float32, the strict target for the fp32 engine

    TF32 is off for both, so the file does not depend on which GPU wrote it. Without CUDA both
    passes run on the CPU, whose bfloat16 kernels round slightly differently but well inside
    `verify`'s tolerance, so an export does not need a working GPU.
    """
    from lehome_vla.policy.checkpoint import load_smolvla_checkpoint

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cpu":
        print("no CUDA device: writing the reference actions on the CPU", flush=True)

    obs = dict(np.load(obs_path))
    meta = json.loads((out / "meta.json").read_text())
    cams = meta["cameras"]
    tf32 = torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False
    ref = load_smolvla_checkpoint(str(ckpt), device=dev)
    g = torch.Generator().manual_seed(1234)
    rec = {f"img.{c}": [] for c in cams} | {
        "state": [],
        "noise": [],
        "actions": [],
        "actions_fp32": [],
    }
    batches = []
    for i in range(n):
        noise = torch.randn((1, meta["spec"]["chunk"], meta["spec"]["max_action"]), generator=g)
        batch = {
            "observation.state": torch.tensor(obs["state"][i], dtype=torch.float32)[None].to(dev)
        }
        for c in cams:
            batch[c] = torch.from_numpy(obs[c][i]).permute(2, 0, 1)[None].float().div(255).to(dev)
            rec[f"img.{c}"].append(obs[c][i])
        rec["state"].append(obs["state"][i])
        rec["noise"].append(noise[0].numpy())
        batches.append((ref.preprocess(batch, task), noise.to(dev)))
    for key in ("actions", "actions_fp32"):
        if key == "actions_fp32":
            ref.policy.float()
        for b, noise in batches:
            with torch.no_grad():
                a = ref.policy.predict_actions(b, noise=noise)
            rec[key].append(ref.postprocess_actions(a).float()[0].cpu().numpy())
    torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = tf32
    np.savez_compressed(out / "golden.npz", **{k: np.stack(v) for k, v in rec.items()})


def load_bundle(
    path: str | Path,
    device: str = "cuda",
    compute_dtype: torch.dtype = torch.bfloat16,
    residual: str = "fp32",
):
    """Load a bundle as (SmolVLA on `device`, meta dict, extras).

    extras holds the bundle's non-model tensors (`lang.*`, `norm.*`, `repr.*`) on `device`.
    """
    path = Path(path)
    meta = json.loads((path / "meta.json").read_text())
    if meta.get("format") != FORMAT:
        raise SystemExit(f"{path}: bundle format {meta.get('format')}, this code reads {FORMAT}")
    spec = ModelSpec(**meta["spec"])
    sd = load_file(str(path / "model.safetensors"), device="cpu")
    model = SmolVLA(spec, compute_dtype, residual)
    shapes = param_shapes(spec, compute_dtype)
    with torch.no_grad():
        for name, (_, dt) in shapes.items():
            model.p(name).copy_(sd[name].to(dt))
    model.to(device).eval()
    extras = {k: v.to(device) for k, v in sd.items() if k.split(".")[0] in ("lang", "norm", "repr")}
    return model, meta, extras
