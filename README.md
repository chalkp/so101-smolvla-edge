# so101-smolvla-edge

SmolVLA inference on edge accelerators and fail-safe control for a bimanual SO-101.

- **CUDA engine** (`model/`): a static-shape re-implementation of SmolVLA (SigLIP, a 16-layer
  SmolLM2 prefix, the action expert), checked against the training code by a bundle's reference
  actions, run as one CUDA graph per action chunk with the vision tower optionally in TensorRT.
- **Servo stack** (`robot/`): Feetech STS3215 protocol and bus driver, calibration, a chunk
  executor and a safety layer that reads load, current, temperature and voltage every tick.

The robot process and the command line come next.

## Install

```bash
uv sync --extra desktop --extra dev          # x86 with an NVIDIA GPU (driver >= 580)
uv sync --python 3.12 --extra jp7 --extra trt # Jetson Orin, JetPack 7
uv run pytest
```

## How the engine works

- **One CUDA graph per chunk.** Preprocessing, vision, the prefix pass, ten denoising steps and the
  unnormalisation are captured once and replayed with one launch. On an Orin the steps are bound by
  kernel launches from the ARM cores, not by the GPU.
- **Work hoisted out of the loop.** The expert's projections of the prefix keys and values, the
  time embedding and the masks are computed once per chunk; a step only writes its own 50 rows.
- **TensorRT vision** (`vision="trt"`), bf16 or W8A8 int8, enqueued inside the same graph.
- **torch.compile** per stage (`compile=True`); `"autotune"` also autotunes the prefix GEMMs.

## Tests

`uv run pytest`: the servo protocol, the bus over a pty against simulated servos, calibration,
the executor and every safety fault path. `SO101_BUNDLE=<bundle>` adds parity against the
bundle's reference actions (CUDA).
