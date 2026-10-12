# so101-smolvla-edge

SmolVLA inference on edge accelerators and fail-safe control for a bimanual SO-101, in one package:
the CUDA engine that runs on a Jetson Orin (and desktop GPUs), one bundle format, one CLI
(`so101-smolvla-edge`) and the robot process.

- **CUDA engine**: the whole action chunk as one CUDA graph, the vision tower in TensorRT, and
  torch.compile with autotuned GEMMs for the prefix pass.
- **Robot process**: Feetech STS3215 bus driver, calibration, chunk executor, leader-arm teleop,
  episode recording and a safety layer that reads load, current, temperature and voltage every tick.

Nothing from `transformers` or `lerobot` runs on the robot: torch, numpy, safetensors, pyserial,
OpenCV and PyYAML.

## Results

**Jetson Orin Nano Super 8 GB** (JetPack 7.2.1, MAXN SUPER, GPU clock pinned), trained bundle.
Images in, 50 x 12 actions out, p50 over 200 chunks; error is the largest action difference from
the training code's float32 output on the bundle's two reference observations (`verify`).

| engine | per chunk | prefix | denoise | max error |
|---|---|---|---|---|
| CUDA graph | 172.0 ms | 20.7 ms | 54.5 ms | 5.6e-3 rad |
| + TensorRT vision, int8 | 135.0 ms | 20.7 ms | 54.5 ms | |
| **+ `--autotune`** | **110.2 ms** | **12.0 ms** | **39.1 ms** | 7.8e-3 rad |

The int8 vision was evaluated on LeHome against bf16 and came out level. The first autotuned start
compiles for minutes; keep `TORCHINDUCTOR_CACHE_DIR` (or `inference.kernel_cache`) on disk.

## Install

```bash
scripts/setup_jetson.sh                       # Jetson Orin: uv venv, jp7 (JetPack 7) or jp6
uv sync --extra desktop --extra dev           # x86 with an NVIDIA GPU (driver >= 580)
sudo nvpmodel -m 2 && sudo jetson_clocks      # 2 = MAXN_SUPER on an Orin Nano (check nvpmodel -q)
sudo usermod -aG dialout,video $USER          # serial and cameras, then log in again
```

## Use

```bash
# desktop: training checkpoint to bundle (needs the training repo's lehome_vla for the references)
so101-smolvla-edge export <ckpt>/pretrained_model bundles/nosig --golden obs.npz

# Jetson: copy the bundle over, then
so101-smolvla-edge build-trt bundles/nosig --precision bf16 int8   # once per device
so101-smolvla-edge verify bundles/nosig --vision trt --trt-precision int8
so101-smolvla-edge bench bundles/nosig --vision trt --trt-precision int8 --autotune --profile

# hardware (docs/real_robot.md walks through it)
so101-smolvla-edge scan --port /dev/serial/by-path/...
cp configs/bimanual.example.yaml robot.yaml                       # ports, calibration, cameras
so101-smolvla-edge monitor robot.yaml                             # read-only telemetry
so101-smolvla-edge run robot.yaml                                 # enable, go, take, pause, resume, home, off

# the whole robot process against simulated servos and the bundle's reference images
so101-smolvla-edge run configs/sim.yaml --sim --commands enable,go,wait5 --bundle bundles/nosig
```

## How the CUDA engine works

`model/smolvla.py` is a static-shape re-implementation of SmolVLA (SigLIP, a 16-layer SmolLM2
prefix, the action expert) checked against the training code by the bundle's reference actions.
`model/engine.py` runs it:

- **One CUDA graph per chunk.** Preprocessing, vision, the prefix pass, ten denoising steps and the
  unnormalisation are captured once and replayed with one launch. On an Orin the steps were bound
  by kernel launches from the ARM cores, not by the GPU.
- **Work hoisted out of the loop.** The expert's projections of the prefix keys and values, the
  time embedding and the masks are computed once per chunk; a step only writes its own 50 rows.
- **TensorRT vision** (`vision: trt`), bf16 or W8A8 int8, enqueued inside the same graph.
- **torch.compile** per stage; `autotune` also autotunes the prefix GEMMs. The denoising GEMMs get
  slower when autotuned, so they keep the default mode.

## Control

`robot/` (protocol, bus, calibration, arm, executor, safety, simulated servos) and `control/`
(config, cameras, runner, teleop, recorder, logs, viewer). Per arm, per tick at 100 Hz on its own
thread and bus: one SYNC_READ of position, velocity, load, voltage, temperature, status and
current; sample the chunk executor; safety checks; output filter; one SYNC_WRITE of goals.

The executor samples chunks by observation time, so inference latency is skipped, not replayed.
It interpolates through the actions, crossfades 0.1 s between chunks and slew-limits every joint.

| condition | default | response |
|---|---|---|
| goal far ahead of the measured position | 0.5 rad; 0.15 rad (gripper 0.25) while load >= 60 % (gripper 40 %) | goal clamped, which caps the force a position servo can build |
| output slew | max_velocity x 1.15 | applied last, so a hold, clamp release or resume never steps a servo |
| joint limits | LeHome limits | command clamped; `enable` refused outside the calibrated range |
| sustained overload | load >= 90 % for 0.3 s (gripper: none) | hold |
| tracking error | > 0.6 rad behind the command for 0.5 s | hold |
| servo error bits | overheat, overload, overcurrent | hold (others warn) |
| supply voltage | outside 6.0 to 14.0 V for 0.2 s | hold; set it for your servos |
| bus | 20 incomplete reads in a row, or a port error | hold; the port is reopened every second |
| temperature | 55 / 62 / 68 C | warn / hold / torque off for that arm |
| inference error | any exception | policy paused |
| stale input | camera frame > 0.15 s, joint state > 0.05 s | inference skipped, the last action held |
| implausible chunk | far from where the live chunk is heading, too old, non-finite | chunk rejected |

A hold is latched and covers both arms: the policy pauses and every joint holds where it was
measured when the fault fired, so force built up against an obstacle is released. `resume` clears
it. Torque is only dropped by the over-temperature fault and the `off` command, because torque-off
drops an SO-101 arm; quitting leaves the arms holding.

Each run writes `<log_dir>/<timestamp>/`: `config.json`, `ticks.csv` (per arm per tick: loop
period, bus round trip, and per joint the target, goal, position, velocity, load, current,
temperature, voltage), `events.jsonl` and `chunks.jsonl`.

## Tests

`uv run pytest`: the servo protocol, the bus over a pty against simulated servos, calibration, the
executor, every safety fault path and end-to-end runner runs. `SO101_BUNDLE=<bundle>` adds parity
against the bundle's reference actions (CUDA). CI runs ruff and the tests on every push.
