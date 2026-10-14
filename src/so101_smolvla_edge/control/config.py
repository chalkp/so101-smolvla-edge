"""Run configuration loaded from YAML, with every default in one place.

configs/bimanual.example.yaml is an annotated example.
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ..robot.arm import ServoSettings
from ..robot.safety import SafetyConfig

SIM_STEP_RAD = 10.0 / 90.0  # rad; the simulator's largest step per action (10 rad/s for 1/90 s)


@dataclass
class CameraConfig:
    device: str  # /dev/v4l/by-path/... (stable across reboots) or an index
    width: int = 640
    height: int = 480
    fps: int = 30
    fourcc: str = "MJPG"
    flip: bool = False  # rotate 180 degrees, for a camera mounted upside down
    crop: float = 1.0  # fraction of the frame width kept (centre crop), to match the simulator's
    # field of view; see cameras.crop_for_fov and docs/real_robot.md
    capture_width: int | None = None  # if not width, e.g. 1280 so a cropped frame keeps its detail
    capture_height: int | None = None  # if not height, e.g. 960
    gain: float = 1.0  # tone curve out = 255 * gain * (in / 255) ** gamma (cameras.tone_lut);
    # 0.88 puts an overexposing camera's clipped white at the simulator's table white
    gamma: float = 1.0  # > 1 darkens the mid-tones that overexposure washed out
    controls: dict[str, int] = field(default_factory=dict)  # V4L2 controls set on open, in order,
    # by driver name, e.g. {"Focus, Automatic Continuous": 0, "Focus, Absolute": 208}.
    # `so101-smolvla-edge cameras CONFIG --controls` lists what each camera has.


@dataclass
class ArmConfig:
    port: str
    calibration: str
    echo: str = "auto"  # auto | on | off: whether the bus adapter echoes its own writes
    convention: str = "degrees"  # degrees: true angles about the calibrated midpoint (lerobot
    # use_degrees) | stretched: range mapped onto the simulator's limits (robot/calibration.py)


@dataclass
class InferenceConfig:
    backend: str = "torch"  # torch: the CUDA engine, set up by the options down to trt_precision
    # | openvino: Intel NPU / CPU, after so101-smolvla-edge build-ov BUNDLE
    ov_device: str = "NPU"  # openvino: one device for every stage, or "vision,prefix,denoise"
    ov_turbo: bool = True  # openvino: NPU_TURBO; an NPU left idle between chunks is slow without it
    dtype: str = "bf16"  # bf16 | fp16 | fp32
    residual: str = "fp32"  # fp32 (as the reference) | compute
    cuda_graph: bool = True
    compile: bool | str = False  # true | "autotune"; needs Triton, falls back to eager kernels
    vision: str = "torch"  # torch | trt (build the plan first: so101-smolvla-edge build-trt BUNDLE)
    trt_precision: str = "bf16"  # bf16, the dtype the policy was trained in; fp16 is no faster
    kernel_cache: str | None = None  # directory for torch.compile / autotune kernels; a cold cache
    # compiles for minutes at start. Unset = $TORCHINDUCTOR_CACHE_DIR, else <bundle>/inductor;
    # set it here so a start does not depend on what the shell exported.
    replan_period_s: float = 0.5  # s; start inferences at most this often without replan_actions
    replan_actions: int | None = None  # actions of each chunk run before replanning, e.g. 45 of 50:
    # close to the simulator evaluation (all 50) and leaves time to compute the next chunk.
    # Replanning every few actions stalls the arms, since chunks often hold still before they move.
    replan_margin_s: float = 0.35  # s; replan at least this long before the chunk ends, so the
    # arm never runs out of actions (inference latency + crossfade)
    action_dt: float = 1.0 / 30.0  # s between actions: the simulator's 1/90 s motion at a third
    # of the speed, which real servos can follow (docs/real_robot.md)
    max_frame_age_s: float = 0.15  # skip an inference if any camera frame is older than this
    max_state_age_s: float = 0.05  # ... or if the joint state is older than this


@dataclass
class ControlConfig:
    rate_hz: float = 100.0  # per arm; each arm has its own bus and thread
    interp: str = "cubic"  # cubic | linear
    crossfade_s: float = 0.10  # s to blend from the previous chunk into a new one
    max_start_jump: float = 2.0  # rad; plausibility check only (robot/executor.py)
    max_first_jump: float = 2.0  # rad; loose because the dataset's first action is well over a
    # radian from the rest pose (robot/executor.py)
    min_remaining_s: float = 0.05  # reject a chunk with less than this left to run on arrival
    max_latency_s: float = 2.0  # reject a chunk whose observation is older than this
    rt_priority: int = 0  # SCHED_FIFO priority for the arm threads if permitted; 0 = off
    home_pose: dict[str, float] | None = None  # rad by joint name, for the `home` command
    home_speed: float = 0.5  # rad/s
    enable_on_start: bool = False
    verify_calibration: bool = True  # check the servos' homing offsets and limits on connect
    configure_servos: bool = True  # write the `servo` settings to the followers on connect


@dataclass
class TeleopConfig:
    """Leader arms driving the followers: demonstrations, and takeovers during a run (DAgger)."""

    rate_hz: float = 60.0  # leader read rate
    max_engage_rad: float = 1.0  # refuse a takeover if a leader joint is further from its follower
    engage_speed: float = 1.0  # rad/s while the followers close that initial gap
    align_rad: float = 0.15  # follow at full speed once every joint is within this of the leaders
    max_leader_age_s: float = 0.2  # hold if the leader reading is older than this


@dataclass
class RecordConfig:
    """Episode recording: camera frames, joint states, commanded goals and control mode at `fps`."""

    enabled: bool = False
    dir: str = field(default_factory=lambda: os.environ.get("SO101_DATA_DIR", "~/so101-data"))
    fps: float = 30.0
    jpeg_quality: int = 92
    queue: int = 120  # frames buffered for the writer; when full, new frames are dropped so the
    # control loop never waits


@dataclass
class VizConfig:
    """Live view in Rerun (control/viz.py), off by default; `run --rerun` also turns it on."""

    rerun: bool = False
    host: str = ""  # address viewers reach this machine by; empty = detect
    grpc_port: int = 9876  # rerun+http://<host>:<grpc_port>/proxy
    web_port: int = 9090  # the web viewer is served from here; 0 = none
    live_hz: float = 5.0  # camera frames and joint traces per second
    jpeg_quality: int = 75  # live frames
    input_jpeg_quality: int = 90  # the frames the model was given
    memory_limit: str = "256MiB"  # what the stream server keeps for viewers that connect late
    cors_allow_origin: list[str] = field(default_factory=list)  # web origins allowed to read the
    # stream besides http://*:<web_port>, e.g. a viewer hosted elsewhere


@dataclass
class RunConfig:
    bundle: str
    arms: dict[str, ArmConfig]
    leaders: dict[str, ArmConfig] = field(default_factory=dict)  # keyed like arms; never torqued
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    teleop: TeleopConfig = field(default_factory=TeleopConfig)
    record: RecordConfig = field(default_factory=RecordConfig)
    viz: VizConfig = field(default_factory=VizConfig)
    log_inputs: bool = True  # save the frames given to the model at every inference to
    # <run log>/inputs/, to see afterwards what the policy saw
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    servo: ServoSettings = field(default_factory=ServoSettings)
    baud: int = 1_000_000
    reply_margin_s: float = 0.0025  # s added to each bus reply deadline (robot/bus.py)
    log_dir: str = field(default_factory=lambda: os.environ.get("SO101_LOG_DIR", "~/so101-logs"))
    log_every: int = 1  # write every n-th control tick to ticks.csv


def _build(cls, d: dict | None):
    """Build dataclass `cls` from a YAML mapping, rejecting unknown keys."""
    d = dict(d or {})
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(d) - names
    if unknown:
        raise ValueError(f"{cls.__name__}: unknown key(s) {sorted(unknown)}")
    for f in dataclasses.fields(cls):
        if f.name in d and isinstance(d[f.name], list) and f.type in ("tuple[str, ...]",):
            d[f.name] = tuple(d[f.name])
    return cls(**d)


def _expand(p) -> str:
    return str(Path(p).expanduser())


def load_config(path: str | Path) -> RunConfig:
    """Load a run config from YAML, expanding ~ in paths and resolving `max_velocity: auto`.

    `auto` sets safety.max_velocity (and max_velocity_gripper) to the simulator's largest step per
    action at this config's action_dt. The policy learned on arms that chase each action at that
    rate, so a run slowed down with a larger action_dt needs a proportionally lower slew limit.
    """
    raw = yaml.safe_load(Path(path).read_text())
    arms = {k: _build(ArmConfig, v) for k, v in raw.pop("arms").items()}
    for a in arms.values():
        a.calibration = _expand(a.calibration)
    cams = {k: _build(CameraConfig, v) for k, v in (raw.pop("cameras", None) or {}).items()}
    leaders = {k: _build(ArmConfig, v) for k, v in (raw.pop("leaders", None) or {}).items()}
    for a in leaders.values():
        a.calibration = _expand(a.calibration)
    orphans = sorted(set(leaders) - set(arms))
    if orphans:
        raise ValueError(f"leaders {orphans} have no follower of the same name under `arms`")
    safety_raw = dict(raw.get("safety") or {})
    dt = float((raw.get("inference") or {}).get("action_dt", InferenceConfig.action_dt))
    for key in ("max_velocity", "max_velocity_gripper"):
        if str(safety_raw.get(key, "")).lower() == "auto":
            safety_raw[key] = SIM_STEP_RAD / dt
    if "safety" in raw:
        raw["safety"] = safety_raw
    cfg = RunConfig(
        bundle=_expand(raw.pop("bundle")),
        arms=arms,
        cameras=cams,
        leaders=leaders,
        teleop=_build(TeleopConfig, raw.pop("teleop", None)),
        record=_build(RecordConfig, raw.pop("record", None)),
        viz=_build(VizConfig, raw.pop("viz", None)),
        inference=_build(InferenceConfig, raw.pop("inference", None)),
        control=_build(ControlConfig, raw.pop("control", None)),
        safety=_build(SafetyConfig, raw.pop("safety", None)),
        servo=_build(ServoSettings, raw.pop("servo", None)),
        **raw,
    )
    if cfg.inference.kernel_cache:
        cfg.inference.kernel_cache = _expand(cfg.inference.kernel_cache)
    cfg.log_dir = _expand(cfg.log_dir)
    cfg.record.dir = _expand(cfg.record.dir)
    return cfg


def as_dict(cfg: RunConfig) -> dict:
    """The resolved config as plain data, for the run log's config.json."""
    return dataclasses.asdict(cfg)
