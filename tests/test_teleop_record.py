"""Leader takeover (DAgger) and episode recording, end to end on simulated servos."""

import json
import math
import threading
import time

import numpy as np
import pytest

from so101_smolvla_edge.cli import FakePolicy
from so101_smolvla_edge.control.cameras import ReplayCamera
from so101_smolvla_edge.control.config import (
    ArmConfig,
    ControlConfig,
    InferenceConfig,
    RecordConfig,
    RunConfig,
    TeleopConfig,
)
from so101_smolvla_edge.control.recorder import Recorder
from so101_smolvla_edge.control.runner import HOLDING, RUNNING, TELEOP, Runner
from so101_smolvla_edge.control.telemetry import RunLog
from so101_smolvla_edge.robot import protocol as P
from so101_smolvla_edge.robot.arm import Arm
from so101_smolvla_edge.robot.bus import Bus
from so101_smolvla_edge.robot.safety import SafetyConfig
from so101_smolvla_edge.robot.servo_sim import ServoSim

CAMS = ["observation.images.top_rgb", "observation.images.left_rgb"]


def wait_for(cond, timeout=4.0):
    """Poll `cond` every 10 ms for up to `timeout` s; return whether it came true."""
    t = time.monotonic() + timeout
    while time.monotonic() < t:
        if cond():
            return True
        time.sleep(0.01)
    return False


@pytest.fixture
def rig(tmp_path, calib_file):
    """Two followers, two torque-off leaders and two replay cameras under a recording Runner.

    Yields (runner, sims, rec), with sims keyed "f:left", "l:right" and so on."""
    sims, arms, leaders = {}, {}, {}
    cfg = RunConfig(
        bundle="",
        arms={},
        inference=InferenceConfig(replan_period_s=0.2),
        control=ControlConfig(rate_hz=100),
        safety=SafetyConfig(load_hard_s=0.2),
        teleop=TeleopConfig(rate_hz=100, max_engage_rad=1.0, engage_speed=1.0, align_rad=0.15),
        record=RecordConfig(enabled=True, dir=str(tmp_path / "data"), fps=30),
        log_dir=str(tmp_path / "logs"),
    )
    for kind, store in (("f", arms), ("l", leaders)):
        for name in ("left", "right"):
            cal = calib_file(f"{kind}_{name}")
            sim = ServoSim(list(range(1, 7)))
            for sid in range(1, 7):
                sim.set(sid, P.MIN_POSITION_LIMIT, 600)
                sim.set(sid, P.MAX_POSITION_LIMIT, 3500)
            sims[f"{kind}:{name}"] = sim
            store[name] = Arm(
                name,
                sim.path,
                str(cal),
                echo="off",
                reply_margin_s=0.01,
                bus=Bus(sim.path, echo=False, reply_margin_s=0.01),
            )
            store[name].connect(configure=kind == "f")
            if kind == "f":
                cfg.arms[name] = ArmConfig(port=sim.path, calibration=str(cal))
            else:
                store[name].set_torque(False)
    joints = [j for a in arms.values() for j in a.joints]
    frames = np.random.default_rng(0).integers(0, 255, (4, 48, 64, 3), dtype=np.uint8)
    cams = {k: ReplayCamera(k, frames) for k in CAMS}
    runlog = RunLog(cfg.log_dir, {n: a.joints for n, a in arms.items()}, {})
    rec = Recorder(cfg.record, joints, cams, {"task": "test"})
    policy = FakePolicy(len(joints), delay_s=0.05)
    runner = Runner(cfg, arms, cams, policy, runlog, joints, leaders=leaders, recorder=rec)
    runner.start()
    yield runner, sims, rec
    runner.stop()
    runlog.close()
    for s in sims.values():
        s.close()


def leader_ticks(runner, name, joint_idx, rad):
    """Ticks that put one leader joint at `rad` (leaders share the followers' calibration here)."""
    return runner.leaders[name].cal[joint_idx].to_ticks(rad)


def test_takeover_follows_the_leader_and_hands_back(rig):
    """The followers track the leaders after take, go hands back, and take works mid-run too."""
    runner, sims, rec = rig
    assert wait_for(lambda: np.isfinite(runner.q).all() and np.isfinite(runner.q_leader).all())
    assert runner.enable() and runner.state == HOLDING
    assert runner.take() and runner.state == TELEOP  # leaders start where the followers are
    target = 0.6
    sims["l:left"].set_position(1, leader_ticks(runner, "left", 0, target))  # left shoulder pan
    i = runner.joints.index("left_shoulder_pan")
    assert wait_for(lambda: abs(runner.q[i] - target) < 0.03)
    assert abs(runner.q[runner.joints.index("right_shoulder_pan")]) < 0.03
    runner.go()
    assert runner.state == RUNNING
    assert wait_for(lambda: runner.chunks_ok >= 1)
    assert runner.take() and runner.state == TELEOP
    runner.pause()
    assert runner.state == HOLDING


def test_takeover_refused_when_the_leader_is_far(rig, capsys):
    """take is refused, naming the joint, when a leader is over max_engage_rad from its follower."""
    runner, sims, rec = rig
    assert wait_for(lambda: np.isfinite(runner.q).all() and np.isfinite(runner.q_leader).all())
    runner.enable()
    sims["l:right"].set_position(2, leader_ticks(runner, "right", 1, 1.4))  # 1.4 rad, limit 1.0
    assert wait_for(
        lambda: abs(runner.q_leader[runner.joints.index("right_shoulder_lift")] - 1.4) < 0.05
    )
    assert not runner.take() and runner.state == HOLDING
    assert "right_shoulder_lift" in capsys.readouterr().out


def test_takeover_closes_a_gap_at_reduced_speed(rig):
    """A gap within max_engage_rad but not aligned is closed at engage_speed, not at full speed."""
    runner, sims, rec = rig
    assert wait_for(lambda: np.isfinite(runner.q).all() and np.isfinite(runner.q_leader).all())
    runner.enable()
    i = runner.joints.index("left_elbow_flex")
    sims["l:left"].set_position(3, leader_ticks(runner, "left", 2, 0.6))  # 0.6 rad gap
    assert wait_for(lambda: abs(runner.q_leader[i] - 0.6) < 0.05)
    t0 = time.monotonic()
    assert runner.take() and not runner._teleop_aligned
    assert wait_for(lambda: runner._teleop_aligned, timeout=3.0)
    took = time.monotonic() - t0
    assert took > 0.35  # (0.6 - 0.15) rad at 1 rad/s, not at 3.33
    assert wait_for(lambda: abs(runner.q[i] - 0.6) < 0.03)


def test_losing_the_leaders_holds(rig):
    """Losing a leader arm during a takeover holds the followers with torque on."""
    runner, sims, rec = rig
    assert wait_for(lambda: np.isfinite(runner.q).all() and np.isfinite(runner.q_leader).all())
    runner.enable()
    assert runner.take()
    sims["l:left"].unplug()
    assert wait_for(lambda: runner.state == HOLDING, timeout=3.0)
    assert all(sims["f:left"].get(i, P.TORQUE_ENABLE) == 1 for i in range(1, 7))


def test_episode_records_frames_states_actions_and_mode(rig):
    """An episode records state, action, leader and mode at the record fps, and every frame.

    `save` keeps it with its label and note; `discard` throws the next episode away."""
    runner, sims, rec = rig
    assert wait_for(lambda: np.isfinite(runner.q).all() and np.isfinite(runner.q_leader).all())
    runner.enable()
    assert not rec.active
    runner.go()
    assert rec.active
    assert wait_for(lambda: runner.chunks_ok >= 2)
    assert runner.take()
    sims["l:left"].set_position(1, leader_ticks(runner, "left", 0, 0.3))
    time.sleep(0.6)
    runner.command("save shirt 3")
    assert runner.state == HOLDING  # ending an episode stops the motion
    assert not rec.active and len(rec.saved) == 1
    d = rec.saved[0]
    meta = json.loads((d / "episode.json").read_text())
    steps = np.load(d / "steps.npz")
    n = meta["frames"]
    assert (
        meta["label"] == "success"
        and meta["note"] == "shirt 3"
        and n >= 10
        and meta["joints"] == runner.joints
    )
    assert (
        steps["state"].shape == (n, 12)
        and steps["action"].shape == (n, 12)
        and steps["leader"].shape == (n, 12)
    )
    # policy frames (0), then leader frames (1), with takeover ramp frames (4) between them if the
    # policy had moved the arm more than align_rad from the leaders by the time of `take`
    mode = steps["mode"].astype(int)
    assert {0, 1} <= set(np.unique(mode)) <= {0, 1, 4}
    assert (np.diff(np.select([mode == 0, mode == 4, mode == 1], [0, 1, 2])) >= 0).all()
    assert np.isfinite(steps["state"]).all() and np.isfinite(steps["action"]).all()
    dt = np.diff(steps["t"])
    assert abs(np.median(dt) - 1 / 30) < 0.01
    for key in CAMS:
        assert len(list((d / key.split(".")[-1]).glob("*.jpg"))) == n
    # in teleop frames the action follows the leader
    i = runner.joints.index("left_shoulder_pan")
    assert steps["action"][steps["mode"] == 1][-1, i] == pytest.approx(0.3, abs=0.05)

    assert wait_for(lambda: abs(runner.q_leader[i] - runner.q[i]) < 0.1)
    assert runner.take()  # starts a second episode
    assert rec.active
    time.sleep(0.2)
    runner.command("discard")
    assert not rec.active and len(rec.saved) == 1 and not (rec.session / "episode_0001").exists()


def test_takeover_ramp_is_not_recorded_as_operator_action(rig):
    """Takeover ramp frames are recorded as engaging (4), not as the operator's action.

    During the ramp the goal is not the leader's pose, so those frames must never become labels."""
    runner, sims, rec = rig
    assert wait_for(lambda: np.isfinite(runner.q).all() and np.isfinite(runner.q_leader).all())
    runner.enable()
    i = runner.joints.index("left_elbow_flex")
    sims["l:left"].set_position(3, leader_ticks(runner, "left", 2, 0.7))  # a ramp of ~0.5 s
    assert wait_for(lambda: abs(runner.q_leader[i] - 0.7) < 0.05)
    assert runner.take()
    assert wait_for(lambda: runner._teleop_aligned, timeout=3.0)
    time.sleep(0.4)
    runner.command("save")
    steps = np.load(rec.saved[0] / "steps.npz")
    mode, gap = steps["mode"], np.abs(steps["leader"] - steps["action"]).max(axis=1)
    assert (mode == 4).sum() >= 5 and (mode == 1).sum() >= 5  # the ramp, then the operator
    assert gap[mode == 4].max() > 0.3  # ramp frames: the goal is not the leader
    assert gap[mode == 1].max() < runner.cfg.teleop.align_rad + 0.02  # operator frames: it is
    meta = json.loads((rec.saved[0] / "episode.json").read_text())
    assert meta["frames_by_mode"]["engaging"] == int((mode == 4).sum())


def test_delayed_take_gives_time_to_reach_the_leaders(rig, capsys):
    """`take 1.5` counts down, printing the gap, so an operator alone can bring the leaders over.

    `take soon` is not a number: it prints usage and changes nothing."""
    runner, sims, rec = rig
    assert wait_for(lambda: np.isfinite(runner.q).all() and np.isfinite(runner.q_leader).all())
    runner.enable()
    sims["l:right"].set_position(2, leader_ticks(runner, "right", 1, 1.4))  # too far for now
    assert wait_for(
        lambda: abs(runner.q_leader[runner.joints.index("right_shoulder_lift")] - 1.4) < 0.05
    )

    def bring_back():
        sims["l:right"].set_position(2, leader_ticks(runner, "right", 1, 0.0))

    threading.Timer(0.5, bring_back).start()  # during the countdown
    t0 = time.monotonic()
    runner.command("take 1.5")
    assert time.monotonic() - t0 >= 1.5 and runner.state == TELEOP
    out = capsys.readouterr().out
    assert (
        "taking over in" in out
        and "right_shoulder_lift" in out
        and "the leaders have the arms" in out
    )
    runner.pause()
    runner.command("take soon")
    assert runner.state == HOLDING and "usage" in capsys.readouterr().out


def test_executor_follow_respects_the_slew_limit():
    """ChunkExecutor.follow moves under the slew limit, or under max_speed while engaging."""
    from so101_smolvla_edge.robot.executor import ChunkExecutor, ExecutorConfig

    cfg = ExecutorConfig(
        lower=np.full(2, -3.0), upper=np.full(2, 3.0), max_velocity=np.full(2, 2.0)
    )
    ex = ChunkExecutor(cfg)
    assert not ex.follow(np.zeros(2))  # no hold target yet
    ex.reset_hold(np.zeros(2), 0.0)
    assert ex.follow(np.array([1.0, 5.0]), max_speed=0.5)
    v = ex.sample(0.01)
    assert v == pytest.approx([0.005, 0.005])  # 0.5 rad/s for 10 ms
    ex.follow(np.array([1.0, 5.0]))
    assert ex.sample(0.02) == pytest.approx([0.025, 0.025])  # then 2 rad/s for 10 ms
    ex.hold()
    assert ex.sample(1.0) == pytest.approx([0.025, 0.025])  # holding the last command
    assert not ex.follow(np.array([math.nan, 0.0]))
