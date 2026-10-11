"""Runner end to end on simulated servos: enable, run a fake policy, trip faults, resume."""

import time

import numpy as np
import pytest

from so101_smolvla_edge.cli import FakePolicy
from so101_smolvla_edge.control.config import ArmConfig, ControlConfig, InferenceConfig, RunConfig
from so101_smolvla_edge.control.runner import FAULT, HOLDING, RUNNING, Runner
from so101_smolvla_edge.control.telemetry import RunLog
from so101_smolvla_edge.robot import protocol as P
from so101_smolvla_edge.robot.arm import Arm
from so101_smolvla_edge.robot.bus import Bus
from so101_smolvla_edge.robot.safety import SafetyConfig
from so101_smolvla_edge.robot.servo_sim import ServoSim


@pytest.fixture
def rig(tmp_path, calib_file):
    """Two simulated arms under a started Runner with a fake policy; yields (runner, sims)."""
    sims, arms = {}, {}
    cfg = RunConfig(
        bundle="",
        arms={},
        inference=InferenceConfig(replan_period_s=0.2),
        control=ControlConfig(rate_hz=100),
        safety=SafetyConfig(load_hard_s=0.2),
        log_dir=str(tmp_path / "logs"),
    )
    for name in ("left", "right"):
        cal = calib_file(name)
        sim = ServoSim(list(range(1, 7)))
        for sid in range(1, 7):
            sim.set(sid, P.MIN_POSITION_LIMIT, 600)
            sim.set(sid, P.MAX_POSITION_LIMIT, 3500)
        sims[name] = sim
        cfg.arms[name] = ArmConfig(port=sim.path, calibration=str(cal))
        arms[name] = Arm(
            name,
            sim.path,
            str(cal),
            echo="off",
            reply_margin_s=0.01,
            bus=Bus(sim.path, echo=False, reply_margin_s=0.01),
        )
        arms[name].connect()
    joints = [j for a in arms.values() for j in a.joints]
    policy_joints = joints[::-1]  # not the arms' order, so the remapping is exercised
    runlog = RunLog(cfg.log_dir, {n: a.joints for n, a in arms.items()}, {})
    runner = Runner(cfg, arms, {}, FakePolicy(len(joints), delay_s=0.05), runlog, policy_joints)
    runner.start()
    yield runner, sims
    runner.stop()
    runlog.close()
    for s in sims.values():
        s.close()


def wait_for(cond, timeout=3.0):
    """Poll `cond` every 10 ms for up to `timeout` s; return whether it came true."""
    t = time.monotonic() + timeout
    while time.monotonic() < t:
        if cond():
            return True
        time.sleep(0.01)
    return False


def test_enable_run_fault_resume(rig):
    """Enable, run, fault on a sustained overload, resume.

    The fault holds position with torque on, since torque-off would drop the arm."""
    runner, sims = rig
    assert wait_for(lambda: np.isfinite(runner.q).all())
    for sim in sims.values():  # configure() wrote lerobot's settings
        assert sim.get(1, P.P_COEFFICIENT) == 16 and sim.get(1, P.RETURN_DELAY_TIME) == 0
    assert runner.enable() and runner.state == HOLDING
    assert all(sims["left"].get(i, P.TORQUE_ENABLE) == 1 for i in range(1, 7))
    start = sims["left"].get(1, P.GOAL_POSITION)
    runner.go()
    assert runner.state == RUNNING
    assert wait_for(lambda: runner.chunks_ok >= 3)
    assert wait_for(lambda: abs(sims["left"].get(1, P.GOAL_POSITION) - start) > 20)

    sims["left"].extra_load[2] = 950  # shoulder_lift pushes against something
    assert wait_for(lambda: runner.state == FAULT)
    assert (
        runner.fault_event.code == "overload" and runner.fault_event.joint == "left_shoulder_lift"
    )
    ok = runner.chunks_ok
    time.sleep(0.3)
    held = [sims["left"].get(i, P.GOAL_POSITION) for i in range(1, 7)]
    time.sleep(0.2)
    assert [sims["left"].get(i, P.GOAL_POSITION) for i in range(1, 7)] == held  # not creeping
    assert runner.chunks_ok == ok
    assert all(sims["left"].get(i, P.TORQUE_ENABLE) == 1 for i in range(1, 7))

    sims["left"].extra_load.clear()
    runner.resume()
    assert runner.state == HOLDING


def test_overtemperature_drops_only_that_arm(rig):
    """Over-temperature turns torque off on that arm only; the other arm keeps holding."""
    runner, sims = rig
    assert wait_for(lambda: np.isfinite(runner.q).all())
    runner.enable()
    sims["right"].set(4, P.PRESENT_TEMPERATURE, 70)
    assert wait_for(lambda: runner.state == FAULT and not runner.slots["right"].torque)
    # the slot flag drops before the six torque writes land: wait for the registers
    assert wait_for(lambda: all(sims["right"].get(i, P.TORQUE_ENABLE) == 0 for i in range(1, 7)))
    assert all(sims["left"].get(i, P.TORQUE_ENABLE) == 1 for i in range(1, 7))


def test_unplugged_bus_faults_instead_of_dying(rig):
    """An unplugged arm is a bus fault on that arm; the other keeps torque and no thread dies."""
    runner, sims = rig
    assert wait_for(lambda: np.isfinite(runner.q).all())
    runner.enable()
    runner.go()
    assert wait_for(lambda: runner.chunks_ok >= 1)
    sims["right"].unplug()
    assert wait_for(lambda: runner.state == FAULT)
    assert runner.fault_event.code == "bus" and runner.fault_event.arm == "right"
    assert runner.slots["left"].torque and all(
        sims["left"].get(i, P.TORQUE_ENABLE) == 1 for i in range(1, 7)
    )
    assert all(th.is_alive() for th in runner._threads)


def test_inference_error_pauses(rig):
    """An exception from the policy pauses the run instead of killing the inference thread."""
    runner, _ = rig

    class Broken:
        def infer(self, images, state):
            raise RuntimeError("CUDA error: an illegal memory access was encountered")

    runner.policy = Broken()
    assert wait_for(lambda: np.isfinite(runner.q).all())
    runner.enable()
    runner.go()
    assert wait_for(lambda: runner.state == HOLDING)
    assert all(th.is_alive() for th in runner._threads)


def test_enable_on_a_dead_bus_refuses_instead_of_raising(rig):
    """enable with an arm unplugged returns False, and the console command does not raise."""
    runner, sims = rig
    assert wait_for(lambda: np.isfinite(runner.q).all())
    sims["left"].unplug()
    assert runner.enable() is False
    runner.command("enable")


def test_enable_with_the_arm_resting_on_its_stops(rig, capsys):
    """Enable works with joints resting on their stops, outside the simulator's limits in degrees.

    Torque comes on there, and the joints ease inside the limits without a fault."""
    runner, sims = rig
    assert wait_for(lambda: np.isfinite(runner.q).all())
    arm = runner.slots["left"].arm
    elbow, lift = arm.joints.index("left_elbow_flex"), arm.joints.index("left_shoulder_lift")
    sims["left"].set_position(arm.ids[elbow], arm.cal[elbow].to_ticks(1.70))  # limit +1.571
    sims["left"].set_position(arm.ids[lift], arm.cal[lift].to_ticks(-1.83))  # limit -1.745
    ie, il = runner.joints.index("left_elbow_flex"), runner.joints.index("left_shoulder_lift")
    assert wait_for(lambda: abs(runner.q[ie] - 1.70) < 0.01 and abs(runner.q[il] + 1.83) < 0.01)
    t0 = time.monotonic()
    assert runner.enable() and runner.state == HOLDING
    assert "left_elbow_flex" in capsys.readouterr().out  # the operator is told it will move
    time.sleep(0.1)
    assert runner.q[ie] > 1.63  # 0.3 rad/s: not at the limit yet
    assert wait_for(
        lambda: (
            abs(runner.q[ie] - arm.upper[elbow]) < 0.01
            and abs(runner.q[il] - arm.lower[lift]) < 0.01
        )
    )
    assert time.monotonic() - t0 > 0.3
    assert runner.state == HOLDING and runner.fault_event is None


def test_replan_actions_sets_how_much_of_a_chunk_runs(rig):
    """replan_actions 45 at action_dt 0.02 s replans every 0.9 s, not every replan_period_s.

    The period is capped so the next chunk is ready before the current one ends."""
    runner, sims = rig
    ic = runner.cfg.inference
    ic.action_dt, ic.replan_actions, ic.replan_period_s = 0.02, 45, 0.05
    assert wait_for(lambda: np.isfinite(runner.q).all())
    runner.enable()
    runner.go()
    assert wait_for(lambda: runner.chunks_ok >= 1)
    assert runner._replan_period() == pytest.approx(0.65)  # 50 x 0.02 s less the 0.35 s margin
    ic.replan_margin_s = 0.05
    assert runner._replan_period() == pytest.approx(0.9)
    n0 = runner.seq
    time.sleep(2.0)
    n = runner.seq - n0
    assert 1 <= n <= 3, n  # ~2 in 2 s, where replan_period_s would give ~40
    ic.replan_actions = None
    assert runner._replan_period() == 0.05
