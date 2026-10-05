"""ArmSafety: telemetry faults, startup checks, and the output filter in front of the servos."""

import numpy as np

from so101_smolvla_edge.robot import protocol as P
from so101_smolvla_edge.robot.safety import HOLD, TORQUE_OFF, ArmSafety, SafetyConfig

JOINTS = ["a_j0", "a_j1", "a_gripper"]  # the name suffix marks the gripper
ZERO = np.zeros(3)  # rad, run()'s default target and measured pose


def tel(load=0, temp=30, volt=12.0):
    """Telemetry of one still servo: `load` in 0.1 % of max torque, `temp` in C, `volt` in V."""
    return P.Telemetry(
        position=2048,
        velocity=0,
        load=load,
        voltage=volt,
        temperature=temp,
        status=0,
        moving=0,
        current=0.0,
    )


def make(**kw):
    """ArmSafety for arm "a" with limits [-2, 2] rad, reset at zero at t = 0."""
    s = ArmSafety("a", JOINTS, SafetyConfig(**kw), lower=[-2] * 3, upper=[2] * 3)
    s.reset(np.zeros(3), 0.0)
    return s


def run(s, ticks, tel_fn, target=ZERO, present=ZERO, errors=None, dt=0.01, t0=0.0):
    """Check `ticks` ticks of `tel_fn()`, `dt` s apart after `t0`; return (first fault, last t)."""
    fault = None
    t = t0
    for _ in range(ticks):
        t += dt
        _, f = s.check(t, tel_fn(), present, target, errors or {}, [1, 2, 3])
        fault = fault or f
    return fault, t


def test_bus_failures_hold_after_n_ticks():
    """bus_fail_ticks incomplete reads in a row are a hold fault naming the missing servo."""
    s = make(bus_fail_ticks=5)
    f, _ = run(s, 4, lambda: [tel(), None, tel()])
    assert f is None
    f, _ = run(s, 1, lambda: [tel(), None, tel()])
    assert f.level == HOLD and f.code == "bus" and "a_j1" in f.msg


def test_overload_must_be_sustained():
    """An overload must last load_hard_s to fault; one normal tick restarts the timer."""
    s = make(load_hard_pct=90, load_hard_s=0.3)
    f, t = run(s, 20, lambda: [tel(load=950), tel(), tel()])  # 0.2 s
    assert f is None
    run(s, 1, lambda: [tel(), tel(), tel()], t0=t)
    f, _ = run(s, 25, lambda: [tel(load=-950), tel(), tel()], t0=t + 0.01)  # either sign counts
    assert f is None
    f, _ = run(s, 10, lambda: [tel(load=-950), tel(), tel()], t0=t + 0.26)
    assert f.code == "overload" and f.joint == "a_j0"


def test_gripper_squeeze_is_not_a_fault():
    """The gripper at full load is no fault: squeezing cloth at its torque limit is its job."""
    s = make()
    f, _ = run(s, 200, lambda: [tel(), tel(), tel(load=1000)])
    assert f is None


def test_temperature_ladder():
    """Warn, hold after temp_s, torque off when hotter; without temp_torque_off_c it only holds."""

    def hot(c):
        return lambda: [tel(temp=c), tel(), tel()]

    s = make()
    f, t = run(s, 10, hot(56))
    assert f is None and "temp:a_j0" in s._warned  # warned once 56 C is the median
    f, t = run(s, 40, hot(63), t0=t)
    assert f is None  # 0.4 s at 63 C, under temp_s
    f, t = run(s, 20, hot(63), t0=t)
    assert f.level == HOLD and f.code == "overtemp" and f.joint == "a_j0"
    s = make()
    f, _ = run(s, 70, hot(69))
    assert f.level == TORQUE_OFF
    s2 = make(temp_torque_off_c=None)
    f, _ = run(s2, 70, hot(69))
    assert f.level == HOLD


def test_single_sample_outliers_are_not_faults():
    """Single-tick temperature and voltage spikes, even two in a row, raise no fault or warning.

    The STS3215s report such spikes now and then, with valid checksums. A sustained
    overtemperature still holds."""
    s = make(voltage_min=6.0, voltage_max=9.0, voltage_warn_margin=0.5)
    _, t = run(s, 5, lambda: [tel(temp=31, volt=7.6)] * 3)  # fills the median window
    n = 0
    for i in range(3000):  # 30 s
        t += 0.01
        temp = 80 if i % 97 == 0 else 62 if i % 97 == 1 else 31  # two consecutive outliers
        volt = 5.0 if i % 89 == 0 else 6.1 if i % 53 == 0 else 7.6
        tels = [tel(temp=temp, volt=volt), tel(temp=31, volt=7.6), tel(temp=31, volt=7.6)]
        ev, f = s.check(t, tels, np.zeros(3), np.zeros(3), {}, [1, 2, 3])
        assert f is None
        n += len(ev)
    assert n == 0
    f, t = run(s, 60, lambda: [tel(temp=63, volt=7.6)] * 3, t0=t)
    assert f.level == HOLD and f.code == "overtemp"


def test_voltage_window():
    """A supply voltage below voltage_min for voltage_s is a voltage fault."""
    s = make(voltage_min=6.0, voltage_s=0.2)
    f, _ = run(s, 30, lambda: [tel(volt=5.5)] * 3)
    assert f.code == "voltage"


def test_tracking_error():
    """A joint more than tracking_error_rad off its target for tracking_s is a tracking fault."""
    s = make(tracking_error_rad=0.5, tracking_s=0.4)
    f, _ = run(s, 30, lambda: [tel()] * 3, target=np.array([0.0, 0.8, 0.0]))
    assert f is None
    f, _ = run(s, 20, lambda: [tel()] * 3, target=np.array([0.0, 0.8, 0.0]), t0=0.3)
    assert f.code == "tracking" and f.joint == "a_j1"


def test_servo_error_bits():
    """Error bits in servo_error_hold (overload) fault the joint; others (angle) do not."""
    s = make()
    _, f = s.check(0.01, [tel()] * 3, np.zeros(3), np.zeros(3), {2: P.ERR_OVERLOAD}, [1, 2, 3])
    assert f.code == "servo_error" and f.joint == "a_j1"
    _, f = s.check(0.02, [tel()] * 3, np.zeros(3), np.zeros(3), {3: P.ERR_ANGLE}, [1, 2, 3])
    assert f is None


def test_output_slew_and_lead_clamp():
    """The output slews and caps its lead over the measured position, tighter under load.

    When the load goes away the goal slews out again instead of jumping to the wider cap."""
    s = make(
        max_velocity=1.0,
        out_slew_margin=1.0,
        max_lead_rad=0.5,
        max_lead_loaded_rad=0.1,
        load_soft_pct=50,
    )
    g = s.output(0.01, np.array([1.0, 1.0, 0.0]), np.zeros(3), [tel()] * 3).goal
    assert np.allclose(g, [0.01, 0.01, 0.0])  # 1 rad/s * 10 ms
    t = 0.01
    for _ in range(100):
        t += 0.01
        r = s.output(t, np.array([1.0, 1.0, 0.0]), np.zeros(3), [tel(), tel(load=600), tel()])
    assert r.goal[0] == 0.5 and abs(r.goal[1] - 0.1) < 1e-12
    assert r.lead_clamped[:2].all()
    r = s.output(t + 0.01, np.array([1.0, 1.0, 0.0]), np.zeros(3), [tel()] * 3)
    assert abs(r.goal[1] - 0.11) < 1e-9


def test_startup_refuses_outside_limits():
    """check_startup refuses joints outside the limits and joints it cannot read."""
    s = make()
    assert s.check_startup(np.array([0.0, 0.0, 0.0])) == []
    bad = s.check_startup(np.array([0.0, 2.2, np.nan]))
    assert len(bad) == 2


def test_enable_on_the_mechanical_stops_then_ease_inside_the_limits():
    """An arm resting on its stops may enable, then its goal eases inside the limits.

    In the degrees convention the stops read outside the simulator's limits. Within the calibrated
    range torque may come on, and the goal moves in at limit_recover_speed instead of jumping."""
    s = ArmSafety(
        "a",
        JOINTS,
        SafetyConfig(max_velocity=3.0, out_slew_margin=1.0, limit_recover_speed=0.3),
        lower=[-1.5] * 3,
        upper=[1.5] * 3,
        reach_lower=[-1.7] * 3,
        reach_upper=[1.7] * 3,
    )
    rest = np.array([1.64, -1.62, 0.0])  # on the stops: 0.14 and 0.12 outside the limits
    assert s.check_startup(rest) == []
    assert len(s.outside_limits(rest)) == 2
    assert len(s.check_startup(np.array([1.80, 0.0, 0.0]))) == 1  # beyond the calibrated range
    s.reset(rest, 0.0)
    target = np.clip(rest, -1.5, 1.5)  # what the executor asks for: the limit
    t, goals = 0.0, []
    for _ in range(60):
        t += 0.01
        goals.append(s.output(t, target, rest, [tel()] * 3).goal)
    goals = np.array(goals)
    assert np.allclose(goals[0], [1.637, -1.617, 0.0])  # 0.3 rad/s * 10 ms, no jump
    assert np.abs(np.diff(goals, axis=0)).max() <= 0.3 * 0.01 + 1e-12
    assert np.allclose(goals[-1], [1.5, -1.5, 0.0])  # inside after ~0.47 s
    # once inside: the normal slew limit, and the limits are hard again
    g = s.output(t + 0.01, np.array([0.0, 0.0, 0.0]), goals[-1], [tel()] * 3).goal
    assert np.allclose(g, [1.47, -1.47, 0.0])
    g = s.output(t + 0.02, np.array([9.0, -9.0, 0.0]), g, [tel()] * 3).goal
    assert (np.abs(g) <= 1.5).all()


def test_zero_durations_mean_immediately_not_always():
    """A duration of 0 s acts on the first bad tick, and never on good ones."""
    s = make(
        temp_s=0.0,
        voltage_s=0.0,
        load_hard_s=0.0,
        tracking_s=0.0,
        voltage_min=6.0,
        voltage_max=14.0,
    )
    f, _ = run(s, 20, lambda: [tel(temp=31, volt=12.0)] * 3)
    assert f is None
    f, _ = run(s, 5, lambda: [tel(temp=70, volt=12.0)] * 3, t0=1.0)
    assert f is not None and f.code == "overtemp"
