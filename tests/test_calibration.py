"""Servo ticks to policy radians in both calibration conventions, and the `calibration` report."""

import math

import numpy as np

from so101_smolvla_edge.robot.calibration import load_lerobot_calibration


def test_stretched_roundtrip_and_limits(calib_file):
    """Stretched: the range spans the simulator's limits, round-trips to half a tick, clamps."""
    cal = load_lerobot_calibration(calib_file(), "left_", "stretched")
    assert [c.name for c in cal][0] == "left_shoulder_pan" and cal[-1].gripper
    for c in cal:
        lo, hi = c.limits_deg
        assert math.isclose(c.to_rad(c.range_min), math.radians(lo)) and math.isclose(
            c.to_rad(c.range_max), math.radians(hi)
        )
        for q in np.linspace(c.lower, c.upper, 17):
            assert abs(c.to_rad(c.to_ticks(q)) - q) <= abs(c.rad_per_tick()) / 2 + 1e-12
        assert c.to_ticks(c.upper + 1.0) == c.range_max
        assert c.to_rad(c.range_min - 100) == c.to_rad(c.range_min)


def test_degrees_is_true_angle_about_the_midpoint(calib_file):
    """Degrees (the default): deg = (ticks - mid) * 360 / 4095, whatever the swept range.

    The gripper stays 0..100 stretched onto [-10, 100] deg, as in LeHome's real_data_transforms.
    The commandable range ends at the simulator's limit or at the calibrated range's end, whichever
    is nearer."""
    cal = load_lerobot_calibration(calib_file(range_min=600, range_max=3500), "left_")
    pan, grip = cal[0], cal[-1]
    assert pan.convention == "degrees" and math.isclose(pan.to_rad(2050), 0.0)  # mid = 2050
    assert math.isclose(math.degrees(pan.to_rad(2050 + 1024)), 1024 * 360 / 4095)  # ~90 deg
    assert abs(math.degrees(pan.to_rad(pan.to_ticks(math.radians(45.0)))) - 45.0) < 0.05
    assert math.isclose(grip.to_rad(600), math.radians(-10.0)) and math.isclose(
        grip.to_rad(3500), math.radians(100.0)
    )
    reach = math.radians(1450 * 360 / 4095)  # 1450 ticks either side of mid: +-127.5 deg
    assert math.isclose(pan.upper, math.radians(110.0))  # the simulator's limit is nearer
    roll = cal[4]
    assert math.isclose(roll.upper, reach) and roll.to_ticks(math.radians(160.0)) == 3500
    for c in cal:
        for q in np.linspace(c.lower, c.upper, 17):
            assert abs(c.to_rad(c.to_ticks(q)) - q) <= abs(c.rad_per_tick()) / 2 + 1e-9


def test_degrees_wrist_roll_full_turn_is_not_scaled(calib_file):
    """Why degrees is the default: a full-turn wrist_roll told 90 deg goes to 90, not 101.25."""
    f = calib_file(range_min=0, range_max=4095)
    deg = load_lerobot_calibration(f, "")[4]
    st = load_lerobot_calibration(f, "", "stretched")[4]
    assert abs((deg.to_ticks(math.radians(90.0)) - 2047.5) * 360 / 4095 - 90.0) < 0.1
    assert abs((st.to_ticks(math.radians(90.0)) - 2047.5) * 360 / 4095 - 101.25) < 0.1


def test_drive_mode_flips(calib_file):
    """drive_mode 1 reverses a joint's direction in both conventions."""
    for conv in ("degrees", "stretched"):
        a = load_lerobot_calibration(calib_file("a"), "", conv)[0]
        b = load_lerobot_calibration(calib_file("b", drive_mode=1), "", conv)[0]
        assert math.isclose(a.to_rad(1000), -b.to_rad(1000))
