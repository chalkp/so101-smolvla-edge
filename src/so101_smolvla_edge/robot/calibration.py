"""Servo ticks <-> policy radians (the simulator's joint angles), from a lerobot calibration file.

Two conventions, set per arm with `convention`:
- degrees (default): arm joints are true angles about the middle of the calibrated range, which
  calibration puts at the simulator's zero pose (lerobot MotorNormMode.DEGREES). This is what the
  LeHome winner ran on real SO-101s (lehome_solution/training/real_data_transforms.py).
- stretched: each calibrated range is mapped linearly onto the simulator's limits, the way the
  simulator dataset was made from leader arms. On a follower that rescales every joint by swept
  range / simulator range, so it is kept for comparison only.
The gripper is always lerobot's 0..100 stretched onto the simulator's [-10, 100] deg.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

MOTORS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
CONVENTIONS = ("degrees", "stretched")
TICKS_PER_TURN = 4095.0  # lerobot divides by resolution - 1 (STS3215: 4096 positions)

# deg, from lehome/assets/robots/lerobot.py: SO101_FOLLOWER_USD_JOINT_LIMLITS
LEHOME_LIMITS_DEG = {
    "shoulder_pan": (-110.0, 110.0),
    "shoulder_lift": (-100.0, 100.0),
    "elbow_flex": (-100.0, 90.0),
    "wrist_flex": (-95.0, 95.0),
    "wrist_roll": (-160.0, 160.0),
    "gripper": (-10.0, 100.0),
}


@dataclass(frozen=True)
class JointCal:
    name: str  # policy joint name, e.g. "left_shoulder_pan"
    motor: str  # lerobot motor name
    id: int  # servo id on the bus
    drive_mode: int  # non-zero: direction inverted
    homing_offset: int  # ticks, as written to the servo
    range_min: int  # ticks
    range_max: int  # ticks
    convention: str = "degrees"  # one of CONVENTIONS

    @property
    def gripper(self) -> bool:
        return self.motor == "gripper"

    @property
    def stretched(self) -> bool:
        """Range mapped onto the simulator's limits: always for the gripper, else per convention."""
        return self.gripper or self.convention == "stretched"

    @property
    def mid(self) -> float:
        """Middle of the calibrated range, ticks."""
        return (self.range_min + self.range_max) / 2.0

    @property
    def limits_deg(self) -> tuple[float, float]:
        return LEHOME_LIMITS_DEG[self.motor]

    @property
    def lower(self) -> float:
        """Lowest commandable angle, rad: the simulator limit, or the calibrated end if tighter."""
        lo = math.radians(self.limits_deg[0])
        if self.stretched:
            return lo
        return max(lo, min(self.to_rad(self.range_min), self.to_rad(self.range_max)))

    @property
    def upper(self) -> float:
        """Highest commandable angle, rad, as `lower`."""
        hi = math.radians(self.limits_deg[1])
        if self.stretched:
            return hi
        return min(hi, max(self.to_rad(self.range_min), self.to_rad(self.range_max)))

    def _norm_range(self) -> tuple[float, float]:
        """lerobot's normalized range for this joint."""
        return (0.0, 100.0) if self.gripper else (-100.0, 100.0)

    def to_rad(self, ticks: float) -> float:
        """Servo ticks to policy radians, clamped to the calibrated range first."""
        mn, mx = self.range_min, self.range_max
        clamped = min(max(ticks, mn), mx)
        if not self.stretched:
            deg = (clamped - self.mid) * 360.0 / TICKS_PER_TURN
            return math.radians(-deg if self.drive_mode else deg)
        if self.gripper:
            norm = (clamped - mn) / (mx - mn) * 100.0
            if self.drive_mode:
                norm = 100.0 - norm
        else:
            norm = (clamped - mn) / (mx - mn) * 200.0 - 100.0
            if self.drive_mode:
                norm = -norm
        lo, hi = self._norm_range()
        dmin, dmax = self.limits_deg
        return math.radians((norm - lo) / (hi - lo) * (dmax - dmin) + dmin)

    def to_ticks(self, rad: float) -> int:
        """Policy radians to the nearest servo tick, clamped to the calibrated range."""
        if not self.stretched:
            deg = math.degrees(rad)
            t = (-deg if self.drive_mode else deg) * TICKS_PER_TURN / 360.0 + self.mid
            return int(round(min(max(t, self.range_min), self.range_max)))
        lo, hi = self._norm_range()
        dmin, dmax = self.limits_deg
        norm = (math.degrees(rad) - dmin) / (dmax - dmin) * (hi - lo) + lo
        if self.gripper:
            if self.drive_mode:
                norm = 100.0 - norm
            norm = min(max(norm, 0.0), 100.0)
            return int(round(norm / 100.0 * (self.range_max - self.range_min) + self.range_min))
        if self.drive_mode:
            norm = -norm
        norm = min(max(norm, -100.0), 100.0)
        return int(
            round((norm + 100.0) / 200.0 * (self.range_max - self.range_min) + self.range_min)
        )

    def rad_per_tick(self) -> float:
        """Slope of to_rad inside the calibrated range, signed by the drive mode."""
        if not self.stretched:
            s = math.radians(360.0 / TICKS_PER_TURN)
            return -s if self.drive_mode else s
        dmin, dmax = self.limits_deg
        s = math.radians(dmax - dmin) / (self.range_max - self.range_min)
        return -s if self.drive_mode else s


def load_lerobot_calibration(
    path: str | Path, prefix: str, convention: str = "degrees"
) -> list[JointCal]:
    """Read a lerobot calibration JSON into one JointCal per motor, in MOTORS order.

    The file maps each motor to {"id", "drive_mode", "homing_offset", "range_min", "range_max"}.
    Joints are named prefix + motor.
    """
    if convention not in CONVENTIONS:
        raise ValueError(f"calibration convention must be one of {CONVENTIONS}, not {convention!r}")
    path = Path(path)
    j = json.loads(path.read_text())
    out = []
    for m in MOTORS:
        if m not in j:
            raise ValueError(f"{path}: no entry for motor {m!r}")
        e = j[m]
        c = JointCal(
            name=prefix + m,
            motor=m,
            id=int(e["id"]),
            drive_mode=int(e.get("drive_mode", 0)),
            homing_offset=int(e["homing_offset"]),
            range_min=int(e["range_min"]),
            range_max=int(e["range_max"]),
            convention=convention,
        )
        if c.range_max <= c.range_min:
            raise ValueError(f"{path}: {m} has range_max <= range_min")
        out.append(c)
    return out
