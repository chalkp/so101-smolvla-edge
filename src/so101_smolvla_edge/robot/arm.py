"""One SO-101 follower arm on its own bus: verification, configuration, telemetry, goals, torque.

Connecting checks that every servo is an STS3215 whose calibration registers match the calibration
file. A swapped USB port or a stale file would otherwise map the policy's radians onto the wrong
joint or range. Torque is never disabled on exit, and configuration writes only what differs, so
restarting while the arm holds writes nothing and does not drop it.
Ported from ros2_ws/src/robot_server/src/so101_backend.cpp.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from . import protocol as P
from .bus import Bus
from .calibration import JointCal, load_lerobot_calibration

log = logging.getLogger("so101_smolvla_edge.arm")


@dataclass
class ServoSettings:
    """Servo register values, lerobot's SO-101 follower defaults."""

    return_delay_time: int = 0
    p_coefficient: int = 16  # lowered from the default 32 to avoid shakiness
    i_coefficient: int = 0
    d_coefficient: int = 32
    acceleration: int = 254
    maximum_acceleration: int = 254
    # gripper only, lowered so a long squeeze does not burn out its motor
    gripper_max_torque_limit: int = 500  # 0.1 % units: 50 % of max torque
    gripper_protection_current: int = 250  # 50 % of max current
    gripper_overload_torque: int = 25  # % torque when overloaded
    torque_limit: int | None = None  # SRAM Torque_Limit for arm joints (0..1000), None = leave it
    gripper_torque_limit: int | None = None  # the same for the gripper


class ArmError(RuntimeError):
    """The arm failed verification or configuration."""


class Arm:
    def __init__(
        self,
        name: str,
        port: str,
        calibration: str,
        baud: int = 1_000_000,
        echo: str = "auto",
        reply_margin_s: float = 0.0025,
        settings: ServoSettings | None = None,
        bus: Bus | None = None,
        convention: str = "degrees",
    ):
        self.name = name
        self.cal: list[JointCal] = load_lerobot_calibration(calibration, name + "_", convention)
        self.ids = [c.id for c in self.cal]  # servo ids, joint order
        self.joints = [c.name for c in self.cal]  # policy joint names
        self.lower = np.array([c.lower for c in self.cal])  # rad, commandable
        self.upper = np.array([c.upper for c in self.cal])  # rad, commandable
        # The calibrated range in the same radians. In the `degrees` convention it is wider than the
        # commandable range: an arm resting on its stops reads outside the simulator's limits, and
        # torque must still be allowed on from there.
        ends = np.array([[c.to_rad(c.range_min), c.to_rad(c.range_max)] for c in self.cal])
        self.reach_lower = np.minimum(ends.min(axis=1), self.lower)
        self.reach_upper = np.maximum(ends.max(axis=1), self.upper)
        self.settings = settings or ServoSettings()
        self.echo_mode = echo  # "auto", "on" or "off"
        self.bus = bus or Bus(port, baud, reply_margin_s, echo={"on": True, "off": False}.get(echo))

    # ------------------------------------------------------------------ connect
    def connect(self, verify_calibration: bool = True, configure: bool = True) -> None:
        """Open the bus, check every servo's model and calibration registers, then configure.

        Raises ArmError, before any register is written, if a servo is missing, is not an STS3215
        or holds a different calibration than the file.
        """
        self.bus.open()
        if self.echo_mode == "auto":
            e = self.bus.detect_echo(self.ids[0])
            if e is None:
                raise ArmError(
                    f"{self.name} arm on {self.bus.port_name}: servo {self.ids[0]} did not answer "
                    "(power? wiring? wrong port?)"
                )
            log.info("%s: adapter echo %s", self.name, "on" if e else "off")
        problems = []
        for c in self.cal:
            model = self.bus.read_reg(c.id, P.MODEL_NUMBER)
            if model is None:
                problems.append(f"  {c.name} (id {c.id}): no answer")
                continue
            if model != P.MODEL_STS3215:
                problems.append(
                    f"  {c.name} (id {c.id}): model {model}, expected STS3215 ({P.MODEL_STS3215})"
                )
                continue
            if verify_calibration:
                hom = self.bus.read_reg(c.id, P.HOMING_OFFSET)
                mn = self.bus.read_reg(c.id, P.MIN_POSITION_LIMIT)
                mx = self.bus.read_reg(c.id, P.MAX_POSITION_LIMIT)
                if None in (hom, mn, mx):
                    problems.append(f"  {c.name}: could not read calibration registers")
                    continue
                hom = P.decode_sign_magnitude(hom, P.HOMING_OFFSET_SIGN_BIT)
                if (hom, mn, mx) != (c.homing_offset, c.range_min, c.range_max):
                    problems.append(
                        f"  {c.name}: servo has homing_offset/min/max {hom}/{mn}/{mx}, "
                        f"calibration file says {c.homing_offset}/{c.range_min}/{c.range_max}"
                    )
        if problems:
            raise ArmError(
                f"{self.name} arm on {self.bus.port_name} failed verification:\n"
                + "\n".join(problems)
                + "\nWrong port, wrong calibration file, or the arm was recalibrated "
                "since the file was written."
            )
        if configure:
            self.configure()

    def configure(self) -> None:
        """Bring the servo registers to `settings`, writing only those that differ.

        EEPROM writes need torque off, so when any differ every joint drops torque briefly, and
        joints that were holding get their goal set to where they are and torque back on. Raises
        ArmError if a holding joint's position is unreadable or a setting fails to write.
        """
        s = self.settings
        pending: list[tuple[JointCal, P.Reg, int]] = []
        for c in self.cal:
            want = [
                (P.RETURN_DELAY_TIME, s.return_delay_time),
                (P.OPERATING_MODE, 0),
                (P.P_COEFFICIENT, s.p_coefficient),
                (P.I_COEFFICIENT, s.i_coefficient),
                (P.D_COEFFICIENT, s.d_coefficient),
                (P.MAXIMUM_ACCELERATION, s.maximum_acceleration),
                (P.ACCELERATION, s.acceleration),
            ]
            if c.gripper:
                want += [
                    (P.MAX_TORQUE_LIMIT, s.gripper_max_torque_limit),
                    (P.PROTECTION_CURRENT, s.gripper_protection_current),
                    (P.OVERLOAD_TORQUE, s.gripper_overload_torque),
                ]
            for reg, value in want:
                if self.bus.read_reg(c.id, reg) != value:
                    pending.append((c, reg, value))
        if pending:
            log.warning(
                "%s: writing %d servo setting(s) that differ; torque drops briefly while EEPROM is "
                "written, so support the arm",
                self.name,
                len(pending),
            )
            was_on, present = {}, {}
            for c in self.cal:
                t = self.bus.read_reg(c.id, P.TORQUE_ENABLE)
                p = self.bus.read_reg(c.id, P.PRESENT_POSITION)
                was_on[c.id], present[c.id] = bool(t), p
                if t and p is None:
                    raise ArmError(
                        f"{c.name}: holding torque but position unreadable; "
                        "refusing to reconfigure blind"
                    )
            for c in self.cal:
                self.bus.write_reg(c.id, P.TORQUE_ENABLE, 0)
                self.bus.write_reg(c.id, P.LOCK, 0)
            failed = [
                f"{c.name}@{reg.addr}"
                for c, reg, v in pending
                if not self.bus.write_reg(c.id, reg, v)
            ]
            for c in self.cal:
                if was_on[c.id]:
                    self.bus.write_reg(c.id, P.GOAL_POSITION, present[c.id])
                    self.bus.write_reg(c.id, P.TORQUE_ENABLE, 1)
                    self.bus.write_reg(c.id, P.LOCK, 1)
            if failed:
                raise ArmError(f"{self.name}: configuration writes failed for {' '.join(failed)}")
        for c in self.cal:
            lim = s.gripper_torque_limit if c.gripper else s.torque_limit
            if lim is not None and self.bus.read_reg(c.id, P.TORQUE_LIMIT) != lim:
                self.bus.write_reg(c.id, P.TORQUE_LIMIT, lim)

    def close(self) -> None:
        """Close the port and leave torque as it is: torque-off drops an SO-101 arm."""
        self.bus.close()

    # ------------------------------------------------------------------ per tick
    def read(self) -> tuple[np.ndarray, list[P.Telemetry | None], dict[int, int]]:
        """Read every servo's telemetry in one SYNC_READ.

        Returns positions (rad, NaN where a servo did not answer), telemetry (None where missing)
        and the error bytes servos reported, by id.
        """
        self.bus.stats.error_bits.clear()
        r = self.bus.sync_read(self.ids, P.TELEMETRY_ADDR, P.TELEMETRY_LEN)
        tel = [P.Telemetry.decode(x.params) if x is not None else None for x in r]
        pairs = zip(self.cal, tel, strict=True)
        q = np.array([c.to_rad(t.position) if t is not None else np.nan for c, t in pairs])
        return q, tel, dict(self.bus.stats.error_bits)

    def write_goal(self, q: np.ndarray) -> None:
        """Send goal positions (rad, joint order) in one SYNC_WRITE."""
        goals = [(c.id, P.le(c.to_ticks(float(v)), 2)) for c, v in zip(self.cal, q, strict=False)]
        self.bus.sync_write(P.GOAL_POSITION.addr, 2, goals)

    def torque_state(self) -> list[int | None]:
        """Torque_Enable per joint, None where unreadable."""
        return [self.bus.read_reg(c.id, P.TORQUE_ENABLE) for c in self.cal]

    def set_torque(self, on: bool, hold_q: np.ndarray | None = None) -> bool:
        """Switch torque on every joint; True if every write succeeded.

        Turning it on needs `hold_q`, the measured positions (rad), to hold.
        """
        ok = True
        if on:
            if hold_q is None:
                raise ValueError("enabling torque needs the hold pose (the measured positions)")
            self.write_goal(hold_q)  # before torque, or the arm jumps to a stale goal
            for c in self.cal:
                ok &= self.bus.write_reg(c.id, P.TORQUE_ENABLE, 1)
                ok &= self.bus.write_reg(c.id, P.LOCK, 1)
        else:
            for c in self.cal:
                ok &= self.bus.write_reg(c.id, P.TORQUE_ENABLE, 0)
                ok &= self.bus.write_reg(c.id, P.LOCK, 0)
        return ok
