"""Per-arm safety: telemetry checks, a latched fault state, and the last filter before the servos.

Fault levels, mildest first. warn is only logged. hold is latched until the operator resumes: the
policy pauses on both arms and every joint holds the position measured when the fault fired, not
the commanded one, so force built up against an obstacle is released. torque_off is kept for
over-temperature and the operator's stop, because torque-off drops an SO-101 arm and is only worth
it when holding would cook a servo.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field

import numpy as np

from . import protocol as P

# fault levels
OK = "ok"
WARN = "warn"
HOLD = "hold"
TORQUE_OFF = "torque_off"


@dataclass
class SafetyConfig:
    max_velocity: float = 3.33  # rad/s executor slew, the simulator's 0.111 rad per action at 30 Hz
    max_velocity_gripper: float = 3.33  # rad/s
    out_slew_margin: float = 1.15  # output filter slew = max_velocity * this
    max_lead_rad: float = 0.5  # rad, cap on |goal - measured| for every joint
    max_lead_loaded_rad: float = 0.15  # rad, the same cap while |load| >= load_soft_pct
    max_lead_loaded_rad_gripper: float = 0.25  # rad
    load_soft_pct: float = 60.0  # % of max torque
    load_soft_pct_gripper: float = 40.0  # %
    load_hard_pct: float | None = 90.0  # %, sustained for load_hard_s is a hold fault
    load_hard_pct_gripper: float | None = None  # None: squeezing cloth at max torque is its job
    load_hard_s: float = 0.30  # s
    tracking_error_rad: float = 0.6  # rad, |executor target - measured| sustained is a hold fault
    tracking_s: float = 0.5  # s
    temp_warn_c: float = 55.0  # deg C
    temp_hold_c: float = 62.0  # deg C
    temp_torque_off_c: float | None = 68.0  # deg C, None: hold only (the servo cuts out at 70 C)
    temp_s: float = 0.5  # s, how long a temperature threshold must be exceeded to act
    telemetry_median: int = 5  # judge temperature/voltage on the median of this many samples;
    # STS3215s report occasional single-sample spikes on both, and one spike is not a fault
    voltage_min: float = 6.0  # V, sustained outside [min, max] is a hold fault. Set it for your
    voltage_max: float = 14.0  # supply: the 7.4 V and 12 V STS3215 variants need different windows
    voltage_warn_margin: float = 0.5  # V
    voltage_s: float = 0.2  # s
    bus_fail_ticks: int = 20  # consecutive incomplete telemetry reads before a hold fault
    servo_error_hold: tuple[str, ...] = ("overheat", "overload", "overcurrent")  # others only warn
    joint_limit_margin_rad: float = 0.05  # rad; refuse to enable this far outside calibrated range
    limit_recover_speed: float = 0.3  # rad/s; a joint enabled outside its commandable limits (an
    # arm resting on its stops) is brought inside them this slowly


@dataclass
class Event:
    t: float  # time.monotonic()
    arm: str
    level: str  # warn | hold | torque_off | info
    code: str  # cause, e.g. "overtemp"
    msg: str  # for the operator
    joint: str | None = None  # None: the whole arm
    value: float | None = None  # the reading that triggered it, in its own unit

    def as_dict(self) -> dict:
        """The fields that are set, for the run log."""
        return {k: v for k, v in self.__dict__.items() if v is not None}


class _Timer:
    """Tracks how long a per-joint condition has held continuously."""

    def __init__(self, n: int):
        self.since = np.full(n, math.nan)

    def update(self, t: float, cond: np.ndarray) -> np.ndarray:
        """Seconds each condition has held, -inf where it does not (never >= a limit, even 0 s)."""
        self.since = np.where(cond, np.where(np.isnan(self.since), t, self.since), math.nan)
        return np.where(cond, t - self.since, -math.inf)  # -inf never reaches a limit, even 0 s


@dataclass
class TickResult:
    goal: np.ndarray  # rad to write
    events: list[Event] = field(default_factory=list)  # warnings raised this tick
    fault: Event | None = None  # a new hold/torque_off fault raised this tick
    lead_clamped: np.ndarray | None = None  # per joint, the lead clamp moved the goal


class ArmSafety:
    def __init__(
        self,
        arm: str,
        joints: list[str],
        cfg: SafetyConfig,
        lower: np.ndarray,
        upper: np.ndarray,
        reach_lower: np.ndarray | None = None,
        reach_upper: np.ndarray | None = None,
    ):
        """`lower`/`upper` (rad) bound what may be commanded. `reach_lower`/`reach_upper` are where
        the joints can physically be, the calibrated range, and default to the commandable limits.
        """
        self.arm = arm
        self.joints = joints
        self.cfg = cfg
        n = len(joints)
        self.lower = np.asarray(lower, float)
        self.upper = np.asarray(upper, float)
        self.reach_lower = (
            self.lower
            if reach_lower is None
            else np.minimum(np.asarray(reach_lower, float), self.lower)
        )
        self.reach_upper = (
            self.upper
            if reach_upper is None
            else np.maximum(np.asarray(reach_upper, float), self.upper)
        )
        g = np.array([j.endswith("gripper") for j in joints])

        def pick(a, b):  # per joint: a on the gripper, b elsewhere, None meaning no limit
            return np.where(g, a if a is not None else np.inf, b if b is not None else np.inf)

        self.vmax_out = (
            np.where(g, cfg.max_velocity_gripper, cfg.max_velocity) * cfg.out_slew_margin
        )
        self.lead = np.full(n, cfg.max_lead_rad)
        self.lead_loaded = np.where(g, cfg.max_lead_loaded_rad_gripper, cfg.max_lead_loaded_rad)
        self.load_soft = pick(cfg.load_soft_pct_gripper, cfg.load_soft_pct)
        self.load_hard = pick(cfg.load_hard_pct_gripper, cfg.load_hard_pct)
        self._load_t = _Timer(n)
        self._track_t = _Timer(n)
        self._volt_t = _Timer(n)
        self._temp_hold_t = _Timer(n)
        self._temp_off_t = _Timer(n)
        m = max(1, int(cfg.telemetry_median))
        self._temp_hist = np.full((m, n), np.nan)  # ring buffer, (m, n)
        self._volt_hist = np.full((m, n), np.nan)  # ring buffer, (m, n)
        self._hist_i = 0  # samples written to the ring buffers
        self._bus_fails = 0  # consecutive incomplete reads
        self._last_goal: np.ndarray | None = None  # rad, last goal written
        self._last_t = 0.0
        self._warned: set[str] = set()  # "code:joint" warnings already given, re-armed on clear

    # ------------------------------------------------------------------ startup
    def check_startup(self, present: np.ndarray) -> list[str]:
        """Reasons not to enable torque at `present` (rad), empty if there are none.

        A joint outside the commandable limits but inside its calibrated range passes: `output`
        brings it in (see `outside_limits`).
        """
        m = self.cfg.joint_limit_margin_rad
        bad = []
        for j, q in enumerate(present):
            if not np.isfinite(q):
                bad.append(f"{self.joints[j]}: position unreadable")
            elif q < self.reach_lower[j] - m or q > self.reach_upper[j] + m:
                bad.append(
                    f"{self.joints[j]}: {q:+.3f} rad outside its calibrated range "
                    f"[{self.reach_lower[j]:+.3f}, {self.reach_upper[j]:+.3f}]"
                )
        return bad

    def outside_limits(self, present: np.ndarray) -> list[str]:
        """Joints that sit outside the commandable limits (to tell the operator they will move)."""
        present = np.asarray(present, float)
        d = np.maximum(self.lower - present, present - self.upper)
        return [
            f"{self.joints[j]} {d[j]:.2f} rad" for j in np.flatnonzero(np.isfinite(d) & (d > 1e-3))
        ]

    def reset(self, present: np.ndarray, t: float) -> None:
        """Restart the output filter from `present` (rad) at time t and clear the fault timers."""
        self._last_goal = np.asarray(present, float).copy()
        self._last_t = t
        self._load_t.since[:] = math.nan
        self._track_t.since[:] = math.nan
        self._volt_t.since[:] = math.nan
        self._temp_hold_t.since[:] = math.nan
        self._temp_off_t.since[:] = math.nan
        self._bus_fails = 0

    def _filtered(self, temp: np.ndarray, volt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Per-joint median of the last `telemetry_median` readings, NaN meaning no reading.

        A servo that has not answered within the window gives -inf temperature and NaN voltage,
        which no threshold acts on.
        """
        i = self._hist_i % len(self._temp_hist)
        self._temp_hist[i], self._volt_hist[i] = temp, volt
        self._hist_i += 1
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN column: handled below
            tf, vf = np.nanmedian(self._temp_hist, axis=0), np.nanmedian(self._volt_hist, axis=0)
        return np.where(np.isnan(tf), -np.inf, tf), vf

    # ------------------------------------------------------------------ per tick
    def _warn_once(
        self,
        ev: list[Event],
        t: float,
        code: str,
        msg: str,
        joint: str | None = None,
        value: float | None = None,
    ) -> None:
        """Append a warning unless this code already warned for this joint since `_clear_warn`."""
        key = f"{code}:{joint}"
        if key not in self._warned:
            self._warned.add(key)
            ev.append(Event(t, self.arm, WARN, code, msg, joint, value))

    def _clear_warn(self, code: str, joint: str | None = None) -> None:
        self._warned.discard(f"{code}:{joint}")

    def check(
        self,
        t: float,
        tel: list[P.Telemetry | None],
        present: np.ndarray,
        target: np.ndarray,
        errors: dict[int, int],
        ids: list[int],
    ) -> tuple[list[Event], Event | None]:
        """Telemetry checks for one tick; returns (warnings, new fault or None).

        `present` (rad) is NaN where a servo did not answer and `target` is the executor's command.
        Checks run bus, servo errors, temperature, voltage, load, tracking; the first fault wins.
        Apart from servo-reported errors, a fault needs its condition to persist, and temperature
        and voltage are median-filtered first: a servo does not heat 30 C in 10 ms.
        """
        c, ev = self.cfg, []
        valid = np.array([x is not None for x in tel])
        if not valid.all():
            self._bus_fails += 1
            if self._bus_fails >= c.bus_fail_ticks:
                missing = [self.joints[j] for j in np.flatnonzero(~valid)]
                msg = (
                    f"{self._bus_fails} consecutive incomplete reads; "
                    f"no reply from {', '.join(missing)}"
                )
                return ev, Event(t, self.arm, HOLD, "bus", msg)
        else:
            self._bus_fails = 0
        if not valid.any():
            return ev, None
        load = np.array([abs(x.load) / 10.0 if x is not None else 0.0 for x in tel])  # %
        temp, volt = self._filtered(
            np.array([x.temperature if x is not None else np.nan for x in tel], float),
            np.array([x.voltage if x is not None else np.nan for x in tel], float),
        )
        for sid, bits in errors.items():
            if sid in ids:
                j = ids.index(sid)
                names = [n for b, n in P.ERROR_NAMES.items() if bits & b]
                if any(n in c.servo_error_hold for n in names):
                    msg = f"servo reports {'+'.join(names)}"
                    return ev, Event(
                        t, self.arm, HOLD, "servo_error", msg, self.joints[j], float(bits)
                    )
                msg = f"servo reports {'+'.join(names) or bits}"
                self._warn_once(ev, t, "servo_error", msg, self.joints[j])
        j = int(np.argmax(temp))
        off_c = c.temp_torque_off_c if c.temp_torque_off_c is not None else np.inf
        off = self._temp_off_t.update(t, temp >= off_c)
        hold = self._temp_hold_t.update(t, temp >= c.temp_hold_c)
        if (off >= c.temp_s).any():
            j = int(np.argmax(off))
            msg = f"{temp[j]:.0f} C >= {c.temp_torque_off_c:.0f} C for {off[j]:.2f} s"
            return ev, Event(
                t, self.arm, TORQUE_OFF, "overtemp", msg, self.joints[j], float(temp[j])
            )
        if (hold >= c.temp_s).any():
            j = int(np.argmax(hold))
            msg = f"{temp[j]:.0f} C >= {c.temp_hold_c:.0f} C for {hold[j]:.2f} s"
            return ev, Event(t, self.arm, HOLD, "overtemp", msg, self.joints[j], float(temp[j]))
        for k in np.flatnonzero(temp >= c.temp_warn_c):
            self._warn_once(ev, t, "temp", f"{temp[k]:.0f} C", self.joints[k], float(temp[k]))
        vok = np.isfinite(volt)
        out = vok & ((volt < c.voltage_min) | (volt > c.voltage_max))
        dur = self._volt_t.update(t, out)
        if (dur >= c.voltage_s).any():
            k = int(np.argmax(dur))
            msg = f"{volt[k]:.1f} V outside [{c.voltage_min}, {c.voltage_max}] for {dur[k]:.2f} s"
            return ev, Event(t, self.arm, HOLD, "voltage", msg, self.joints[k], float(volt[k]))
        margin = c.voltage_warn_margin
        near = vok & ((volt < c.voltage_min + margin) | (volt > c.voltage_max - margin))
        if near.any():
            k = int(np.flatnonzero(near)[0])
            self._warn_once(
                ev, t, "voltage", f"{volt[k]:.1f} V near the limit", None, float(volt[k])
            )
        else:
            self._clear_warn("voltage")
        dur = self._load_t.update(t, valid & (load >= self.load_hard))
        if (dur >= c.load_hard_s).any():
            k = int(np.argmax(dur))
            msg = f"|load| {load[k]:.0f} % >= {self.load_hard[k]:.0f} % for {dur[k]:.2f} s"
            return ev, Event(t, self.arm, HOLD, "overload", msg, self.joints[k], float(load[k]))
        err = np.where(valid, np.abs(target - present), 0.0)
        dur = self._track_t.update(t, err > c.tracking_error_rad)
        if (dur >= c.tracking_s).any():
            k = int(np.argmax(dur))
            msg = (
                f"{err[k]:.2f} rad behind the command for {dur[k]:.2f} s (blocked or not following)"
            )
            return ev, Event(t, self.arm, HOLD, "tracking", msg, self.joints[k], float(err[k]))
        return ev, None

    def output(
        self, t: float, target: np.ndarray, present: np.ndarray, tel: list[P.Telemetry | None]
    ) -> TickResult:
        """The goal actually written: joint limits, a load-dependent lead clamp, then a slew limit.

        A position-controlled servo pushes in proportion to its position error, so capping how far
        the goal may lead the measured position, tighter while |load| is high, caps the force it can
        build against an obstacle (or the cloth, for the gripper) without a torque mode. The slew
        limit covers everything written, so nothing downstream of the executor (a fault hold, the
        lead clamp releasing, a resume) can step a servo.
        """
        goal = np.clip(np.asarray(target, float), self.lower, self.upper)
        res = TickResult(goal=goal)
        load = np.array([abs(x.load) / 10.0 if x is not None else 0.0 for x in tel])
        loaded = load >= self.load_soft
        lead = np.where(loaded, self.lead_loaded, self.lead)
        ok = np.isfinite(present)
        clamped = ok & (np.abs(goal - present) > lead)
        goal = np.where(ok, np.clip(goal, present - lead, present + lead), goal)
        for k in np.flatnonzero(loaded & clamped):
            msg = (
                f"|load| {load[k]:.0f} %: goal held within {lead[k]:.2f} rad "
                "of the measured position"
            )
            self._warn_once(res.events, t, "load_clamp", msg, self.joints[k], float(load[k]))
        for k in np.flatnonzero(~loaded):
            self._clear_warn("load_clamp", self.joints[k])
        if self._last_goal is not None:
            dt = min(max(t - self._last_t, 0.0), 0.05)
            last = self._last_goal
            # a joint enabled on its stops walks back inside at limit_recover_speed, no jump
            out = (last < self.lower) | (last > self.upper)
            recover = np.minimum(self.vmax_out, self.cfg.limit_recover_speed)
            step = np.where(out, recover, self.vmax_out) * dt
            goal = np.clip(goal, last - step, last + step)
            goal = np.clip(goal, np.minimum(self.lower, last), np.maximum(self.upper, last))
        else:
            goal = np.clip(goal, self.lower, self.upper)
        self._last_goal, self._last_t = goal, t
        res.goal, res.lead_clamped = goal, clamped
        return res
