"""The robot process: camera threads, one control thread per arm, an inference thread, an operator.

    DISABLED --enable--> HOLDING --go--> RUNNING --pause--> HOLDING
        ^                   ^                |
        |                 resume           fault (latched; both arms hold where they are)
        |                   |                v
        +------enable------ FAULT <----------+      (over-temperature also drops that arm's torque)

    HOLDING or RUNNING --take--> TELEOP --go--> RUNNING        (leader arms configured)
                                   |
                                 pause --> HOLDING

In TELEOP the followers track the leader arms, for a demonstration or a correction in the middle
of a policy run (DAgger). The leaders' pose goes through the same executor slew limit and safety
output filter as a policy chunk, and faults work the same way. A takeover is refused while a leader
is far from its follower, and closes the remaining gap slowly before following at full speed.

Each arm has a control thread on its own bus, at rate_hz: read registers 56..70 of all six servos
in one SYNC_READ, sample the executor at the current time, run the safety checks, filter the goal
(limits, load-dependent lead clamp, slew limit) and SYNC_WRITE it. Torque is only dropped by an
over-temperature fault or an operator command, so quitting leaves the arms holding.

The inference thread runs in RUNNING only. Once per replan period (or when the previous inference
finishes, if that is later) it snapshots the newest frames and joint positions, runs the policy
and pushes the chunk stamped with the snapshot time. The executor enters it at "now", so inference
latency is skipped rather than replayed.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from dataclasses import dataclass

import numpy as np

from ..robot.arm import Arm
from ..robot.executor import Chunk, ChunkExecutor, ExecutorConfig
from ..robot.safety import HOLD, TORQUE_OFF, ArmSafety, Event
from .config import RunConfig
from .telemetry import RunLog

log = logging.getLogger("so101_smolvla_edge.runner")

DISABLED, HOLDING, RUNNING, FAULT, TELEOP = "disabled", "holding", "running", "fault", "teleop"


@dataclass
class _ArmSlot:
    """One follower arm as the runner sees it."""

    arm: Arm
    first: int  # index of the arm's first joint in the runner's joint order
    safety: ArmSafety
    torque: bool = False  # torque on, as far as the runner knows
    t_read: float = 0.0  # time.monotonic() of the last successful read
    loop_ms: float = 0.0  # last control period
    overruns: int = 0  # ticks that missed their deadline
    last_goal: np.ndarray | None = None  # rad, the last goal written to the servos
    max_temp: float = 0.0  # C, hottest servo at the last read
    min_volt: float = 99.0  # V, lowest servo supply voltage at the last read
    max_load: float = 0.0  # %, highest servo load at the last read


def _per_servo(tel: list, attr: str, scale: float = 1.0) -> np.ndarray:
    """One telemetry value per servo, NaN where a servo did not answer."""
    return np.array([getattr(x, attr) * scale if x is not None else np.nan for x in tel])


def _arm_status(name: str, s: _ArmSlot) -> str:
    """Status of one arm: torque, temperature, voltage, load, loop time, bus round trip."""
    stats = s.arm.bus.stats
    return (
        f"{name}[{'T' if s.torque else '-'} {s.max_temp:.0f}C {s.min_volt:.1f}V "
        f"load{s.max_load:.0f}% loop{s.loop_ms:.1f}ms ovr{s.overruns} rtt{stats.last_rtt_ms:.1f}ms "
        f"miss{stats.missing_replies}]"
    )


class Runner:
    """The arms, cameras, policy and leaders of one run, and the state machine above."""

    def __init__(
        self,
        cfg: RunConfig,
        arms: dict[str, Arm],
        cameras: dict,
        policy,
        runlog: RunLog,
        policy_joints: list[str],
        leaders: dict[str, Arm] | None = None,
        recorder=None,
        viz=None,
    ):
        """`policy` has .infer(images: dict, state: np.ndarray) -> (chunk, n_joints) in
        policy_joints order (the Engine), or is None for a hold-only run.

        `leaders` are read-only leader arms keyed like `arms`, with the same joint names. `recorder`
        is a control.recorder.Recorder and `viz` a control.viz.RerunViz, both optional.
        """
        self.cfg, self.cameras, self.policy, self.log = cfg, cameras, policy, runlog
        self.leaders = leaders or {}
        self.recorder = recorder
        self.viz = viz
        self.slots: dict[str, _ArmSlot] = {}
        first = 0
        lower, upper, vmax = [], [], []
        sc = cfg.safety
        for name, arm in arms.items():
            reach = (getattr(arm, "reach_lower", None), getattr(arm, "reach_upper", None))
            safety = ArmSafety(name, arm.joints, sc, arm.lower, arm.upper, *reach)
            self.slots[name] = _ArmSlot(arm, first, safety)
            first += len(arm.joints)
            lower.append(arm.lower)
            upper.append(arm.upper)
            gripper = [j.endswith("gripper") for j in arm.joints]
            vmax.append([sc.max_velocity_gripper if g else sc.max_velocity for g in gripper])
        self.joints = [j for a in arms.values() for j in a.joints]
        missing = set(policy_joints) - set(self.joints)
        if missing:
            raise ValueError(
                f"the policy drives joints the configured arms do not have: {sorted(missing)}"
            )
        self.policy_joints = policy_joints
        # chunk[:, to_exec] is in the runner's joint order, q[to_policy] in the policy's
        self.to_exec = [policy_joints.index(j) for j in self.joints]
        self.to_policy = [self.joints.index(j) for j in policy_joints]
        c = cfg.control
        self.ex = ChunkExecutor(
            ExecutorConfig(
                lower=np.concatenate(lower),
                upper=np.concatenate(upper),
                max_velocity=np.concatenate(vmax),
                interp=c.interp,
                crossfade_s=c.crossfade_s,
                max_start_jump=c.max_start_jump,
                max_first_jump=c.max_first_jump,
                min_remaining_s=c.min_remaining_s,
                max_latency_s=c.max_latency_s,
            )
        )
        self.q = np.full(len(self.joints), np.nan)
        self.q_leader = np.full(len(self.joints), np.nan)
        self.t_leader = 0.0
        self._teleop_aligned = False
        self._chunk_len = 0  # actions per chunk, known after the first inference
        self._leader_idx = {
            n: [self.joints.index(j) for j in a.joints] for n, a in self.leaders.items()
        }
        self.state = DISABLED
        self.fault_event: Event | None = None
        self._active_faults: set = set()
        self._mu = threading.RLock()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self.seq = 0
        self.infer_ms: list[float] = []
        self.chunks_ok = self.chunks_bad = 0

    # ------------------------------------------------------------------ events and state
    def event(self, level: str, code: str, msg: str, arm: str = "", **kw) -> None:
        """Record an event in events.jsonl, the viewer and the console log."""
        rec = {
            "t": time.monotonic(),
            "wall": time.time(),
            "level": level,
            "code": code,
            "arm": arm,
            "msg": msg,
            **kw,
        }
        self.log.event(rec)
        who = arm + " " if arm else ""
        if self.viz is not None:
            self.viz.event(level, f"{who}{code}: {msg}", rec["t"])
        emit = log.warning if level in ("warn", HOLD, TORQUE_OFF) else log.info
        emit("[%s] %s%s: %s", level, who, code, msg)

    def _set_state(self, new: str, why: str) -> None:
        """Change state; entering RUNNING or TELEOP starts a recording if none is open."""
        with self._mu:
            old, self.state = self.state, new
        if old != new:
            self.event("info", "state", f"{old} -> {new} ({why})")
            if self.recorder is not None and new in (RUNNING, TELEOP) and not self.recorder.active:
                d = self.recorder.begin()
                if d is not None:
                    print(
                        f"recording {d.name}: end it with 'save', 'fail' or 'discard'", flush=True
                    )

    def _hold_pose(self) -> np.ndarray:
        """The measured positions, with an arm's last goal standing in for joints not read."""
        q = self.q.copy()
        for s in self.slots.values():
            sl = slice(s.first, s.first + len(s.arm.joints))
            if s.last_goal is not None:
                q[sl] = np.where(np.isfinite(q[sl]), q[sl], s.last_goal)
        return q

    def raise_fault(self, ev: Event) -> None:
        """Latch FAULT and hold; a TORQUE_OFF event also drops that arm's torque.

        The hold pose is anchored to the measured positions once, on entry. Re-anchoring on every
        tick a condition persists would walk the arm down under gravity, since each measurement
        sags a little below its goal.
        """
        key = (ev.arm, ev.code, ev.joint, ev.level)
        with self._mu:
            first = self.state != FAULT
            if first:
                self.fault_event = ev
                self._active_faults = set()
                self.state = FAULT
                self.ex.reset_hold(self._hold_pose(), time.monotonic())
            elif key in self._active_faults and not (
                ev.level == TORQUE_OFF and self.slots[ev.arm].torque
            ):
                return
            self._active_faults.add(key)
        self.log.event({"wall": time.time(), **ev.as_dict()})
        log.error("FAULT [%s] %s %s %s: %s", ev.level, ev.arm, ev.code, ev.joint or "", ev.msg)
        if ev.level == TORQUE_OFF and self.slots[ev.arm].torque:
            s = self.slots[ev.arm]
            s.torque = False
            ok = s.arm.set_torque(False)
            result = "ok" if ok else "some writes failed"
            self.event(TORQUE_OFF, "torque_off", f"torque disabled ({result})", ev.arm)
        if first:
            print(
                f"\n*** FAULT: {ev.arm} {ev.code} {ev.joint or ''}: {ev.msg}\n"
                "*** both arms holding. Fix the cause, then 'resume' "
                "(or 'enable' if torque was dropped).",
                flush=True,
            )

    # ------------------------------------------------------------------ operator commands
    def enable(self) -> bool:
        """Turn torque on and hold the current pose; False if an arm fails its startup check.

        An arm resting outside its commandable limits is moved inside them slowly.
        """
        with self._mu:
            if self.state in (RUNNING, TELEOP):
                return True
            t = time.monotonic()
            todo = [s for s in self.slots.values() if not s.torque]
            for s in todo:
                try:
                    q, _, _ = s.arm.read()
                except Exception as exc:  # noqa: BLE001
                    msg = f"bus error: {type(exc).__name__}: {exc}"
                    self.event("warn", "enable_refused", msg, s.arm.name)
                    return False
                bad = s.safety.check_startup(q)
                if bad:
                    self.event("warn", "enable_refused", "; ".join(bad), s.arm.name)
                    print(
                        f"enable refused for {s.arm.name}:\n  "
                        + "\n  ".join(bad)
                        + "\n  Move the arm inside its limits by hand (torque is off) and retry.",
                        flush=True,
                    )
                    return False
                moving = s.safety.outside_limits(q)
                if moving:
                    msg = (
                        f"resting outside the commandable limits ({', '.join(moving)}): "
                        f"moving inside at {self.cfg.safety.limit_recover_speed:g} rad/s"
                    )
                    self.event("info", "limit_recover", msg, s.arm.name)
                    print(f"{s.arm.name}: {msg}", flush=True)
                self.q[s.first : s.first + len(q)] = q
            hold = self._hold_pose()
            for s in todo:
                sl = slice(s.first, s.first + len(s.arm.joints))
                if not s.arm.set_torque(True, hold[sl]):
                    self.event("warn", "torque_on", "some torque writes failed", s.arm.name)
                s.safety.reset(hold[sl], t)
                s.last_goal = hold[sl].copy()
                s.torque = True
            self.ex.reset_hold(hold, t)
            self.fault_event = None
            self._set_state(HOLDING, "enable")
            return True

    def go(self) -> None:
        """Hand the arms to the policy, from HOLDING or TELEOP."""
        with self._mu:
            if self.state not in (HOLDING, TELEOP):
                print(f"cannot start the policy from state '{self.state}'", flush=True)
                return
            if self.policy is None:
                print("no policy loaded (hold-only run)", flush=True)
                return
            if self.state == TELEOP:
                self.ex.hold()  # hand back: the policy's next chunk enters from rest
            self._set_state(RUNNING, "operator")

    def pause(self) -> None:
        """Stop following the policy, a `home` move or the leaders, and hold the current command."""
        with self._mu:
            self.ex.hold()
            if self.state in (RUNNING, TELEOP):
                self._set_state(HOLDING, "operator")

    def take(self) -> bool:
        """Hand the followers to the leader arms.

        From HOLDING this starts a demonstration, from RUNNING a correction. Refused while the
        leaders are not reporting or one is more than teleop.max_engage_rad from its follower.
        """
        tc = self.cfg.teleop
        with self._mu:
            if not self.leaders:
                print("no leader arms configured (add `leaders:` to the config)", flush=True)
                return False
            if self.state not in (HOLDING, RUNNING):
                print(f"cannot take over from state '{self.state}'", flush=True)
                return False
            leader_age = time.monotonic() - self.t_leader
            if not np.isfinite(self.q_leader).all() or leader_age > tc.max_leader_age_s:
                print("takeover refused: the leader arms are not reporting", flush=True)
                return False
            gap = np.abs(self.q_leader - self.ex.last_command())
            j = int(np.argmax(gap))
            if gap[j] > tc.max_engage_rad:
                print(
                    f"takeover refused: leader {self.joints[j]} is {gap[j]:.2f} rad "
                    f"from its follower (limit {tc.max_engage_rad:.2f}). "
                    "Bring the leaders close to the followers' pose and retry.",
                    flush=True,
                )
                self.event("warn", "takeover_refused", f"{self.joints[j]} gap {gap[j]:.2f} rad")
                return False
            self._teleop_aligned = bool(gap[j] < tc.align_rad)
            self.ex.follow(self.q_leader, None if self._teleop_aligned else tc.engage_speed)
            self._set_state(TELEOP, f"operator takeover (gap {gap[j]:.2f} rad)")
            return True

    def take_after(self, delay_s: float) -> bool:
        """`take` after a countdown of `delay_s` (at most 30 s), printing the gap every second.

        An operator working alone types the command, then needs both hands to bring the leaders to
        the followers' pose.
        """
        end = time.monotonic() + min(max(delay_s, 0.0), 30.0)
        while (left := end - time.monotonic()) > 0:
            with self._mu:
                if self.leaders:
                    gap = np.abs(self.q_leader - self.ex.last_command())
                else:
                    gap = np.array([np.nan])
            if np.isfinite(gap).all():
                where = f"leader gap {np.max(gap):.2f} rad ({self.joints[int(np.argmax(gap))]})"
            else:
                where = "leaders: no data"
            print(f"  taking over in {left:.0f} s | {where}", flush=True)
            time.sleep(min(1.0, left))
        ok = self.take()
        outcome = "the leaders have the arms" if ok else "NOT taken over: still " + self.state
        print("\a" + outcome, flush=True)
        return ok

    def resume(self) -> None:
        """Clear a fault and hold where the arms are; needs torque on everywhere."""
        with self._mu:
            if self.state != FAULT:
                return
            if not all(s.torque for s in self.slots.values()):
                print("an arm has torque off: use 'enable'", flush=True)
                return
            t = time.monotonic()
            hold = self._hold_pose()
            for s in self.slots.values():
                s.safety.reset(hold[s.first : s.first + len(s.arm.joints)], t)
            self.ex.reset_hold(hold, t)
            self.fault_event = None
            self._set_state(HOLDING, "operator resume")

    def home(self) -> None:
        """From HOLDING, move to control.home_pose at control.home_speed."""
        pose = self.cfg.control.home_pose
        with self._mu:
            if self.state != HOLDING or not pose:
                print("home needs state 'holding' and control.home_pose in the config", flush=True)
                return
            now = time.monotonic()
            q0 = self._hold_pose()
            q1 = q0.copy()
            for j, v in pose.items():
                q1[self.joints.index(j)] = v
            T = max(float(np.max(np.abs(q1 - q0))) / self.cfg.control.home_speed, 0.1)
            ok, why = self.ex.push(Chunk(q=np.stack([q0, q1]), t0=now, dt=T, seq=-1), now)
            self.event("info", "home", f"{'moving' if ok else 'refused'} ({why}), {T:.1f} s")

    def torque_off(self) -> None:
        """Drop torque on every arm; the arms fall unless supported."""
        with self._mu:
            for s in self.slots.values():
                s.torque = False
                s.arm.set_torque(False)
            self._set_state(DISABLED, "operator torque off")

    # ------------------------------------------------------------------ threads
    def _arm_loop(self, name: str) -> None:
        """One arm's control loop: read, sample the executor, check safety, write the goal."""
        s = self.slots[name]
        n = len(s.arm.joints)
        sl = slice(s.first, s.first + n)
        prio = self.cfg.control.rt_priority
        if prio > 0:
            try:
                os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(prio))
            except (PermissionError, AttributeError, OSError) as exc:
                self.event("warn", "rt_priority", f"SCHED_FIFO not granted ({exc})", name)
        period = 1.0 / self.cfg.control.rate_hz
        nxt = time.monotonic()
        last = nxt
        k = 0
        nan = np.full(n, np.nan)
        while not self._stop.is_set():
            t0 = time.monotonic()
            s.loop_ms, last = (t0 - last) * 1e3, t0
            try:
                q, tel, errs = s.arm.read()
            except Exception as exc:  # noqa: BLE001 (USB unplugged, adapter reset, ...)
                self._bus_lost(s, exc)
                nxt = time.monotonic() + period
                continue
            t = time.monotonic()
            with self._mu:
                self.q[sl] = q
                s.t_read = t
                state = self.state
            target = goal = nan
            clamp = np.zeros(n)
            if s.torque and state != DISABLED:
                target = self.ex.sample(t, s.first, n)
                evs, fault = s.safety.check(t, tel, q, target, errs, s.arm.ids)
                for e in evs:
                    self.log.event({"wall": time.time(), **e.as_dict()})
                    log.warning("[warn] %s %s %s: %s", e.arm, e.code, e.joint or "", e.msg)
                if fault is not None:
                    self.raise_fault(fault)
                    target = self.ex.sample(t, s.first, n)
                if s.torque:
                    res = s.safety.output(t, target, q, tel)
                    for e in res.events:
                        self.log.event({"wall": time.time(), **e.as_dict()})
                    goal = res.goal
                    clamp = res.lead_clamped.astype(float)
                    try:
                        s.arm.write_goal(goal)
                    except Exception as exc:  # noqa: BLE001
                        self._bus_lost(s, exc)
                        nxt = time.monotonic() + period
                        continue
                    s.last_goal = goal
            valid = [x for x in tel if x is not None]
            if valid:
                s.max_temp = max(x.temperature for x in valid)
                s.min_volt = min(x.voltage for x in valid)
                s.max_load = max(abs(x.load) / 10 for x in valid)
            if k % self.cfg.log_every == 0:
                telemetry = [
                    _per_servo(tel, "velocity"),
                    _per_servo(tel, "load", 0.1),  # %
                    _per_servo(tel, "current"),
                    _per_servo(tel, "temperature"),
                    _per_servo(tel, "voltage"),
                ]
                cols = np.stack([target, goal, q, *telemetry, clamp])
                self.log.tick(t, name, s.loop_ms, s.arm.bus.stats.last_rtt_ms, state, cols)
            k += 1
            nxt += period
            sleep = nxt - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)
            else:
                s.overruns += 1
                if sleep < -5 * period:
                    nxt = time.monotonic()  # fell far behind: re-anchor instead of bursting

    def _leader_loop(self) -> None:
        """Read the leader arms and, in TELEOP, make their pose the executor's target.

        TELEOP falls back to HOLDING when the leaders stop reporting.
        """
        tc = self.cfg.teleop
        period = 1.0 / tc.rate_hz
        fails = 0
        while not self._stop.is_set():
            t0 = time.monotonic()
            q = np.full(len(self.joints), np.nan)
            try:
                for name, arm in self.leaders.items():
                    ql, _, _ = arm.read()
                    q[self._leader_idx[name]] = ql
                fails = 0
            except Exception as exc:  # noqa: BLE001 (unplugged leader, adapter reset)
                fails += 1
                if fails in (1, 50) or fails % 1000 == 0:
                    self.event(
                        "warn", "leader_read", f"{type(exc).__name__}: {exc} ({fails} in a row)"
                    )
                q[:] = np.nan
                time.sleep(0.05)
            now = time.monotonic()
            with self._mu:
                if np.isfinite(q).all():
                    self.q_leader, self.t_leader = q, now
                if self.state == TELEOP:
                    if now - self.t_leader > tc.max_leader_age_s:
                        self.ex.hold()
                        self._set_state(HOLDING, "leader arms stopped reporting")
                        print("\n*** leader arms stopped reporting: holding", flush=True)
                    else:
                        if not self._teleop_aligned:
                            gap = np.max(np.abs(self.q_leader - self.ex.last_command()))
                            if gap < tc.align_rad:
                                self._teleop_aligned = True
                                msg = "followers aligned with the leaders: full speed"
                                self.event("info", "teleop", msg)
                        speed = None if self._teleop_aligned else tc.engage_speed
                        self.ex.follow(self.q_leader, speed)
            sleep = period - (time.monotonic() - t0)
            if sleep > 0:
                time.sleep(sleep)

    # ------------------------------------------------------------------ recording
    def _record_sample(self):
        """(state, goal, mode, leader, seq) for the recorder, in the runner's joint order."""
        with self._mu:
            goal = np.full(len(self.joints), np.nan)
            for s in self.slots.values():
                if s.last_goal is not None:
                    goal[s.first : s.first + len(s.arm.joints)] = s.last_goal
            # until the followers catch up, the goal is the ramp toward the leaders, not the
            # operator's motion: mark those frames so they never become action labels
            mode = "engaging" if self.state == TELEOP and not self._teleop_aligned else self.state
            return self.q.copy(), goal, mode, self.q_leader.copy(), self.seq

    def end_episode(self, label: str, note: str = "") -> None:
        """End the episode: hold, then save the recording as `label`, or delete it for "discard".

        The next `go` or `take` starts a new one.
        """
        self.pause()
        if self.recorder is None or not self.recorder.active:
            print("no episode is being recorded", flush=True)
            return
        if label == "discard":
            self.recorder.discard()
            print("episode discarded", flush=True)
        else:
            d = self.recorder.end(label, note)
            name = d.name if d else "(empty)"
            print(f"episode saved: {name} [{label}]" + (f" {note}" if note else ""), flush=True)
        self.event("info", "episode", f"{label} {note}".strip())

    def _bus_lost(self, s: _ArmSlot, exc: Exception) -> None:
        """Fault on a port that failed outright, then try to reopen it after a second.

        Both arms hold and the servos keep their last goal. The arm loop calls this again on the
        next failure, so the port is retried once a second. Torque is re-read after a reconnect,
        since servos that browned out come back with torque off.
        """
        msg = f"port error: {type(exc).__name__}: {exc}"
        self.raise_fault(Event(time.monotonic(), s.arm.name, HOLD, "bus", msg))
        time.sleep(1.0)
        if self._stop.is_set():
            return
        try:
            s.arm.bus.close()
            s.arm.bus.open()
            torque = s.arm.torque_state()
        except Exception as exc2:  # noqa: BLE001
            msg = f"still failing: {type(exc2).__name__}: {exc2}"
            self.event("warn", "reconnect", msg, s.arm.name)
            return
        if s.torque and not all(torque):
            s.torque = False
            msg = (
                f"port back, but torque is off on some servos ({torque}): "
                "'enable' after checking the arm"
            )
            self.event(TORQUE_OFF, "reconnect", msg, s.arm.name)
        else:
            self.event("info", "reconnect", "port back; still faulted until 'resume'", s.arm.name)

    def _replan_period(self) -> float:
        """Seconds between inference starts: `replan_actions` actions, else `replan_period_s`.

        The action count is capped so the next chunk is ready before this one ends.
        """
        ic = self.cfg.inference
        if not ic.replan_actions:
            return ic.replan_period_s
        period = ic.replan_actions * ic.action_dt
        if self._chunk_len:
            period = min(
                period, max(self._chunk_len * ic.action_dt - ic.replan_margin_s, ic.action_dt)
            )
        return period

    def _infer_loop(self) -> None:
        """While RUNNING, snapshot frames and joints, run the policy and push each chunk.

        An inference is skipped when a frame or the joint state is too old, and an inference error
        pauses the policy.
        """
        ic = self.cfg.inference
        last_start = 0.0
        rejects_in_row = skips_in_row = 0
        while not self._stop.is_set():
            if self.state != RUNNING:
                time.sleep(0.01)
                continue
            wait = last_start + self._replan_period() - time.monotonic()
            if wait > 0:
                time.sleep(min(wait, 0.01))
                continue
            t_obs = time.monotonic()
            last_start = t_obs
            images, stale = {}, []
            for key, cam in self.cameras.items():
                img, t_img = cam.latest()
                if img is None or t_obs - t_img > ic.max_frame_age_s:
                    stale.append(key)
                images[key] = img
            with self._mu:
                q = self.q.copy()
                ages = [t_obs - s.t_read for s in self.slots.values()]
            if stale or not np.isfinite(q).all() or max(ages) > ic.max_state_age_s:
                if stale:
                    why = f"stale cameras {stale}"
                else:
                    why = f"joint state missing/old (age {max(ages) * 1e3:.0f} ms)"
                skips_in_row += 1
                self.log.chunk({"t_obs": t_obs, "skipped": why})
                if skips_in_row in (1, 10, 100) or skips_in_row % 1000 == 0:
                    self.event("warn", "skip_inference", f"{why} ({skips_in_row} in a row)")
                time.sleep(0.05)
                continue
            skips_in_row = 0
            state = q[self.to_policy].astype(np.float32)
            t_a = time.monotonic()
            try:
                chunk = self.policy.infer(images, state)
            except Exception as exc:  # noqa: BLE001 (CUDA error, out of memory, ...)
                self.event(HOLD, "inference_error", f"{type(exc).__name__}: {exc}; policy paused")
                self.pause()
                continue
            t_b = time.monotonic()
            self.infer_ms.append((t_b - t_a) * 1e3)
            del self.infer_ms[:-200]
            self.seq += 1
            rec = {
                "seq": self.seq,
                "t_obs": t_obs,
                "infer_ms": (t_b - t_a) * 1e3,
                "latency_s": t_b - t_obs,
            }
            if self.cfg.log_inputs:
                self.log.inputs(self.seq, images)
            if self.state != RUNNING:
                rec["dropped"] = f"state became {self.state} during inference"
                self.log.chunk(rec)
                continue
            self._chunk_len = int(chunk.shape[0])
            q_exec = chunk[:, self.to_exec].astype(np.float64)
            new = Chunk(q=q_exec, t0=t_obs, dt=ic.action_dt, seq=self.seq)
            ok, why = self.ex.push(new, time.monotonic())
            rec.update(
                accepted=ok,
                reason=why,
                start_jump=float(np.max(np.abs(q_exec[0] - q))),
                state=np.round(q, 4).tolist(),
                first=np.round(q_exec[0], 4).tolist(),
                last=np.round(q_exec[-1], 4).tolist(),
            )
            self.log.chunk(rec)
            if self.viz is not None:
                self.viz.inference(self.seq, t_obs, images, q, q_exec, ic.action_dt, ok, why)
            if ok:
                self.chunks_ok += 1
                rejects_in_row = 0
            else:
                self.chunks_bad += 1
                rejects_in_row += 1
                if rejects_in_row in (1, 5, 20):
                    self.event("warn", "chunk_rejected", f"{why} ({rejects_in_row} in a row)")

    def status_line(self) -> str:
        """One line for the console: state, inference time, chunks, leader gap, recording, arms."""
        im = sorted(self.infer_ms)
        p50 = f"{im[len(im) // 2]:.0f}" if im else "-"
        st = self.ex.status(time.monotonic())
        arms = " ".join(_arm_status(n, s) for n, s in self.slots.items())
        rec = ""
        if self.leaders:  # a takeover needs the leaders close to the followers
            gap = np.abs(self.q_leader - self.ex.last_command())
            if np.isfinite(gap).all():
                rec = f" | leader gap {np.max(gap):.2f} rad ({self.joints[int(np.argmax(gap))]})"
            else:
                rec = " | leaders: no data"
        if self.recorder is not None:
            rec += f" | rec {'ON' if self.recorder.active else 'off'} ep{self.recorder.episodes}"
        return (
            f"{self.state:8s} infer p50 {p50} ms | "
            f"chunks ok {self.chunks_ok} rej {self.chunks_bad} left {st.remaining_s:.2f}s"
            f"{rec} | {arms}"
        )

    def start(self) -> None:
        """Start the cameras and the arm, inference, leader and recorder threads."""
        for cam in self.cameras.values():
            cam.start()
        for name in self.slots:
            th = threading.Thread(
                target=self._arm_loop, args=(name,), daemon=True, name=f"arm-{name}"
            )
            th.start()
            self._threads.append(th)
        if self.policy is not None:
            th = threading.Thread(target=self._infer_loop, daemon=True, name="infer")
            th.start()
            self._threads.append(th)
        if self.leaders:
            th = threading.Thread(target=self._leader_loop, daemon=True, name="leaders")
            th.start()
            self._threads.append(th)
        if self.recorder is not None:
            self.recorder.start(self._record_sample)
        leaders = f"; leaders {list(self.leaders)}" if self.leaders else ""
        self.event("info", "start", f"joints {self.joints}{leaders}")

    def stop(self) -> None:
        """Hold, let the arm threads write the hold a few more times, and stop. Torque stays."""
        with self._mu:
            if self.state in (RUNNING, TELEOP):
                self.ex.hold()
                self._set_state(HOLDING, "quit")
        time.sleep(3.0 / self.cfg.control.rate_hz)
        if self.recorder is not None:
            self.recorder.close()
        self._stop.set()
        for th in self._threads:
            th.join(timeout=2)
        for cam in self.cameras.values():
            cam.stop()
        for s in self.slots.values():
            s.arm.close()
        for a in self.leaders.values():
            a.close()
        torque = ", ".join(f"{n}={'on' if s.torque else 'off'}" for n, s in self.slots.items())
        self.event("info", "stop", "torque left " + torque)

    # ------------------------------------------------------------------ console
    HELP = (
        "commands: enable | go | pause | resume | home | status"
        " | off (torque off: the arms DROP) | quit\n"
        "          take [s] (leader arms drive the followers,"
        " after s seconds if given; 'go' hands back)\n"
        "          save [note] | fail [note] | discard   (end the episode being recorded)\n"
        "          (Enter alone prints status)"
    )

    def console(self, stdin=None) -> None:
        """Run operator commands from `stdin` until quit, printing the status every 2 s."""
        stdin = stdin or sys.stdin
        print(self.HELP, flush=True)
        stop_status = threading.Event()

        def ticker():
            while not stop_status.wait(2.0):
                print(self.status_line(), flush=True)

        threading.Thread(target=ticker, daemon=True).start()
        try:
            for line in stdin:
                cmd = line.strip().lower()
                self.event("info", "operator", cmd or "status")
                if cmd in ("quit", "q", "exit"):
                    break
                try:
                    self.command(cmd, stdin)
                except Exception as exc:  # noqa: BLE001 (a failed command must not cost the console)
                    self.event("warn", "command_failed", f"{cmd}: {type(exc).__name__}: {exc}")
        except KeyboardInterrupt:
            pass
        finally:
            stop_status.set()

    def command(self, cmd: str, stdin=None) -> None:
        """Run one operator command, from the console or `run --commands`."""
        if cmd in ("", "status", "s"):
            print(self.status_line(), flush=True)
        elif cmd in ("enable", "e"):
            self.enable()
        elif cmd in ("go", "g"):
            self.go()
        elif cmd in ("pause", "p", "hold", "h"):
            self.pause()
        elif cmd in ("resume", "r"):
            self.resume()
        elif cmd == "home":
            self.home()
        elif cmd.split(" ", 1)[0] in ("take", "t"):
            arg = cmd.partition(" ")[2].strip()
            try:
                delay = float(arg) if arg else 0.0
            except ValueError:
                print("usage: take [seconds to wait first]", flush=True)
                return
            if delay > 0:
                self.take_after(delay)
            else:
                self.take()
        elif cmd.split(" ", 1)[0] in ("save", "ok", "fail", "discard"):
            word, _, note = cmd.partition(" ")
            self.end_episode({"save": "success", "ok": "success"}.get(word, word), note.strip())
        elif cmd == "off":
            print(
                "Torque off: the arms will DROP. Support them, then type YES: ", end="", flush=True
            )
            if stdin is not None and stdin.readline().strip() == "YES":
                self.torque_off()
            else:
                print("cancelled", flush=True)
        else:
            print(self.HELP, flush=True)
