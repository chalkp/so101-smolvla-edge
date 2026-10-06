"""Turns a stream of action chunks into a continuous, rate-limited joint target.

Action i of a chunk targets t0 + i * dt, t0 being the capture time of its observation. Chunks are
sampled at the current time rather than played from their start, so inference latency is skipped
and the two arms, sampled by different threads, stay consistent. Past the end of a chunk the last
action is held, and no timeout ever disables torque: torque-off drops an SO-101 arm.
Ported from ros2_ws/src/robot_server/src/chunk_executor.cpp (same rules, same defaults).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np


@dataclass
class Chunk:
    q: np.ndarray  # (horizon, n_joints) rad, executor joint order
    t0: float  # observation capture time, time.monotonic() clock
    dt: float = 1.0 / 30.0  # s between actions
    seq: int = 0


@dataclass
class ExecutorConfig:
    lower: np.ndarray  # rad, per joint
    upper: np.ndarray  # rad, per joint
    max_velocity: np.ndarray  # rad/s per joint, the slew limit
    interp: str = "cubic"  # cubic (Catmull-Rom) | linear
    crossfade_s: float = 0.10  # s, blend from the previous chunk after a switch

    # Reject a chunk that starts this far from where the active one is heading. A plausibility
    # check only, the slew limit does the smoothing. A new chunk takes over near the end of the old
    # one, from a fresh look many actions later, so the two can differ by over a radian, and a false
    # reject costs a whole chunk of standing still.
    max_start_jump: float = 2.0  # rad
    # The same check for the first chunk after a hold, looser on purpose: dataset episodes start
    # with the leader arms already over the table while the followers rest, so the first action is
    # far from the state. The slew limit turns that step into a move at the trained speed.
    max_first_jump: float = 2.0  # rad

    min_remaining_s: float = 0.05  # s; reject a chunk with less than this left on arrival
    max_latency_s: float = 2.0  # s; reject a chunk whose observation is older than this
    max_future_s: float = 0.05  # s; tolerated clock skew for observation stamps in the future
    max_slew_dt: float = 0.05  # s; largest time step one slew-limited sample may cover

    def __post_init__(self) -> None:
        self.lower = np.asarray(self.lower, dtype=np.float64)
        self.upper = np.asarray(self.upper, dtype=np.float64)
        self.max_velocity = np.broadcast_to(
            np.asarray(self.max_velocity, dtype=np.float64), self.lower.shape
        ).copy()
        same = self.lower.shape == self.upper.shape == self.max_velocity.shape
        if not same or self.lower.ndim != 1:
            raise ValueError("lower/upper/max_velocity must be 1-D and the same length")


@dataclass
class ExecutorStatus:
    executing: bool = False  # the active chunk has time left
    has_target: bool = False  # a hold pose was set (torque has been enabled)
    active_seq: int = 0
    remaining_s: float = 0.0  # s left in the active chunk
    accepted: int = 0
    rejected: int = 0
    last_reject: str = ""  # reason for the last rejection


def chunk_value(c: Chunk, t: float, cubic: bool) -> np.ndarray:
    """Chunk `c` at time t, Catmull-Rom or linear between actions, held at both ends."""
    H = c.q.shape[0]
    if H == 1:
        return c.q[0]
    s = (t - c.t0) / c.dt
    if s <= 0.0:
        return c.q[0]
    if s >= H - 1:
        return c.q[H - 1]
    i = int(s)
    u = s - i
    p0, p1 = c.q[i], c.q[i + 1]
    if not cubic:
        return p0 + u * (p1 - p0)
    m0 = (p1 - p0) if i == 0 else 0.5 * (p1 - c.q[i - 1])
    m1 = (p1 - p0) if i + 2 >= H else 0.5 * (c.q[i + 2] - p0)
    u2, u3 = u * u, u * u * u
    h00 = 2 * u3 - 3 * u2 + 1  # cubic Hermite basis
    h10 = u3 - 2 * u2 + u
    h01 = -2 * u3 + 3 * u2
    h11 = u3 - u2
    return h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1


class ChunkExecutor:
    """Thread-safe. Chunks are pushed from the inference thread and sampled per joint group."""

    def __init__(self, cfg: ExecutorConfig):
        self.cfg = cfg
        n = cfg.lower.shape[0]
        self.n = n  # joints
        self._mu = threading.Lock()
        self._active: Chunk | None = None
        self._previous: Chunk | None = None  # crossfaded out after a switch
        self._switch_t = 0.0  # when the active chunk took over
        self._last_out = np.zeros(n)  # rad, last command per joint
        self._last_t = np.zeros(n)  # time of the last sample per joint
        self._initialised = False  # a hold pose was set
        self._follow: np.ndarray | None = None  # teleoperation: track this pose instead of a chunk
        self._follow_vmax: float | None = None  # rad/s, extra speed cap while following
        self.accepted = 0
        self.rejected = 0
        self.last_reject = ""

    def _raw(self, t: float) -> np.ndarray:
        """Target at t before the slew limit: followed pose, crossfaded chunk, or hold."""
        cubic = self.cfg.interp == "cubic"
        if self._follow is not None:
            return self._follow.copy()
        if self._active is None:
            return self._last_out.copy()
        v_new = chunk_value(self._active, t, cubic)
        fade = self.cfg.crossfade_s
        if self._previous is not None and fade > 0 and t < self._switch_t + fade:
            w = min(max((t - self._switch_t) / fade, 0.0), 1.0)
            return (1.0 - w) * chunk_value(self._previous, t, cubic) + w * v_new
        return v_new

    def reset_hold(self, q: np.ndarray, t: float) -> None:
        """Drop all chunks and hold `q` (measured positions when torque is enabled)."""
        q = np.asarray(q, dtype=np.float64)
        if q.shape != (self.n,):
            raise ValueError("reset_hold: wrong joint count")
        with self._mu:
            self._active = self._previous = self._follow = None
            self._last_out = np.clip(q, self.cfg.lower, self.cfg.upper)
            self._last_t[:] = t
            self._initialised = True

    def hold(self) -> None:
        """Drop all chunks (and any followed pose) and hold the current command."""
        with self._mu:
            self._active = self._previous = self._follow = None

    def follow(self, q: np.ndarray, max_speed: float | None = None) -> bool:
        """Track `q`, a leader arm's pose, instead of a chunk.

        The command still moves toward it under the slew limit, further capped by `max_speed`
        (rad/s) while a takeover closes its initial gap. Returns False and changes nothing if `q`
        is malformed or no hold pose is set yet.
        """
        q = np.asarray(q, dtype=np.float64)
        if q.shape != (self.n,) or not np.isfinite(q).all():
            return False
        with self._mu:
            if not self._initialised:
                return False
            self._active = self._previous = None
            self._follow = np.clip(q, self.cfg.lower, self.cfg.upper)
            self._follow_vmax = max_speed
            return True

    def last_command(self) -> np.ndarray:
        """The command most recently handed to the arms (slew-limited)."""
        with self._mu:
            return self._last_out.copy()

    def push(self, c: Chunk, t_now: float) -> tuple[bool, str]:
        """Offer a chunk at time t_now; returns (accepted, reason).

        Rejects malformed chunks, observations too old or from the future, chunks exhausted on
        arrival, and chunks that start too far from where the command is heading.
        """
        with self._mu:

            def reject(why: str) -> tuple[bool, str]:
                self.rejected += 1
                self.last_reject = why
                return False, why

            q = np.asarray(c.q, dtype=np.float64)
            if q.ndim != 2 or q.shape[0] == 0 or q.shape[1] != self.n or not c.dt > 0:
                return reject("bad shape")
            if not np.isfinite(q).all():
                return reject("bad shape: non-finite value")
            c = Chunk(q=q, t0=c.t0, dt=c.dt, seq=c.seq)
            latency = t_now - c.t0
            if latency < -self.cfg.max_future_s:
                return reject(f"observation stamp {-latency:.3f} s in the future (clock mismatch?)")
            if latency > self.cfg.max_latency_s:
                return reject(f"observation too old: {latency:.3f} s")
            t_end = c.t0 + (q.shape[0] - 1) * c.dt
            if q.shape[0] > 1 and t_end - t_now < self.cfg.min_remaining_s:
                return reject(f"chunk exhausted on arrival: {t_end - t_now:.3f} s left")
            if not self._initialised:
                return reject("no hold target yet (torque never enabled)")
            # compare with where the command is heading, not the slew-limited command that trails it
            cubic = self.cfg.interp == "cubic"
            start = np.clip(chunk_value(c, t_now, cubic), self.cfg.lower, self.cfg.upper)
            heading = np.clip(self._raw(t_now), self.cfg.lower, self.cfg.upper)
            jump = np.abs(start - heading)
            j = int(np.argmax(jump))
            # The first chunk after a hold, or after the active one ran out (slow inference, a
            # stalled camera), enters from rest and gets the looser limit.
            a = self._active
            live = a is not None and t_now < a.t0 + (a.q.shape[0] - 1) * a.dt
            limit = self.cfg.max_start_jump if live else self.cfg.max_first_jump
            if jump[j] > limit:
                return reject(
                    f"chunk starts too far from the current command: joint {j} jumps "
                    f"{jump[j]:.3f} rad (limit {limit:.2f}{'' if live else ', entering from rest'})"
                )
            if self._active is not None:
                self._previous = self._active
            else:
                self._previous = Chunk(q=self._last_out[None].copy(), t0=t_now)  # the hold pose
            self._switch_t = t_now
            self._active = c
            self._follow = None
            self.accepted += 1
            return True, "accepted"

    def sample(self, t: float, first: int = 0, count: int | None = None) -> np.ndarray:
        """Command for joints [first, first + count) at time t.

        The raw target is slew-limited to max_velocity and clamped to the joint limits, so no
        discontinuity reaches the servos. Stateful: call it from one thread per joint group.
        """
        count = self.n - first if count is None else count
        sl = slice(first, first + count)
        with self._mu:
            if self._previous is not None and t >= self._switch_t + self.cfg.crossfade_s:
                self._previous = None
            raw = self._raw(t)[sl]
            dt = np.clip(t - self._last_t[sl], 0.0, self.cfg.max_slew_dt)
            vmax = self.cfg.max_velocity[sl]
            if self._follow is not None and self._follow_vmax is not None:
                vmax = np.minimum(vmax, self._follow_vmax)
            step = vmax * dt
            last = self._last_out[sl]
            slewed = np.clip(raw, last - step, last + step)
            v = np.clip(slewed, self.cfg.lower[sl], self.cfg.upper[sl])
            self._last_out[sl] = v
            self._last_t[sl] = t
            return v.copy()

    def status(self, t: float) -> ExecutorStatus:
        """Snapshot, with the active chunk's remaining time measured at t."""
        with self._mu:
            s = ExecutorStatus(
                has_target=self._initialised,
                accepted=self.accepted,
                rejected=self.rejected,
                last_reject=self.last_reject,
            )
            if self._active is not None:
                a = self._active
                s.active_seq = a.seq
                s.remaining_s = max(0.0, a.t0 + (a.q.shape[0] - 1) * a.dt - t)
                s.executing = s.remaining_s > 0.0
            return s
