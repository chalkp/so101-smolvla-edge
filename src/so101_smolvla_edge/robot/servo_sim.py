"""Simulated STS3215 servos behind a pseudo-terminal, for tests and `run --sim`.

The servos answer PING, READ, WRITE, SYNC_READ and SYNC_WRITE on a pty, so the real `Bus` runs end
to end. With torque on, each one moves toward its goal at a capped speed and reports a load
proportional to the remaining error. Tests can inject faults: silence, heat, load, voltage.
"""

from __future__ import annotations

import os
import pty
import select
import threading
import time
import tty

from . import protocol as P


class ServoSim:
    """Servos `ids` on one simulated bus; open `path` with a `Bus` to talk to them."""

    def __init__(
        self,
        ids: list[int],
        position: int = 2048,
        speed_ticks_s: float = 3000.0,
        load_per_tick: float = 2.0,
        echo: bool = False,
    ):
        self.master, slave = pty.openpty()  # master fd is -1 after unplug()
        tty.setraw(self.master)
        tty.setraw(slave)
        self._slave_keepalive = slave
        self.path = os.ttyname(slave)
        self.echo = echo  # echo every byte received, like some USB adapters
        self.speed = speed_ticks_s  # ticks/s
        self.load_per_tick = load_per_tick  # 0.1 % units per tick of position error
        self.lock = threading.Lock()  # guards regs and positions against the sim thread
        self.regs: dict[int, bytearray] = {}  # servo id -> control table
        self.silent: set[int] = set()  # servos that do not answer
        self.extra_load: dict[int, int] = {}  # injected load, 0.1 % units
        for sid in ids:
            r = bytearray(256)
            r[P.MODEL_NUMBER.addr : P.MODEL_NUMBER.addr + 2] = P.le(P.MODEL_STS3215, 2)
            r[P.ID.addr] = sid
            r[P.RETURN_DELAY_TIME.addr] = 250
            r[P.MAX_POSITION_LIMIT.addr : P.MAX_POSITION_LIMIT.addr + 2] = P.le(4095, 2)
            r[P.PRESENT_POSITION.addr : P.PRESENT_POSITION.addr + 2] = P.le(position, 2)
            r[P.GOAL_POSITION.addr : P.GOAL_POSITION.addr + 2] = P.le(position, 2)
            r[P.PRESENT_VOLTAGE.addr] = 120
            r[P.PRESENT_TEMPERATURE.addr] = 30
            self.regs[sid] = r
        self._pos = {sid: float(position) for sid in ids}  # ticks, unrounded
        self._stop = False
        self._t = time.monotonic()  # time of the last physics step
        self._thread = threading.Thread(target=self._run, daemon=True, name="servo-sim")
        self._thread.start()

    # ------------------------------------------------------------------ test hooks
    def set(self, sid: int, reg: P.Reg, value: int) -> None:
        with self.lock:
            self.regs[sid][reg.addr : reg.addr + reg.size] = P.le(value, reg.size)

    def get(self, sid: int, reg: P.Reg) -> int:
        with self.lock:
            b = self.regs[sid]
            return b[reg.addr] if reg.size == 1 else P.u16(b, reg.addr)

    def set_position(self, sid: int, ticks: int) -> None:
        with self.lock:
            self._pos[sid] = float(ticks)
            self.regs[sid][P.PRESENT_POSITION.addr : P.PRESENT_POSITION.addr + 2] = P.le(ticks, 2)

    def unplug(self) -> None:
        """Stop answering and close the master: port reads fail with EIO, like a pulled cable."""
        self._stop = True
        self._thread.join(timeout=1)
        if self.master >= 0:
            os.close(self.master)
            self.master = -1

    def close(self) -> None:
        if self.master < 0:
            os.close(self._slave_keepalive)
            return
        self._stop = True
        self._thread.join(timeout=1)
        os.close(self._slave_keepalive)
        os.close(self.master)

    # ------------------------------------------------------------------ physics
    def _step(self) -> None:
        """Advance every servo's position, velocity, load, current and Moving flag to now."""
        now = time.monotonic()
        dt, self._t = now - self._t, now
        for sid, r in self.regs.items():
            goal = P.u16(r, P.GOAL_POSITION.addr)
            pos = self._pos[sid]
            err = goal - pos
            if r[P.TORQUE_ENABLE.addr]:
                step = max(-self.speed * dt, min(self.speed * dt, err))
                pos += step
                vel = step / dt if dt > 0 else 0.0
            else:
                vel = 0.0
            self._pos[sid] = pos
            p = int(round(pos))
            r[P.PRESENT_POSITION.addr : P.PRESENT_POSITION.addr + 2] = P.le(max(0, min(4095, p)), 2)
            r[P.PRESENT_VELOCITY.addr : P.PRESENT_VELOCITY.addr + 2] = P.le(
                P.encode_sign_magnitude(int(max(-32767, min(32767, vel))), P.VELOCITY_SIGN_BIT), 2
            )
            load = int(self.load_per_tick * (goal - pos)) if r[P.TORQUE_ENABLE.addr] else 0
            load = max(-1000, min(1000, load + self.extra_load.get(sid, 0)))
            r[P.PRESENT_LOAD.addr : P.PRESENT_LOAD.addr + 2] = P.le(
                P.encode_sign_magnitude(load, P.LOAD_SIGN_BIT), 2
            )
            cur = int(abs(load) * 2)
            r[P.PRESENT_CURRENT.addr : P.PRESENT_CURRENT.addr + 2] = P.le(cur, 2)
            r[P.MOVING.addr] = int(abs(err) > 1)

    def _reply(self, sid: int, params: bytes = b"") -> bytes:
        """Status packet with no error bits."""
        body = bytes([sid, len(params) + 2, 0]) + params
        return b"\xff\xff" + body + bytes([P.checksum(body)])

    def _handle(self, pkt: bytes) -> bytes:
        """Execute one instruction packet; return the replies to send, possibly none."""
        sid, instr, params = pkt[2], pkt[4], pkt[5:-1]
        out = bytearray()
        if instr == P.SYNC_READ:
            addr, n = params[0], params[1]
            for s in params[2:]:
                if s in self.regs and s not in self.silent:
                    out += self._reply(s, bytes(self.regs[s][addr : addr + n]))
        elif instr == P.SYNC_WRITE:
            addr, n = params[0], params[1]
            body = params[2:]
            for k in range(0, len(body), n + 1):
                s = body[k]
                if s in self.regs:
                    self.regs[s][addr : addr + n] = body[k + 1 : k + 1 + n]
        elif sid in self.regs and sid not in self.silent:
            r = self.regs[sid]
            if instr == P.PING:
                out += self._reply(sid)
            elif instr == P.READ:
                out += self._reply(sid, bytes(r[params[0] : params[0] + params[1]]))
            elif instr == P.WRITE:
                addr, data = params[0], params[1:]
                r[addr : addr + len(data)] = data
                out += self._reply(sid)
        return bytes(out)

    def _run(self) -> None:
        """Sim thread: step the physics at least every 2 ms and answer each valid packet."""
        buf = bytearray()
        while not self._stop:
            ready, _, _ = select.select([self.master], [], [], 0.002)
            with self.lock:
                self._step()
                if not ready:
                    continue
                try:
                    data = os.read(self.master, 4096)
                except OSError:
                    return
                if self.echo:
                    os.write(self.master, data)
                buf += data
                while True:
                    while len(buf) >= 2 and not (buf[0] == 0xFF and buf[1] == 0xFF):
                        del buf[0]
                    if len(buf) < 4 or len(buf) < 4 + buf[3]:
                        break
                    total = 4 + buf[3]
                    pkt = bytes(buf[:total])
                    del buf[:total]
                    if P.checksum(pkt[2:-1]) != pkt[-1]:
                        continue
                    reply = self._handle(pkt)
                    if reply:
                        os.write(self.master, reply)
