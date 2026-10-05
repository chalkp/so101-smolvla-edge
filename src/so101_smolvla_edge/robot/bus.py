"""One Feetech serial bus (one USB adapter, one arm): transactions with deadlines and statistics."""

from __future__ import annotations

import logging
import select
import threading
import time
from dataclasses import dataclass, field

from . import protocol as P

log = logging.getLogger("so101_smolvla_edge.bus")


@dataclass
class BusStats:
    transactions: int = 0
    timeouts: int = 0  # transactions that ended with replies missing
    missing_replies: int = 0
    checksum_errors: int = 0
    servo_errors: int = 0  # replies with a non-zero error byte
    echo_mismatches: int = 0  # adapter echo that differed from what was sent
    last_rtt_ms: float = 0.0  # ms, write to last reply
    max_rtt_ms: float = 0.0  # ms
    error_bits: dict[int, int] = field(default_factory=dict)  # servo id -> last non-zero error byte


class Bus:
    """A serial port of Feetech servos. Transactions are serialised by a lock."""

    def __init__(
        self,
        port: str,
        baud: int = 1_000_000,
        reply_margin_s: float = 0.0025,
        echo: bool | None = None,
        retries: int = 1,
    ):
        self.port_name = port
        self.baud = baud
        self.reply_margin = reply_margin_s  # s, USB latency and servo turnaround past the wire time
        self.echo = echo  # adapter echoes TX onto RX; None = unknown until detect_echo()
        self.retries = retries  # extra attempts for ping, read and write_reg
        self.stats = BusStats()
        self._parser = P.StatusParser()
        self._lock = threading.Lock()
        self._ser = None  # serial.Serial while open
        self._guard_until = 0.0  # time.monotonic() before which a late reply may still arrive

    # ------------------------------------------------------------------ port
    def open(self) -> None:
        """Open the port with timeout=0 and leave it that way.

        Reads wait in select() on the descriptor instead of pyserial's timeout, because changing
        that timeout costs a tcsetattr call every time.
        """
        import serial  # pyserial

        self._ser = serial.Serial(
            self.port_name, self.baud, timeout=0, write_timeout=0.05, exclusive=True
        )
        try:
            self._ser.set_low_latency_mode(True)  # FTDI-style; CDC-ACM is low latency already
        except Exception:  # unsupported on many adapters, and harmless
            pass

    def close(self) -> None:
        if self._ser is not None:
            self._ser.close()
            self._ser = None

    def __enter__(self) -> Bus:
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def wire_time(self, n_bytes: int) -> float:
        """Seconds to send n_bytes at 8N1 (10 bits per byte)."""
        return n_bytes * 10.0 / self.baud

    def _read_until(self, n: int, deadline: float) -> bytes:
        """Read up to n bytes, returning what arrived by `deadline`."""
        out = bytearray()
        fd = self._ser.fileno()
        while len(out) < n:
            left = deadline - time.monotonic()
            if left <= 0:
                break
            ready, _, _ = select.select([fd], [], [], left)
            if not ready:
                break
            chunk = self._ser.read(n - len(out))
            if not chunk:  # readable but empty: the device went away
                raise OSError(
                    f"{self.port_name}: device reports readable but returned no data "
                    "(disconnected?)"
                )
            out += chunk
        return bytes(out)

    # ------------------------------------------------------------------ transactions
    def transact(self, tx: bytes, expect_ids: list[int], n_params: int) -> list[P.Status | None]:
        """Send `tx`, then collect one status packet per id in `expect_ids`, None where missing.

        Flushes stale input first, consumes the adapter's echo if it has one, and reads until every
        expected servo answered or the deadline passed. The deadline is the wire time of the whole
        exchange plus `reply_margin_s`, so a missing servo costs a few milliseconds, not a fixed
        timeout.
        """
        if self._ser is None:
            raise RuntimeError(f"bus {self.port_name} is not open")
        with self._lock:
            st = self.stats
            st.transactions += 1
            now = time.monotonic()
            if now < self._guard_until:  # let a late reply to a timed-out read land first
                time.sleep(self._guard_until - now)
            self._ser.reset_input_buffer()
            self._parser.reset()
            chk0 = self._parser.checksum_errors
            rx_bytes = len(expect_ids) * P.status_size(n_params)
            start = time.monotonic()
            deadline = start + self.wire_time(len(tx) + rx_bytes) + self.reply_margin
            self._ser.write(tx)
            if self.echo:
                got = self._read_until(len(tx), deadline)
                if got != tx:
                    st.echo_mismatches += 1
                    self._parser.feed(got)  # maybe the adapter does not echo after all
            out: list[P.Status | None] = [None] * len(expect_ids)
            remaining = len(expect_ids)
            index = {sid: i for i, sid in enumerate(expect_ids)}

            def drain() -> int:
                nonlocal remaining
                while (pkt := self._parser.next()) is not None:
                    i = index.get(pkt.id)
                    if i is None or out[i] is not None or len(pkt.params) != n_params:
                        continue  # foreign, duplicate or late reply
                    if pkt.error:
                        st.servo_errors += 1
                        st.error_bits[pkt.id] = pkt.error
                    out[i] = pkt
                    remaining -= 1
                return remaining

            while remaining and time.monotonic() < deadline:
                chunk = self._read_until(1, deadline)
                if not chunk:
                    break
                waiting = self._ser.in_waiting
                if waiting:
                    chunk += self._ser.read(waiting)
                self._parser.feed(chunk)
                drain()
            rtt = (time.monotonic() - start) * 1e3
            st.last_rtt_ms, st.max_rtt_ms = rtt, max(st.max_rtt_ms, rtt)
            st.checksum_errors += self._parser.checksum_errors - chk0
            if remaining:
                st.timeouts += 1
                st.missing_replies += remaining
                self._guard_until = time.monotonic() + self.wire_time(P.status_size(n_params))
            return out

    def ping(self, sid: int) -> bool:
        """True if the servo answers, retrying as configured."""
        return any(
            self.transact(P.ping(sid), [sid], 0)[0] is not None for _ in range(self.retries + 1)
        )

    def read(self, sid: int, addr: int, n: int) -> P.Status | None:
        """READ n bytes at addr, retrying as configured; None if the servo never answered."""
        for _ in range(self.retries + 1):
            r = self.transact(P.read(sid, addr, n), [sid], n)[0]
            if r is not None:
                return r
        return None

    def read_reg(self, sid: int, reg: P.Reg) -> int | None:
        """One register as an unsigned value, or None."""
        r = self.read(sid, reg.addr, reg.size)
        if r is None:
            return None
        return r.params[0] if reg.size == 1 else P.u16(r.params)

    def write_reg(self, sid: int, reg: P.Reg, value: int) -> bool:
        """Write one register, value masked to its size; True if the servo acknowledged cleanly."""
        tx = P.write(sid, reg.addr, P.le(value & ((1 << (8 * reg.size)) - 1), reg.size))
        for _ in range(self.retries + 1):
            r = self.transact(tx, [sid], 0)[0]
            if r is not None:
                return r.error == 0
        return False

    def sync_write(self, addr: int, n: int, entries: list[tuple[int, bytes]]) -> None:
        """SYNC_WRITE n bytes at addr for each (id, data). No reply, so no retry."""
        self.transact(P.sync_write(addr, n, entries), [], 0)

    def sync_read(self, ids: list[int], addr: int, n: int) -> list[P.Status | None]:
        """SYNC_READ n bytes at addr from each id, None for servos that did not answer. No retry."""
        return self.transact(P.sync_read(addr, n, ids), ids, n)

    def detect_echo(self, sid: int) -> bool | None:
        """Find out whether the adapter echoes TX onto RX and set `echo` to match.

        Returns None, leaving `echo` alone, if the servo did not answer at all.
        """
        tx = P.read(sid, P.MODEL_NUMBER.addr, 1)
        with self._lock:
            self._ser.reset_input_buffer()
            parser = P.StatusParser()
            wire = self.wire_time(2 * len(tx) + P.status_size(1))
            deadline = time.monotonic() + wire + 2 * self.reply_margin
            self._ser.write(tx)
            saw_echo = saw_reply = False
            while time.monotonic() < deadline and not (saw_reply and saw_echo):
                chunk = self._read_until(1, deadline)
                if not chunk:
                    break
                parser.feed(chunk + self._ser.read(self._ser.in_waiting))
                while (pkt := parser.next()) is not None:
                    if pkt.id != sid:
                        continue
                    # our own READ parses as a status packet with ERR = READ, params = addr, n
                    if (
                        len(pkt.params) == 2
                        and pkt.error == P.READ
                        and pkt.params[0] == P.MODEL_NUMBER.addr
                    ):
                        saw_echo = True
                    elif len(pkt.params) == 1:
                        saw_reply = True
        if not saw_reply:
            return None
        self.echo = saw_echo
        return saw_echo
