"""Feetech STS ("protocol 0") packets and a resynchronising status parser. No I/O here.

    instruction:  FF FF ID LEN INSTR P1..Pn CHK               LEN = n + 2
    status:       FF FF ID LEN ERR   P1..Pn CHK
    CHK = ~(ID + LEN + INSTR/ERR + P1 + .. + Pn) & 0xFF

    SYNC_WRITE:   FF FF FE LEN 83 ADDR L [ID D1..DL] x N CHK  no reply
    SYNC_READ:    FF FF FE LEN 82 ADDR L ID1..IDN CHK         one status packet per ID

Multi-byte fields are little-endian. Control table as in the lerobot copy that read the LeHome
dataset (lehome/devices/lerobot/common/motors/feetech/tables.py).
Ported from ros2_ws/src/feetech_bus.
"""

from __future__ import annotations

from dataclasses import dataclass

BROADCAST = 0xFE  # id addressing every servo

# instructions
PING = 0x01
READ = 0x02
WRITE = 0x03
SYNC_READ = 0x82
SYNC_WRITE = 0x83

# bits of the status packet's error byte
ERR_VOLTAGE = 1 << 0
ERR_ANGLE = 1 << 1
ERR_OVERHEAT = 1 << 2
ERR_OVERCURRENT = 1 << 3
ERR_OVERLOAD = 1 << 5
ERROR_NAMES = {
    ERR_VOLTAGE: "voltage",
    ERR_ANGLE: "angle",
    ERR_OVERHEAT: "overheat",
    ERR_OVERCURRENT: "overcurrent",
    ERR_OVERLOAD: "overload",
}


@dataclass(frozen=True)
class Reg:
    addr: int  # control table address
    size: int  # bytes, 1 or 2


# EEPROM: writes need Lock = 0 and wear the flash, so write only what differs
MODEL_NUMBER = Reg(3, 2)
ID = Reg(5, 1)
RETURN_DELAY_TIME = Reg(7, 1)
MIN_POSITION_LIMIT = Reg(9, 2)
MAX_POSITION_LIMIT = Reg(11, 2)
MAX_TEMPERATURE_LIMIT = Reg(13, 1)
MAX_VOLTAGE_LIMIT = Reg(14, 1)
MIN_VOLTAGE_LIMIT = Reg(15, 1)
MAX_TORQUE_LIMIT = Reg(16, 2)
P_COEFFICIENT = Reg(21, 1)
D_COEFFICIENT = Reg(22, 1)
I_COEFFICIENT = Reg(23, 1)
PROTECTION_CURRENT = Reg(28, 2)
HOMING_OFFSET = Reg(31, 2)
OPERATING_MODE = Reg(33, 1)
OVERLOAD_TORQUE = Reg(36, 1)
# SRAM
TORQUE_ENABLE = Reg(40, 1)
ACCELERATION = Reg(41, 1)
GOAL_POSITION = Reg(42, 2)
TORQUE_LIMIT = Reg(48, 2)
LOCK = Reg(55, 1)
PRESENT_POSITION = Reg(56, 2)
PRESENT_VELOCITY = Reg(58, 2)
PRESENT_LOAD = Reg(60, 2)
PRESENT_VOLTAGE = Reg(62, 1)
PRESENT_TEMPERATURE = Reg(63, 1)
STATUS = Reg(65, 1)
MOVING = Reg(66, 1)
PRESENT_CURRENT = Reg(69, 2)
# factory area: not in Feetech's published table, but lerobot writes it
MAXIMUM_ACCELERATION = Reg(85, 1)

MODEL_STS3215 = 777
HOMING_OFFSET_SIGN_BIT = 11
VELOCITY_SIGN_BIT = 15
LOAD_SIGN_BIT = 10  # Present_Load is 0.1 % of max torque, direction in bit 10
CURRENT_SIGN_BIT = 15
CURRENT_A_PER_LSB = 0.0065  # A, Present_Current is 6.5 mA per unit

# addresses 56..70: one SYNC_READ returns everything the safety layer watches
TELEMETRY_ADDR = 56
TELEMETRY_LEN = 15


def decode_sign_magnitude(raw: int, sign_bit: int) -> int:
    mag = raw & ((1 << sign_bit) - 1)
    return -mag if (raw >> sign_bit) & 1 else mag


def encode_sign_magnitude(value: int, sign_bit: int) -> int:
    mag = abs(value)
    if mag >= 1 << sign_bit:
        raise ValueError(f"{value} does not fit in {sign_bit} magnitude bits")
    return mag | (1 << sign_bit) if value < 0 else mag


def u16(b: bytes, off: int = 0) -> int:
    """Little-endian 16-bit value at b[off]."""
    return b[off] | (b[off + 1] << 8)


def le(value: int, size: int) -> bytes:
    """`value` as `size` little-endian bytes."""
    return bytes((value >> (8 * i)) & 0xFF for i in range(size))


def checksum(body: bytes) -> int:
    return ~sum(body) & 0xFF


def packet(id_: int, instr: int, params: bytes = b"") -> bytes:
    """Instruction packet with header, length and checksum."""
    if len(params) + 2 > 0xFF:
        raise ValueError("packet too long for the 8-bit LEN field")
    body = bytes([id_, len(params) + 2, instr]) + params
    return b"\xff\xff" + body + bytes([checksum(body)])


def ping(id_: int) -> bytes:
    return packet(id_, PING)


def read(id_: int, addr: int, n: int) -> bytes:
    return packet(id_, READ, bytes([addr, n]))


def write(id_: int, addr: int, data: bytes) -> bytes:
    return packet(id_, WRITE, bytes([addr]) + data)


def sync_write(addr: int, n: int, entries: list[tuple[int, bytes]]) -> bytes:
    params = bytearray([addr, n])
    for id_, data in entries:
        if len(data) != n:
            raise ValueError(f"sync_write entry for id {id_} has {len(data)} bytes, expected {n}")
        params.append(id_)
        params += data
    return packet(BROADCAST, SYNC_WRITE, bytes(params))


def sync_read(addr: int, n: int, ids: list[int]) -> bytes:
    return packet(BROADCAST, SYNC_READ, bytes([addr, n, *ids]))


def status_size(n_params: int) -> int:
    """Bytes in a status packet carrying n_params parameters."""
    return 6 + n_params


@dataclass
class Status:
    id: int  # servo that answered
    error: int  # error byte, ERR_* bits
    params: bytes


class StatusParser:
    """Status packet parser: `feed()` it whatever the port returned, pull packets with `next()`.

    On any inconsistency it drops one byte and rescans, so leading junk, a truncated reply from a
    browned-out servo or line noise costs that packet and nothing after it.
    """

    def __init__(self) -> None:
        self.buf = bytearray()  # bytes not parsed yet
        self.checksum_errors = 0  # cumulative, reset() keeps it
        self.discarded = 0  # bytes dropped while resynchronising

    def reset(self) -> None:
        """Drop buffered bytes; the counters keep counting."""
        self.buf.clear()

    def feed(self, data: bytes) -> None:
        self.buf += data

    def next(self) -> Status | None:
        """Next complete, valid packet, or None until more bytes arrive."""
        b = self.buf
        while True:
            if len(b) < 2:
                return None
            if b[0] != 0xFF or b[1] != 0xFF:
                del b[0]
                self.discarded += 1
                continue
            if len(b) < 4:
                return None
            if b[2] == 0xFF or b[3] < 2:  # FF FF FF: the first FF is junk. LEN < 2 is impossible
                del b[0]
                self.discarded += 1
                continue
            total = 4 + b[3]
            if len(b) < total:
                return None
            if checksum(bytes(b[2 : total - 1])) != b[total - 1]:
                self.checksum_errors += 1
                del b[0]
                self.discarded += 1
                continue
            pkt = Status(b[2], b[4], bytes(b[5 : total - 1]))
            del b[:total]
            return pkt


@dataclass
class Telemetry:
    """Decoded addresses 56..70 of one servo, raw units."""

    position: int  # ticks, 0..4095
    velocity: int  # signed servo units (steps/s on the STS3215, not verified on this hardware)
    load: int  # signed, 0.1 % of max torque
    voltage: float  # V
    temperature: int  # deg C
    status: int  # Status register, raw
    moving: int  # Moving flag
    current: float  # A, signed

    @classmethod
    def decode(cls, p: bytes) -> Telemetry:
        o = TELEMETRY_ADDR
        return cls(
            position=u16(p, PRESENT_POSITION.addr - o),
            velocity=decode_sign_magnitude(u16(p, PRESENT_VELOCITY.addr - o), VELOCITY_SIGN_BIT),
            load=decode_sign_magnitude(u16(p, PRESENT_LOAD.addr - o), LOAD_SIGN_BIT),
            voltage=p[PRESENT_VOLTAGE.addr - o] / 10.0,
            temperature=p[PRESENT_TEMPERATURE.addr - o],
            status=p[STATUS.addr - o],
            moving=p[MOVING.addr - o],
            current=(
                decode_sign_magnitude(u16(p, PRESENT_CURRENT.addr - o), CURRENT_SIGN_BIT)
                * CURRENT_A_PER_LSB
            ),
        )
