"""Bus against simulated servos: register access, sync read/write, timeouts, echo detection."""

import time

import pytest

from so101_smolvla_edge.robot import protocol as P
from so101_smolvla_edge.robot.bus import Bus
from so101_smolvla_edge.robot.servo_sim import ServoSim


@pytest.fixture
def sim():
    """Three simulated servos, ids 1 to 3."""
    s = ServoSim([1, 2, 3])
    yield s
    s.close()


def test_read_write_sync(sim):
    """Ping, register read and write, sync write and sync read all reach the servos."""
    with Bus(sim.path, echo=False, reply_margin_s=0.02) as bus:
        assert bus.ping(2) and not bus.ping(7)
        assert bus.read_reg(1, P.MODEL_NUMBER) == P.MODEL_STS3215
        assert bus.write_reg(2, P.P_COEFFICIENT, 16) and sim.get(2, P.P_COEFFICIENT) == 16
        bus.sync_write(P.GOAL_POSITION.addr, 2, [(1, P.le(1000, 2)), (3, P.le(3000, 2))])
        time.sleep(0.02)  # sync write has no reply to wait for
        assert sim.get(1, P.GOAL_POSITION) == 1000 and sim.get(3, P.GOAL_POSITION) == 3000
        r = bus.sync_read([1, 2, 3], P.TELEMETRY_ADDR, P.TELEMETRY_LEN)
        assert all(x is not None for x in r)
        assert P.Telemetry.decode(r[1].params).voltage == 12.0


def test_missing_servo_costs_a_deadline_not_a_hang(sim):
    """A servo that never answers costs one reply deadline and a None in its slot, nothing more."""
    sim.silent.add(2)
    with Bus(sim.path, echo=False, reply_margin_s=0.02) as bus:
        t = time.monotonic()
        r = bus.sync_read([1, 2, 3], P.TELEMETRY_ADDR, P.TELEMETRY_LEN)
        assert time.monotonic() - t < 0.1
        assert r[0] is not None and r[1] is None and r[2] is not None
        assert bus.stats.timeouts == 1 and bus.stats.missing_replies == 1


def test_echo_adapter_detected():
    """An adapter that echoes TX onto RX is detected, and reads still work through it."""
    s = ServoSim([1], echo=True)
    try:
        with Bus(s.path, echo=None, reply_margin_s=0.02) as bus:
            assert bus.detect_echo(1) is True
            assert bus.read_reg(1, P.MODEL_NUMBER) == P.MODEL_STS3215
    finally:
        s.close()
