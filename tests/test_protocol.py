"""Feetech STS packet encoding, sign-magnitude fields and the status parser."""

from so101_smolvla_edge.robot import protocol as P


def test_checksum_and_packet():
    """Ping and write packets have the documented layout and a valid checksum."""
    assert P.ping(1) == bytes([0xFF, 0xFF, 0x01, 0x02, 0x01, 0xFB])
    pkt = P.write(3, P.GOAL_POSITION.addr, P.le(2048, 2))
    assert pkt[:5] == bytes([0xFF, 0xFF, 3, 5, P.WRITE]) and pkt[5:8] == bytes([42, 0x00, 0x08])
    assert P.checksum(pkt[2:-1]) == pkt[-1]


def test_sync_packets():
    """Sync write and sync read go to the broadcast id with the right length byte."""
    sw = P.sync_write(42, 2, [(1, b"\x00\x08"), (2, b"\x01\x08")])
    assert sw[2] == P.BROADCAST and sw[3] == (2 + 1) * 2 + 4 and sw[4] == P.SYNC_WRITE
    sr = P.sync_read(56, 15, [1, 2, 3])
    assert sr[3] == 3 + 4 and list(sr[5:10]) == [56, 15, 1, 2, 3]


def test_sign_magnitude():
    """Sign-magnitude values round-trip, and Present_Load carries its sign in bit 10."""
    for v in (0, 1, -1, 2047, -2047):
        assert P.decode_sign_magnitude(P.encode_sign_magnitude(v, 11), 11) == v
    assert P.decode_sign_magnitude(0x0400 | 5, P.LOAD_SIGN_BIT) == -5


def status(sid, params, err=0):
    """A status packet from servo `sid` with a valid checksum."""
    body = bytes([sid, len(params) + 2, err]) + params
    return b"\xff\xff" + body + bytes([P.checksum(body)])


def test_parser_resyncs_over_junk_and_corruption():
    """The parser skips junk and a bad checksum, and completes a packet split across reads."""
    good1, good2 = status(1, b"\x10\x20"), status(2, b"\x30\x40")
    bad = bytearray(status(9, b"\x00\x00"))
    bad[-1] ^= 0xFF
    p = P.StatusParser()
    p.feed(b"\x00\x13\xff" + bytes(bad) + good1 + b"\xff" + good2[:3])
    a = p.next()
    assert a.id == 1 and a.params == b"\x10\x20"
    assert p.next() is None
    p.feed(good2[3:])
    b = p.next()
    assert b.id == 2 and b.params == b"\x30\x40"
    assert p.checksum_errors == 1


def test_telemetry_decode():
    """One telemetry read decodes to position, signed velocity and load, volts, C and amps."""
    raw = bytearray(15)
    raw[0:2] = P.le(3000, 2)
    raw[2:4] = P.le(P.encode_sign_magnitude(-100, 15), 2)
    raw[4:6] = P.le(P.encode_sign_magnitude(-250, 10), 2)
    raw[6], raw[7] = 121, 41
    raw[13:15] = P.le(200, 2)
    t = P.Telemetry.decode(bytes(raw))
    assert (t.position, t.velocity, t.load, t.voltage, t.temperature) == (
        3000,
        -100,
        -250,
        12.1,
        41,
    )
    assert abs(t.current - 1.3) < 1e-9
