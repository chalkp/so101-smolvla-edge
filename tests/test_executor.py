"""ChunkExecutor: which chunks it accepts, latency skipping, crossfades, slew and joint limits."""

import numpy as np
import pytest

from so101_smolvla_edge.robot.executor import Chunk, ChunkExecutor, ExecutorConfig

N = 2  # joints


def ex(**kw):
    """An executor for N joints in [-2, 2] rad at 3 rad/s, holding zero from t = 0."""
    cfg = ExecutorConfig(lower=[-2.0] * N, upper=[2.0] * N, max_velocity=[3.0] * N, **kw)
    e = ChunkExecutor(cfg)
    e.reset_hold(np.zeros(N), 0.0)
    return e


def ramp(t0, start=0.0, step=0.01, H=50, dt=1 / 30):
    """A chunk of H actions from `start`, rising `step` rad per action, the same on every joint."""
    return Chunk(q=np.tile((start + step * np.arange(H))[:, None], (1, N)), t0=t0, dt=dt)


def test_rejections():
    """Each reason to reject a chunk fires on its own case, and every rejection is counted."""
    e = ChunkExecutor(ExecutorConfig(lower=[-1] * N, upper=[1] * N, max_velocity=[1] * N))
    assert e.push(ramp(0.0), 0.0)[1].startswith("no hold target")
    e = ex(max_first_jump=1.0)
    assert "future" in e.push(ramp(1.0), 0.5)[1]
    assert "too old" in e.push(ramp(0.0), 3.0)[1]
    assert "exhausted" in e.push(ramp(0.0), 1.66)[1]
    assert "jumps" in e.push(ramp(0.0, start=1.2), 0.0)[1]  # 1.2 rad from the hold, limit 1.0
    assert e.push(Chunk(q=np.full((50, N), np.nan), t0=0.0), 0.0)[1].startswith("bad shape")
    assert e.push(Chunk(q=np.zeros((50, N + 1)), t0=0.0), 0.0)[1] == "bad shape"
    assert e.rejected == 6 and e.accepted == 0


def test_latency_is_skipped_not_replayed():
    """A chunk that arrives 0.4 s after its observation is entered 0.4 s in, not at its start."""
    e = ex(crossfade_s=0.0)
    assert e.push(ramp(0.0, step=0.003), 0.4)[0]
    out = e.sample(0.4)
    assert out == pytest.approx(0.036, abs=1e-3)  # action 12, reachable in one slew step


def test_slew_limit_and_hold_past_end():
    """Output steps stay within max_velocity * dt, and the last action is held past the end."""
    e = ex(crossfade_s=0.0)
    e.push(Chunk(q=np.tile(np.linspace(0, 0.3, 50)[:, None], (1, N)), t0=0.0), 0.0)
    prev, t = 0.0, 0.0
    for _ in range(300):
        t += 0.01
        v = e.sample(t)
        assert abs(v[0] - prev) <= 3.0 * 0.01 + 1e-12
        prev = v[0]
    assert v[0] == pytest.approx(0.3)
    assert not e.status(t).executing


def test_crossfade_is_continuous():
    """A new chunk blends in over crossfade_s: a 0.2 rad step is spread over 0.1 s."""
    e = ex(crossfade_s=0.1)
    e.push(ramp(0.0, step=0.0), 0.0)
    for k in range(10):
        e.sample(k * 0.01)
    assert e.push(ramp(0.1, start=0.2, step=0.0), 0.1)[0]
    vals = [e.sample(0.1 + k * 0.005)[0] for k in range(1, 30)]
    steps = np.abs(np.diff([0.0] + vals))
    assert steps.max() < 0.2 * 0.005 / 0.1 + 1e-9
    assert vals[-1] == pytest.approx(0.2)


def test_joint_limits_clamp():
    """Targets beyond the joint limits are clamped to them."""
    e = ex(crossfade_s=0.0, max_start_jump=10.0, max_first_jump=10.0)
    e.push(Chunk(q=np.full((50, N), 5.0), t0=0.0), 0.0)
    t = 0.0
    for _ in range(200):
        t += 0.01
        v = e.sample(t)
    assert (v == 2.0).all()


def test_first_chunk_tolerance_then_consistency():
    """The first chunk after a hold may jump max_first_jump, the next only max_start_jump."""
    e = ex(crossfade_s=0.0, max_start_jump=0.35, max_first_jump=1.0)  # independent of the defaults
    assert e.push(ramp(0.0, start=0.6, step=0.0), 0.0)[0]  # 0.6 rad from the hold pose
    assert not e.push(ramp(0.1, start=0.0, step=0.0), 0.1)[0]
    assert "limit 0.35" in e.last_reject


def test_after_a_stall_the_next_chunk_enters_from_rest():
    """After a stall (the active chunk ran out), the next chunk gets the first-chunk limit."""
    e = ex(crossfade_s=0.0, max_start_jump=0.35, max_first_jump=1.0)
    assert e.push(ramp(0.0, step=0.0), 0.0)[0]  # 50 actions: runs until 1.63 s
    assert not e.push(ramp(1.0, start=0.6, step=0.0), 1.0)[0]  # still live: limit 0.35
    assert e.push(ramp(3.0, start=0.6, step=0.0), 3.0)[0]  # ran out: limit 1.0


def test_first_chunk_may_start_as_far_from_rest_as_the_dataset_does():
    """The first chunk may start as far from rest as the dataset's first actions do.

    In the training episodes the shoulder pans rest well over a radian from their first action.
    Such a chunk is accepted and played as a move at the slew limit; 2.5 rad is refused."""
    e = ChunkExecutor(ExecutorConfig(lower=[-2.0] * N, upper=[2.0] * N, max_velocity=[3.33] * N))
    rest = np.zeros(N)
    rest[0] = -1.2363
    e.reset_hold(rest, 0.0)
    q = np.tile(rest, (50, 1))
    q[:, 0] = 0.14  # a 1.38 rad first step
    ok, why = e.push(Chunk(q=q, t0=0.0, dt=1 / 30, seq=1), 0.1)
    assert ok, why
    v = [e.sample(0.10 + 0.01 * k)[0] for k in range(1, 12)]
    assert np.diff(v) == pytest.approx(0.0333, abs=1e-3)  # 3.33 rad/s for 10 ms
    e2 = ChunkExecutor(ExecutorConfig(lower=[-3.0] * N, upper=[3.0] * N, max_velocity=[3.33] * N))
    e2.reset_hold(rest, 0.0)
    q[:, 0] = 1.3  # 2.5 rad away: not something the dataset does
    assert not e2.push(Chunk(q=q, t0=0.0, dt=1 / 30, seq=1), 0.1)[0]
