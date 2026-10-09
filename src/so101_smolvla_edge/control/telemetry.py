"""Run logs.

    <log_dir>/<YYYYmmdd-HHMMSS>/
        config.json     the resolved run configuration and versions
        ticks.csv       one row per arm per logged control tick: loop time, bus round trip, state,
                        then FIELDS for each joint
        events.jsonl    state changes, faults, warnings, operator commands
        chunks.jsonl    one line per inference: observation time, latency, accepted or not and why
        inputs/         <seq>_<camera>.jpg, the frames each inference was given (log_inputs)

A background thread does the writing, so the control loops never wait on the disk.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from pathlib import Path

import numpy as np

log = logging.getLogger("so101_smolvla_edge.telemetry")

# ticks.csv columns per joint: executor target, goal written, measured position and velocity,
# load (%), current (A), temperature (C), voltage (V), lead clamp applied (0/1)
FIELDS = ("target", "goal", "pos", "vel", "load", "cur", "temp", "volt", "clamp")


class RunLog:
    """One run's log directory.

    Records are queued for a writer thread, and dropped (and counted) if it falls behind.
    """

    def __init__(self, root: str | Path, joints_by_arm: dict[str, list[str]], config: dict):
        self.dir = Path(root) / time.strftime("%Y%m%d-%H%M%S")
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "config.json").write_text(json.dumps(config, indent=2, default=str))
        self._q: queue.Queue = queue.Queue(maxsize=20000)
        self.dropped = 0
        n = max(len(j) for j in joints_by_arm.values())
        per_joint = [f"{f}{k}" for k in range(n) for f in FIELDS]
        cols = ["t", "arm", "loop_ms", "rtt_ms", "state"] + per_joint
        self._ticks = open(self.dir / "ticks.csv", "w", buffering=1 << 16)
        self._ticks.write("# joints: " + json.dumps(joints_by_arm) + "\n" + ",".join(cols) + "\n")
        self._events = open(self.dir / "events.jsonl", "a", buffering=1)
        self._chunks = open(self.dir / "chunks.jsonl", "a", buffering=1)
        self._thread = threading.Thread(target=self._run, daemon=True, name="runlog")
        self._thread.start()

    def _put(self, item) -> None:
        try:
            self._q.put_nowait(item)
        except queue.Full:
            self.dropped += 1

    def tick(
        self, t: float, arm: str, loop_ms: float, rtt_ms: float, state: str, cols: np.ndarray
    ) -> None:
        """Log one control tick of `arm`; `cols` is (len(FIELDS), n_joints)."""
        self._put(("tick", t, arm, loop_ms, rtt_ms, state, cols))

    def event(self, rec: dict) -> None:
        """Append `rec` to events.jsonl."""
        self._put(("event", rec))

    def chunk(self, rec: dict) -> None:
        """Append `rec` to chunks.jsonl."""
        self._put(("chunk", rec))

    def inputs(self, seq: int, images: dict) -> None:
        """Save the frames one inference was given (RGB uint8 by camera key) to inputs/.

        No copy is made: the camera threads replace their frame arrays, never modify them.
        """
        self._put(("inputs", seq, dict(images)))

    def _run(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                break
            kind = item[0]
            if kind == "tick":
                _, t, arm, loop_ms, rtt_ms, state, cols = item
                vals = ",".join(f"{v:.5g}" for v in cols.T.reshape(-1))
                self._ticks.write(f"{t:.4f},{arm},{loop_ms:.2f},{rtt_ms:.2f},{state},{vals}\n")
            elif kind == "event":
                self._events.write(json.dumps(item[1], default=float) + "\n")
            elif kind == "inputs":
                self._write_inputs(item[1], item[2])
            else:
                self._chunks.write(json.dumps(item[1], default=float) + "\n")

    def _write_inputs(self, seq: int, images: dict) -> None:
        try:
            import cv2

            d = self.dir / "inputs"
            d.mkdir(exist_ok=True)
            for key, rgb in images.items():
                if rgb is not None:
                    path = d / f"{seq:06d}_{key.split('.')[-1]}.jpg"
                    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(path), bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        except Exception:  # noqa: BLE001 (a full disk must not stop the tick log)
            self.dropped += 1

    def close(self) -> None:
        """Write what is queued and close the files."""
        self._q.put(None)
        self._thread.join(timeout=5)
        for f in (self._ticks, self._events, self._chunks):
            f.close()
