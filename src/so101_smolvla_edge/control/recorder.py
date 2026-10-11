"""Episode recording for real-robot data: camera frames, joint states, commanded goals and who was
in control, at a fixed rate.

    <record.dir>/<session>/episode_0003/
        top_rgb/000000.jpg ...  one directory per camera (the key's last component)
        steps.npz               t, seq, mode (n), state, action, leader (n, J), frame_age (n, C)
        episode.json            label, note, joints, cameras, fps, frames, dropped, ...

`state` is the measured joint positions and `action` the goal written to the servos at that
instant, both in policy radians and in the runner's joint order (listed in episode.json): the same
pair the simulator dataset stores. `mode` says who produced the action: 0 policy, 1 leader arms
(a demonstration or a takeover), 2 holding, 3 fault, 4 engaging. The training repo's
`wm_rank/scripts/real_to_lerobot.py` turns sessions into the LeRobot dataset the trainer reads.

A sampler thread snapshots frames and joints every 1/fps and a writer thread encodes them. The
queue between them is bounded, so a slow disk costs dropped (and counted) frames, never a stalled
control loop.
"""

from __future__ import annotations

import json
import logging
import queue
import shutil
import threading
import time
from pathlib import Path

import numpy as np

log = logging.getLogger("so101_smolvla_edge.recorder")

# `mode` in steps.npz by runner state. "engaging" is a takeover's ramp toward the leaders: the goal
# is the ramp, not the operator, so those frames must never become action labels.
MODES = {"running": 0, "teleop": 1, "holding": 2, "fault": 3, "disabled": 3, "engaging": 4}


class Recorder:
    """Records episodes into a session directory; begin() and end() (or discard()) bracket one."""

    def __init__(self, cfg, joints: list[str], cameras: dict, meta: dict | None = None):
        self.cfg, self.joints, self.cameras = cfg, joints, cameras
        self.meta = meta or {}
        self.session = Path(cfg.dir) / time.strftime("%Y%m%d-%H%M%S")
        self.session.mkdir(parents=True, exist_ok=True)
        self._q: queue.Queue = queue.Queue(maxsize=cfg.queue)
        self._mu = threading.Lock()
        self._stop = threading.Event()
        self._ep: dict | None = None
        self.episodes = 0
        self.saved: list[Path] = []
        self._sample = None  # set by start()
        self._threads = [
            threading.Thread(target=self._sampler, daemon=True, name="rec-sample"),
            threading.Thread(target=self._writer, daemon=True, name="rec-write"),
        ]

    def start(self, sample) -> None:
        """Start the threads; `sample()` returns the runner's (q, goal, mode, leader, seq)."""
        self._sample = sample
        for th in self._threads:
            th.start()

    @property
    def active(self) -> bool:
        return self._ep is not None

    # ------------------------------------------------------------------ episode control
    def begin(self) -> Path | None:
        """Open a new episode and return its directory, or None if one is already open."""
        with self._mu:
            if self._ep is not None:
                return None
            d = self.session / f"episode_{self.episodes:04d}"
            for key in self.cameras:
                (d / key.split(".")[-1]).mkdir(parents=True, exist_ok=True)
            self._ep = {
                "dir": d,
                "t": [],
                "state": [],
                "action": [],
                "mode": [],
                "leader": [],
                "age": [],
                "seq": [],
                "n": 0,
                "dropped": 0,
                "wall0": time.time(),
                "t0": time.monotonic(),
            }
            log.info("recording %s", d)
            return d

    def end(self, label: str, note: str = "") -> Path | None:
        """Close the episode and write it; None if nothing was open or recorded.

        `label` is the operator's verdict (success, fail, ...).
        """
        with self._mu:
            ep, self._ep = self._ep, None
        if ep is None:
            return None
        self._q.join()  # every queued frame of this episode is on disk
        d, n = ep["dir"], ep["n"]
        if n == 0:
            shutil.rmtree(d, ignore_errors=True)
            return None
        np.savez_compressed(
            d / "steps.npz",
            t=np.asarray(ep["t"]),
            state=np.asarray(ep["state"], np.float32),
            action=np.asarray(ep["action"], np.float32),
            mode=np.asarray(ep["mode"], np.int8),
            leader=np.asarray(ep["leader"], np.float32),
            frame_age=np.asarray(ep["age"], np.float32),
            seq=np.asarray(ep["seq"], np.int64),
        )
        modes = np.asarray(ep["mode"])
        info = {
            "label": label,
            "note": note,
            "frames": n,
            "dropped": ep["dropped"],
            "fps": self.cfg.fps,
            "duration_s": float(ep["t"][-1] - ep["t"][0]) if n > 1 else 0.0,
            "frames_by_mode": {
                k: int((modes == v).sum()) for k, v in MODES.items() if k != "disabled"
            },
            "joints": self.joints,
            "cameras": list(self.cameras),
            "wall_start": ep["wall0"],
            **self.meta,
        }
        (d / "episode.json").write_text(json.dumps(info, indent=1))
        self.episodes += 1
        self.saved.append(d)
        log.info("saved %s: %s, %d frames (%d dropped)", d.name, label, n, ep["dropped"])
        return d

    def discard(self) -> bool:
        """Drop the open episode and delete its directory; False if none was open."""
        with self._mu:
            ep, self._ep = self._ep, None
        if ep is None:
            return False
        self._q.join()
        shutil.rmtree(ep["dir"], ignore_errors=True)
        log.info("discarded %s", ep["dir"].name)
        return True

    def close(self) -> None:
        """Save an open episode as "unlabelled" and stop the threads."""
        if self._ep is not None:
            self.end("unlabelled", "run ended with the episode open")
        self._stop.set()
        self._q.put(None)
        for th in self._threads:
            if th.is_alive():
                th.join(timeout=3)

    # ------------------------------------------------------------------ threads
    def _sampler(self) -> None:
        """Every 1/fps, add a runner snapshot to the open episode and queue its frames."""
        period = 1.0 / self.cfg.fps
        nxt = time.monotonic()
        while not self._stop.is_set():
            now = time.monotonic()
            if now < nxt:
                time.sleep(min(nxt - now, 0.005))
                continue
            nxt = max(nxt + period, now - period)  # no burst after a stall
            if self._ep is None:
                continue
            # snapshot without holding self._mu: the runner calls begin() under its own lock, so
            # taking the two locks in the other order here would deadlock
            q, goal, mode, leader, seq = self._sample()
            if not (np.isfinite(q).all() and np.isfinite(goal).all()):
                continue  # nothing commanded yet (torque off)
            frames, ages = {}, []
            for key, cam in self.cameras.items():
                img, t_img = cam.latest()
                frames[key] = img
                ages.append(now - t_img if img is not None else np.nan)
            with self._mu:
                ep = self._ep
                if ep is None:
                    continue
                if any(v is None for v in frames.values()):
                    ep["dropped"] += 1
                    continue
                try:
                    self._q.put_nowait((ep["dir"], ep["n"], frames))
                except queue.Full:
                    ep["dropped"] += 1
                    continue
                ep["t"].append(now)
                ep["state"].append(q)
                ep["action"].append(goal)
                ep["mode"].append(MODES.get(mode, 3))
                ep["leader"].append(leader)
                ep["age"].append(ages)
                ep["seq"].append(seq)
                ep["n"] += 1

    def _writer(self) -> None:
        """Write queued frames as JPEG until close()."""
        import cv2

        params = [cv2.IMWRITE_JPEG_QUALITY, int(self.cfg.jpeg_quality)]
        while True:
            item = self._q.get()
            try:
                if item is None:
                    return
                d, i, frames = item
                for key, rgb in frames.items():
                    path = d / key.split(".")[-1] / f"{i:06d}.jpg"
                    ok = cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), params)
                    if not ok:
                        log.warning("could not write frame %d of %s", i, key)
            except Exception as exc:  # noqa: BLE001 (a full disk must not kill the thread silently)
                log.error("recorder write failed: %s: %s", type(exc).__name__, exc)
            finally:
                self._q.task_done()
