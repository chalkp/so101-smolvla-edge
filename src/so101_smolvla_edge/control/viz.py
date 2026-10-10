"""Live view of a run in Rerun: what the cameras see, what the model was given, what it asked for.

The robot process serves a Rerun stream (`run --rerun`, or `viz.rerun: true`) and a viewer on
another machine connects to it:

    rerun --connect rerun+http://<robot>:9876/proxy   (pip install rerun-sdk at the robot's version)
    http://<robot>:9090/?url=...                      (the web viewer, served by the robot)

Logged on one timeline, `time`, in seconds since the run started:

    model/input/<camera>   the exact frame handed to the policy, at every inference
    live/<camera>          the newest camera frame, at viz.live_hz
    plan/<joint>           every action of every chunk, at the time it is scheduled to execute
    joints/<joint>/goal    the goal written to the servo, at viz.live_hz
    joints/<joint>/pos     the measured position, at viz.live_hz
    log                    state changes, faults, accepted and rejected chunks

Nothing here is on the control path. Every call swallows its own errors, and a viewer that is
absent, slow or closed costs the robot only the JPEG encoding.
"""

from __future__ import annotations

import logging
import socket
import threading
import time
from urllib.parse import quote

import numpy as np

log = logging.getLogger("so101_smolvla_edge.viz")


def local_ip() -> str:
    """The address other machines on the network reach this one by (no packet is sent)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def short(key: str) -> str:
    """Last dotted component of a camera key, used in entity paths."""
    return key.split(".")[-1]


class RerunViz:
    """Streams a run to Rerun viewers. Every logging call is best effort."""

    def __init__(self, cfg, joints: list[str], cameras: dict, app_id: str = "so101_smolvla_edge"):
        import rerun as rr  # optional dependency: fail here at start-up, with a clear message

        self.rr, self.cfg, self.joints, self.cameras = rr, cfg, joints, cameras
        self.app_id = app_id
        self.t0 = time.monotonic()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._sample = None
        self._warned = False
        self.urls: list[str] = []

    # ------------------------------------------------------------------ lifecycle
    def start(self, sample) -> list[str]:
        """Serve the stream and the web viewer, start the live thread, and return what to open.

        `sample()` returns (q, goal, mode, leader, seq), the recorder's snapshot. The stream server
        must name the web viewer's origin (http://*:<web_port>) or a browser refuses the stream with
        "Data source failed: js api error", since Rerun allows only localhost and rerun.io by
        default. The native viewer is not affected.
        """
        rr = self.rr
        rr.init(self.app_id)
        host = self.cfg.host or local_ip()
        kw = dict(
            grpc_port=self.cfg.grpc_port,
            default_blueprint=self._blueprint(),
            server_memory_limit=self.cfg.memory_limit,
        )
        origins = [f"http://*:{self.cfg.web_port}"] if self.cfg.web_port else []
        try:
            uri = rr.serve_grpc(**kw, cors_allow_origin=origins + list(self.cfg.cors_allow_origin))
        except TypeError:  # Rerun without cors_allow_origin; the native viewer still works
            uri = rr.serve_grpc(**kw)
        public = f"rerun+http://{host}:{self.cfg.grpc_port}/proxy"
        self.urls = [f"rerun --connect {public}"]
        if self.cfg.web_port:
            try:
                rr.serve_web_viewer(
                    web_port=self.cfg.web_port, open_browser=False, connect_to=public
                )
                self.urls.append(f"http://{host}:{self.cfg.web_port}/?url={quote(public, safe='')}")
            except Exception as exc:  # noqa: BLE001 (the native viewer still works)
                log.warning("web viewer not started: %s: %s", type(exc).__name__, exc)
        series = (("plan", (120, 120, 255)), ("goal", (255, 170, 0)), ("pos", (0, 200, 120)))
        for j in self.joints:  # one plot per joint: plan, goal and measured position together
            for leaf, colour in series:
                path = f"plan/{j}" if leaf == "plan" else f"joints/{j}/{leaf}"
                self._safe(rr.log, path, rr.SeriesLines(colors=[colour], names=[leaf]), static=True)
        self._sample = sample
        self._thread = threading.Thread(target=self._live, daemon=True, name="viz-live")
        self._thread.start()
        log.info("rerun stream at %s (served by %s)", public, uri)
        return self.urls

    def close(self) -> None:
        """Stop the live thread and disconnect."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        self._safe(self.rr.disconnect)

    # ------------------------------------------------------------------ what the runner calls
    def inference(
        self,
        seq: int,
        t_obs: float,
        images: dict,
        state: np.ndarray,
        chunk: np.ndarray | None,
        action_dt: float,
        accepted: bool | None,
        reason: str = "",
    ) -> None:
        """Log one inference: the frames the policy got, its chunk as a plan, and the verdict.

        `images` are exactly what the policy received (RGB uint8); `chunk` is (H, J) in `joints`
        order. `accepted` is None for a skipped inference.
        """
        rr = self.rr
        self._time(t_obs)
        for key, img in images.items():
            self._image(f"model/input/{short(key)}", img, self.cfg.input_jpeg_quality)
        if chunk is not None:
            times = (t_obs - self.t0) + action_dt * np.arange(len(chunk))
            for k, j in enumerate(self.joints):
                self._safe(
                    rr.send_columns,
                    f"plan/{j}",
                    indexes=[rr.TimeColumn("time", duration=times)],
                    columns=rr.Scalars.columns(scalars=np.asarray(chunk[:, k], dtype=np.float64)),
                )
        verdict = "skipped" if accepted is None else "accepted" if accepted else "REJECTED"
        detail = f": {reason}" if reason and not accepted else ""
        self.event("info" if accepted else "warn", f"chunk {seq} {verdict}{detail}", t_obs)

    def event(self, level: str, text: str, t: float | None = None) -> None:
        """Add a line to the text log at time `t` (time.monotonic(), default now)."""
        self._time(time.monotonic() if t is None else t)
        lvl = {"info": "INFO", "warn": "WARN"}.get(level, "ERROR")
        self._safe(self.rr.log, "log", self.rr.TextLog(text, level=lvl))

    # ------------------------------------------------------------------ internals
    def _blueprint(self):
        """Layout: model inputs, live frames and the log on the left, a joint plot tab per arm."""
        import rerun.blueprint as rrb

        cams = [short(k) for k in self.cameras]
        arms: dict[str, list[str]] = {}
        for j in self.joints:
            arms.setdefault(j.split("_", 1)[0], []).append(j)
        plots = []
        for arm, js in arms.items():
            views = [
                rrb.TimeSeriesView(
                    name=j.split("_", 1)[1], contents=[f"plan/{j}", f"joints/{j}/**"]
                )
                for j in js
            ]
            plots.append(rrb.Grid(*views, name=arm))
        model_views = [
            rrb.Spatial2DView(name=f"model saw: {c}", origin=f"model/input/{c}") for c in cams
        ]
        live_views = [rrb.Spatial2DView(name=f"live: {c}", origin=f"live/{c}") for c in cams]
        return rrb.Blueprint(
            rrb.Horizontal(
                rrb.Vertical(
                    rrb.Horizontal(*model_views),
                    rrb.Horizontal(*live_views),
                    rrb.TextLogView(name="log", origin="log"),
                    row_shares=[3, 3, 2],
                ),
                rrb.Tabs(*plots),
                column_shares=[3, 2],
            ),
            collapse_panels=True,
        )

    def _time(self, t: float) -> None:
        self._safe(self.rr.set_time, "time", duration=max(t - self.t0, 0.0))

    def _image(self, path: str, rgb: np.ndarray | None, quality: int) -> None:
        if rgb is None:
            return
        import cv2

        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
        if ok:
            image = self.rr.EncodedImage(contents=buf.tobytes(), media_type="image/jpeg")
            self._safe(self.rr.log, path, image)

    def _live(self) -> None:
        """Send the newest frames and joint positions at viz.live_hz until close()."""
        rr, period = self.rr, 1.0 / max(self.cfg.live_hz, 0.1)
        nxt = time.monotonic()
        while not self._stop.is_set():
            now = time.monotonic()
            if now < nxt:
                time.sleep(min(nxt - now, 0.02))
                continue
            nxt = max(nxt + period, now - period)
            self._time(now)
            for key, cam in self.cameras.items():
                try:
                    img, _ = cam.latest()
                except Exception:  # noqa: BLE001
                    img = None
                self._image(f"live/{short(key)}", img, self.cfg.jpeg_quality)
            if self._sample is None:
                continue
            try:
                q, goal, _mode, _leader, _seq = self._sample()
            except Exception:  # noqa: BLE001
                continue
            for k, j in enumerate(self.joints):
                if np.isfinite(q[k]):
                    self._safe(rr.log, f"joints/{j}/pos", rr.Scalars(float(q[k])))
                if np.isfinite(goal[k]):
                    self._safe(rr.log, f"joints/{j}/goal", rr.Scalars(float(goal[k])))

    def _safe(self, fn, *a, **kw) -> None:
        """Call `fn`, swallowing its errors so the view never reaches the control loops."""
        try:
            fn(*a, **kw)
        except Exception as exc:  # noqa: BLE001
            if not self._warned:
                self._warned = True
                log.warning(
                    "rerun logging failed (further failures are silent): %s: %s",
                    type(exc).__name__,
                    exc,
                )
