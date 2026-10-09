"""Camera capture threads that always hold the newest frame, timestamped on arrival.

Inference wants the latest image, never a queued one, so each camera has a thread draining the
driver (V4L2 buffer of 1 where supported) and keeping only the most recent RGB frame.
"""

from __future__ import annotations

import logging
import threading
import time

import numpy as np

log = logging.getLogger("so101_smolvla_edge.cameras")


def crop_for_fov(camera_hfov_deg: float, target_hfov_deg: float) -> float:
    """Centre-crop fraction that narrows a pinhole camera of `camera_hfov_deg` to `target_hfov_deg`.

    The result is tan(target / 2) / tan(camera / 2). Above 1 the camera is too narrow, and cropping
    cannot widen it.
    """
    import math

    return math.tan(math.radians(target_hfov_deg) / 2) / math.tan(math.radians(camera_hfov_deg) / 2)


def fit_frame(bgr: np.ndarray, size: tuple[int, int], crop: float = 1.0) -> np.ndarray:
    """Centre-crop to the aspect ratio of `size` (w, h), then resize, so nothing is distorted.

    `crop` is the fraction of the frame width kept (1.0 = as much as the aspect allows), which is
    what narrows the horizontal field of view to the simulator's (`crop_for_fov`). A 16:9 capture
    for a 4:3 output keeps at most 75 % of its width.
    """
    import cv2

    h, w = bgr.shape[:2]
    aspect = size[0] / size[1]
    cw = min(w * crop, h * aspect)  # widest aspect-correct crop no wider than asked
    cw, ch = max(2, round(cw)), max(2, round(cw / aspect))
    if (cw, ch) != (w, h):
        x0, y0 = (w - cw) // 2, (h - ch) // 2
        bgr = bgr[y0 : y0 + ch, x0 : x0 + cw]
    if (bgr.shape[1], bgr.shape[0]) != tuple(size):
        shrink = bgr.shape[1] > size[0]
        bgr = cv2.resize(
            bgr, tuple(size), interpolation=cv2.INTER_AREA if shrink else cv2.INTER_LINEAR
        )
    return bgr


def tone_lut(gain: float = 1.0, gamma: float = 1.0) -> np.ndarray | None:
    """Lookup table for out = 255 * gain * (in / 255) ** gamma, or None if it changes nothing.

    Meant for a camera that overexposes and has no manual exposure control: `gain` puts its clipped
    white at the simulator's table white (225 of 255, so 0.88) and `gamma` above 1 darkens the
    mid-tones that overexposure washed out. Clipped detail stays lost, but the simulator's table
    has almost none, so matching its level is what matters. Fit both on two known surfaces (the
    white table, the yellow arms) against the simulator reference frame.
    """
    if gain == 1.0 and gamma == 1.0:
        return None
    if gain <= 0 or gamma <= 0:
        raise ValueError(f"camera tone: gain and gamma must be positive, got {gain}, {gamma}")
    x = np.arange(256, dtype=np.float64) / 255.0
    return np.clip(np.rint(255.0 * gain * x**gamma), 0, 255).astype(np.uint8)


class Camera:
    """One V4L2 camera and its capture thread; latest() gives the newest frame."""

    def __init__(
        self,
        key: str,
        device: str,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        fourcc: str = "MJPG",
        flip: bool = False,
        crop: float = 1.0,
        capture_width: int | None = None,
        capture_height: int | None = None,
        controls: dict[str, int] | None = None,
        gain: float = 1.0,
        gamma: float = 1.0,
    ):
        if not 0.0 < crop <= 1.0:
            raise ValueError(
                f"camera {key}: crop must be in (0, 1], got {crop} (a camera narrower than the "
                "simulator's view cannot be fixed by cropping)"
            )
        self.key = key
        self.device = device
        self.size = (width, height)  # output frame (w, h)
        self.fps = fps
        self.fourcc = fourcc
        self.flip = flip
        self.crop = crop
        self.capture = (capture_width or width, capture_height or height)
        self.controls = dict(controls or {})
        self._lut = tone_lut(gain, gamma)
        self.control_problems: list[str] = []
        self.native: tuple[int, int] | None = None  # (w, h) the driver actually delivers
        self._frame: np.ndarray | None = None
        self._t = 0.0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.frames = 0
        self.failures = 0

    def start(self) -> Camera:
        """Open the device, apply `controls` and start the capture thread."""
        import cv2

        dev = int(self.device) if str(self.device).isdigit() else self.device
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        if not cap.isOpened():
            raise RuntimeError(f"camera {self.key}: cannot open {self.device}")
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.capture[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.capture[1])
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if self.controls and not isinstance(dev, int):
            from .v4l2 import set_controls

            try:
                self.control_problems = set_controls(dev, self.controls)
            except OSError as exc:
                self.control_problems = [f"cannot set controls: {exc}"]
            # wrong focus or exposure still gives usable frames: warn, do not stop
            for msg in self.control_problems:
                log.warning("camera %s: %s", self.key, msg)
        self._cap = cap
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"cam-{self.key}")
        self._thread.start()
        return self

    def _run(self) -> None:
        """Capture loop: crop, tone-map and convert each frame, keep only the newest."""
        import cv2

        warned = False
        while not self._stop.is_set():
            ok, bgr = self._cap.read()
            t = time.monotonic()
            if not ok:
                self.failures += 1
                time.sleep(0.01)
                continue
            if (bgr.shape[1], bgr.shape[0]) != self.capture and not warned:
                log.warning(
                    "camera %s delivers %dx%d, not the requested %dx%d",
                    self.key,
                    bgr.shape[1],
                    bgr.shape[0],
                    *self.capture,
                )
                warned = True
            self.native = (bgr.shape[1], bgr.shape[0])
            bgr = fit_frame(bgr, self.size, self.crop)
            if self._lut is not None:
                bgr = cv2.LUT(bgr, self._lut)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if self.flip:
                rgb = rgb[::-1, ::-1].copy()
            with self._lock:
                self._frame, self._t = rgb, t
            self.frames += 1

    def latest(self) -> tuple[np.ndarray | None, float]:
        """The newest RGB frame (None before the first) and its time.monotonic() arrival time."""
        with self._lock:
            return self._frame, self._t

    def stop(self) -> None:
        """Stop the capture thread and release the device."""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1)
        if getattr(self, "_cap", None) is not None:
            self._cap.release()


class ReplayCamera:
    """Serves frames from an (n, H, W, 3) uint8 array in a loop, for dry runs and tests."""

    def __init__(self, key: str, frames: np.ndarray, fps: float = 30.0):
        self.key, self.frames_src, self.fps = key, frames, fps
        self._t0 = time.monotonic()
        self.frames = 0
        self.failures = 0

    def start(self) -> ReplayCamera:
        return self

    def latest(self) -> tuple[np.ndarray, float]:
        now = time.monotonic()
        i = int((now - self._t0) * self.fps)
        self.frames = i
        return self.frames_src[i % len(self.frames_src)], self._t0 + i / self.fps

    def stop(self) -> None:
        pass
