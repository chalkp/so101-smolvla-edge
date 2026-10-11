"""V4L2 camera controls by ioctl: list them and set them by name.

OpenCV reaches only a few controls and reports success for ones the camera ignores, and v4l2-ctl
is not on the Jetson image. A control is named as the driver names it ("Focus, Absolute"), ignoring
case and punctuation. `so101-smolvla-edge cameras CONFIG --controls` lists what each camera has.
Controls can be set while another handle is streaming.
"""

from __future__ import annotations

import fcntl
import os
import struct

VIDIOC_QUERYCTRL = 0xC0445624
VIDIOC_G_CTRL = 0xC008561B
VIDIOC_S_CTRL = 0xC008561C
_NEXT = 0x80000000  # V4L2_CTRL_FLAG_NEXT_CTRL
_QUERY = "II32siiiiI8x"  # struct v4l2_queryctrl
_TYPES = {1: "int", 2: "bool", 3: "menu", 4: "button", 5: "int64"}  # enum v4l2_ctrl_type
_DISABLED = 0x1  # V4L2_CTRL_FLAG_DISABLED
_INACTIVE = 0x10  # V4L2_CTRL_FLAG_INACTIVE
_CLASS = 6  # V4L2_CTRL_TYPE_CTRL_CLASS, a group heading rather than a control


def list_controls(device: str) -> list[dict]:
    """Every control of a video device: id, name, type, min, max, step, default, value, inactive."""
    fd = os.open(os.path.realpath(device), os.O_RDWR)
    try:
        return _list(fd)
    finally:
        os.close(fd)


def _list(fd: int) -> list[dict]:
    """Query every enabled control of an open device, with its current value."""
    out, cid = [], _NEXT
    while True:
        buf = bytearray(struct.pack(_QUERY, cid, 0, b"", 0, 0, 0, 0, 0))
        try:
            fcntl.ioctl(fd, VIDIOC_QUERYCTRL, buf)
        except OSError:
            break
        cid_, typ, name, lo, hi, step, default, flags = struct.unpack(_QUERY, buf)
        cid = cid_ | _NEXT
        if typ == _CLASS or flags & _DISABLED:
            continue
        try:
            reply = fcntl.ioctl(fd, VIDIOC_G_CTRL, struct.pack("Ii", cid_, 0))
            value = struct.unpack("Ii", reply)[1]
        except OSError:
            value = None
        out.append(
            {
                "id": cid_,
                "name": name.split(b"\0")[0].decode(errors="replace"),
                "type": _TYPES.get(typ, str(typ)),
                "min": lo,
                "max": hi,
                "step": step,
                "default": default,
                "value": value,
                "inactive": bool(flags & _INACTIVE),
            }
        )
    return out


def match(name: str, controls: list[dict]) -> dict | None:
    """The control called `name`, ignoring case, spaces and punctuation.

    "focus_absolute" matches "Focus, Absolute".
    """

    def key(s: str) -> str:
        return "".join(ch for ch in s.lower() if ch.isalnum())

    want = key(name)
    return next((c for c in controls if key(c["name"]) == want), None)


def set_controls(device: str, values: dict[str, int]) -> list[str]:
    """Set controls by name, in the order given, and return a message for each one not set.

    Order matters: manual exposure only takes effect once auto exposure is off.
    """
    problems = []
    fd = os.open(os.path.realpath(device), os.O_RDWR)
    try:
        known = _list(fd)
        for name, value in values.items():
            c = match(name, known)
            if c is None:
                problems.append(f"no control '{name}' (has: {', '.join(k['name'] for k in known)})")
                continue
            v = int(value)
            if not c["min"] <= v <= c["max"]:
                problems.append(f"'{c['name']}' = {v} is outside {c['min']}..{c['max']}")
                continue
            try:
                fcntl.ioctl(fd, VIDIOC_S_CTRL, struct.pack("Ii", c["id"], v))
            except OSError as exc:
                problems.append(f"'{c['name']}' = {v} refused: {exc.strerror}")
    finally:
        os.close(fd)
    return problems
