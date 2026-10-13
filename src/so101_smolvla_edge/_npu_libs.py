"""Make the Intel NPU visible to OpenVINO when its driver is not installed system-wide.

OpenVINO reaches the NPU through Level Zero: libze_loader (Ubuntu's `libze1`) loads the driver,
libze_intel_npu.so, which loads the graph compiler next to it. Without root, the setup script
(scripts/setup_intel_npu.sh) unpacks Intel's .debs into a directory, and `preload` tells the loader
where the driver is through ZE_ENABLE_ALT_DRIVERS. The loader reads that once, as it initialises,
so it must be set before OpenVINO is imported.

Do not load the unpacked stack's own libze_loader ahead of the system one (by LD_PRELOAD or
ctypes): on the machine this was written on, OpenVINO then finds no NPU at all.
"""

from __future__ import annotations

import ctypes
import importlib.util
import os
import sys
from pathlib import Path

DRIVER = "libze_intel_npu.so.1"  # the Level Zero NPU driver
_done: str | None | bool = False  # False until preload() has run, then what it returned


def _openvino_env() -> Path | None:
    """<env>/npu-stack of the environment openvino is installed in.

    Under `uv run --with ...` that is not sys.prefix (nor $VIRTUAL_ENV): both name a throwaway
    overlay on top of the project's venv.
    """
    try:
        spec = importlib.util.find_spec("openvino")  # finds the package without importing it
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.origin:
        return None
    return Path(spec.origin).parents[4] / "npu-stack"  # <env>/lib/pythonX.Y/site-packages/openvino/


def find_stack() -> Path | None:
    """The directory holding the unpacked driver, or None.

    Looks in $SO101_NPU_STACK, then npu-stack in the environment (where the setup script puts it),
    then ~/.local/share/intel-npu. Each may be the library directory itself or the root the .debs
    were unpacked into.
    """
    roots = [
        os.environ.get("SO101_NPU_STACK"),
        Path(sys.prefix) / "npu-stack",
        _openvino_env(),
        Path.home() / ".local/share/intel-npu",
    ]
    for root in roots:
        if not root:
            continue
        for d in (Path(root), Path(root) / "usr/lib/x86_64-linux-gnu"):
            if (d / DRIVER).exists():
                return d
    return None


def preload() -> str | None:
    """Point Level Zero at an unpacked NPU driver. Call before OpenVINO is imported.

    Returns the driver path it set up, or None when there is nothing to do: a system driver, a
    driver already named in the environment, no driver, or openvino already imported.
    """
    global _done
    if _done is not False:
        return _done
    _done = None
    if os.environ.get("ZE_ENABLE_ALT_DRIVERS") or "openvino" in sys.modules:
        return None
    d = find_stack()
    if d is None:
        return None
    _done = os.environ["ZE_ENABLE_ALT_DRIVERS"] = str(d / DRIVER)
    # the compiler's one dependency outside the stack, often unpacked beside it from libtbb12
    tbb = d / "libtbb.so.12"
    if tbb.exists():
        try:
            ctypes.CDLL(str(tbb), mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass
    return _done
