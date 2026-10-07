"""Make torch importable on Jetson wheels that need libraries pip installs outside the loader path.

The JetPack 6 torch wheels from pypi.jetson-ai-lab.io link libcudss.so.0, which only the
`nvidia-cudss-cu12` pip package provides, in site-packages/nvidia/.../lib and off LD_LIBRARY_PATH.
Loading it RTLD_GLOBAL before `import torch` lets the dynamic loader resolve torch's reference.
Wheels whose torch imports on its own (desktop, JetPack 7's pytorch.org builds) are unaffected.
"""

from __future__ import annotations

import ctypes
import glob
import os
import site
import sys

_done = False  # preload() has run


def preload() -> None:
    """Load the pip-installed CUDA libraries torch links against. Call before torch is imported."""
    global _done
    if _done or "torch" in sys.modules:
        return
    _done = True
    roots = [*site.getsitepackages(), site.getusersitepackages()]
    for name in ("libcudss.so.0",):
        # the first copy that loads wins
        for root in roots:
            for path in glob.glob(os.path.join(root, "nvidia", "**", name), recursive=True):
                try:
                    ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                    break
                except OSError:
                    continue
            else:
                continue
            break
